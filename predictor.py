"""
Trajectory Predictor
Lightweight ML model for predicting future cursor positions.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.linear_model import Ridge
from collections import deque


class TrajectoryPredictor:
    """
    Predicts future mouse positions based on recent movement patterns.
    Uses a lightweight Ridge regression model with online learning.
    """
    
    def __init__(self, lookback: int = 10, predict_ahead: int = 5):
        """
        Initialize the predictor.
        
        Args:
            lookback: Number of past positions to use for prediction
            predict_ahead: Number of future positions to predict
        """
        self.lookback = lookback
        self.predict_ahead = predict_ahead
        self.model_x = Ridge(alpha=0.1)
        self.model_y = Ridge(alpha=0.1)
        self.is_trained = False
        self.training_history = deque(maxlen=500)
        self.prediction_errors = deque(maxlen=100)
        
    def _extract_features(self, positions: np.ndarray) -> np.ndarray:
        """
        Extract features from position sequence.
        
        Args:
            positions: Array of shape (n, 6) with [x, y, vx, vy, speed, accel]
            
        Returns:
            Feature vector for prediction
        """
        if len(positions) < 2:
            return np.zeros(self.lookback * 6)
        
        # Flatten the positions array
        features = positions[-self.lookback:].flatten()
        
        # Pad if necessary
        if len(features) < self.lookback * 6:
            padding = np.zeros(self.lookback * 6 - len(features))
            features = np.concatenate([padding, features])
            
        return features
    
    def _prepare_training_data(self, movement_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from movement history.
        
        Args:
            movement_data: Array of movement records
            
        Returns:
            Tuple of (X_train, y_train_x, y_train_y)
        """
        if len(movement_data) < self.lookback + 1:
            return None, None, None
        
        X_train = []
        y_train_x = []
        y_train_y = []
        
        # Create sliding windows
        for i in range(len(movement_data) - self.lookback - 1):
            window = movement_data[i:i + self.lookback]
            target = movement_data[i + self.lookback]
            
            features = self._extract_features(window)
            X_train.append(features)
            y_train_x.append(target[0])  # x coordinate
            y_train_y.append(target[1])  # y coordinate
        
        return np.array(X_train), np.array(y_train_x), np.array(y_train_y)
    
    def train(self, movement_data: np.ndarray):
        """
        Train the prediction model on movement data.
        
        Args:
            movement_data: Array of shape (n, 6) with movement features
        """
        if len(movement_data) < self.lookback + 1:
            return
        
        X_train, y_train_x, y_train_y = self._prepare_training_data(movement_data)
        
        if X_train is None or len(X_train) == 0:
            return
        
        # Train models
        self.model_x.fit(X_train, y_train_x)
        self.model_y.fit(X_train, y_train_y)
        self.is_trained = True
        
        # Store training info
        self.training_history.append({
            'timestamp': np.datetime64('now'),
            'samples': len(X_train)
        })
    
    def partial_fit(self, movement_data: np.ndarray):
        """
        Update model with new data (online learning).
        
        Args:
            movement_data: Recent movement data
        """
        if len(movement_data) < self.lookback + 1:
            return
        
        X_train, y_train_x, y_train_y = self._prepare_training_data(movement_data)
        
        if X_train is None or len(X_train) == 0:
            return
        
        if not self.is_trained:
            self.train(movement_data)
        else:
            # Incremental update (retrain on recent data)
            self.model_x.fit(X_train, y_train_x)
            self.model_y.fit(X_train, y_train_y)
    
    def predict(self, recent_positions: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict future positions based on recent movement.
        
        Args:
            recent_positions: Array of recent positions (n, 6)
            
        Returns:
            Array of predicted positions (predict_ahead, 2) or None if not trained
        """
        if not self.is_trained or len(recent_positions) < 2:
            return None
        
        predictions = []
        current_sequence = recent_positions.copy()
        
        # Iteratively predict next positions
        for _ in range(self.predict_ahead):
            features = self._extract_features(current_sequence).reshape(1, -1)
            
            # Predict next position
            pred_x = self.model_x.predict(features)[0]
            pred_y = self.model_y.predict(features)[0]
            
            predictions.append([pred_x, pred_y])
            
            # Update sequence for next prediction
            # Estimate velocity and other features based on prediction
            if len(current_sequence) >= 2:
                last_pos = current_sequence[-1]
                dt = 0.033  # Assume ~30Hz sampling
                
                velocity_x = (pred_x - last_pos[0]) / dt
                velocity_y = (pred_y - last_pos[1]) / dt
                speed = np.sqrt(velocity_x**2 + velocity_y**2)
                
                # Simple acceleration estimate
                if len(current_sequence) >= 2:
                    last_vx = last_pos[2]
                    last_vy = last_pos[3]
                    acceleration = np.sqrt((velocity_x - last_vx)**2 + (velocity_y - last_vy)**2) / dt
                else:
                    acceleration = 0
                
                new_point = np.array([[pred_x, pred_y, velocity_x, velocity_y, speed, acceleration]])
            else:
                new_point = np.array([[pred_x, pred_y, 0, 0, 0, 0]])
            
            current_sequence = np.vstack([current_sequence, new_point])
            current_sequence = current_sequence[-self.lookback:]
        
        return np.array(predictions)
    
    def calculate_accuracy(self, predicted: np.ndarray, actual: np.ndarray) -> dict:
        """
        Calculate prediction accuracy metrics.
        
        Args:
            predicted: Predicted positions (n, 2)
            actual: Actual positions (n, 2)
            
        Returns:
            Dictionary with accuracy metrics
        """
        if predicted is None or actual is None or len(predicted) == 0 or len(actual) == 0:
            return {'mse': 0, 'mae': 0, 'rmse': 0}
        
        # Ensure same length
        min_len = min(len(predicted), len(actual))
        predicted = predicted[:min_len]
        actual = actual[:min_len]
        
        # Calculate errors
        errors = np.linalg.norm(predicted - actual, axis=1)
        mse = np.mean(errors**2)
        mae = np.mean(errors)
        rmse = np.sqrt(mse)
        
        # Store error
        self.prediction_errors.append(mae)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    def get_average_error(self) -> float:
        """Get average prediction error over recent predictions."""
        if not self.prediction_errors:
            return 0.0
        return np.mean(list(self.prediction_errors))
    
    def get_model_info(self) -> dict:
        """Get information about the model state."""
        return {
            'is_trained': self.is_trained,
            'lookback': self.lookback,
            'predict_ahead': self.predict_ahead,
            'training_samples': len(self.training_history),
            'avg_error': self.get_average_error()
        }


if __name__ == "__main__":
    # Test the predictor
    print("Testing TrajectoryPredictor...")
    
    # Generate sample movement data (simulated mouse movement)
    np.random.seed(42)
    n_samples = 100
    t = np.linspace(0, 10, n_samples)
    x = 500 + 200 * np.sin(t)
    y = 400 + 150 * np.cos(t)
    
    # Create movement data with features
    movement_data = []
    for i in range(1, len(x)):
        dt = t[i] - t[i-1]
        vx = (x[i] - x[i-1]) / dt
        vy = (y[i] - y[i-1]) / dt
        speed = np.sqrt(vx**2 + vy**2)
        accel = 0 if i == 1 else speed - movement_data[-1][4]
        
        movement_data.append([x[i], y[i], vx, vy, speed, accel])
    
    movement_data = np.array(movement_data)
    
    # Train predictor
    predictor = TrajectoryPredictor(lookback=10, predict_ahead=5)
    predictor.train(movement_data[:80])
    
    print(f"Model info: {predictor.get_model_info()}")
    
    # Test prediction
    recent = movement_data[70:80]
    predictions = predictor.predict(recent)
    
    if predictions is not None:
        print(f"\nPredicted next {len(predictions)} positions:")
        print(predictions)
        
        # Compare with actual
        actual = movement_data[80:85, :2]
        accuracy = predictor.calculate_accuracy(predictions, actual)
        print(f"\nAccuracy metrics: {accuracy}")
