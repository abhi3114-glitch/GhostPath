"""
Mouse Movement Logger
Captures mouse coordinates, calculates velocity and acceleration in real-time.
"""

import time
import threading
from collections import deque
from typing import Dict, List, Optional
import numpy as np
from pynput import mouse


class MouseLogger:
    """
    Captures mouse movement data with position, velocity, and acceleration.
    Uses a circular buffer for efficient memory management.
    """
    
    def __init__(self, buffer_size: int = 1000, sampling_rate: float = 0.033):
        """
        Initialize the MouseLogger.
        
        Args:
            buffer_size: Maximum number of movement records to keep
            sampling_rate: Time between samples in seconds (default: ~30Hz)
        """
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.data = deque(maxlen=buffer_size)
        self.is_running = False
        self.listener = None
        self.last_position = None
        self.last_time = None
        self.last_velocity = None
        self._lock = threading.Lock()
        
    def _on_move(self, x: int, y: int):
        """
        Callback for mouse movement events.
        
        Args:
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        current_time = time.time()
        
        # Calculate time delta
        if self.last_time is None:
            self.last_time = current_time
            self.last_position = (x, y)
            return
            
        dt = current_time - self.last_time
        
        # Respect sampling rate
        if dt < self.sampling_rate:
            return
            
        # Calculate velocity
        dx = x - self.last_position[0]
        dy = y - self.last_position[1]
        
        velocity_x = dx / dt if dt > 0 else 0
        velocity_y = dy / dt if dt > 0 else 0
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Calculate acceleration
        if self.last_velocity is None:
            acceleration = 0
        else:
            dv_x = velocity_x - self.last_velocity[0]
            dv_y = velocity_y - self.last_velocity[1]
            acceleration = np.sqrt(dv_x**2 + dv_y**2) / dt if dt > 0 else 0
        
        # Store data
        movement_data = {
            'timestamp': current_time,
            'x': x,
            'y': y,
            'dx': dx,
            'dy': dy,
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'speed': speed,
            'acceleration': acceleration
        }
        
        with self._lock:
            self.data.append(movement_data)
        
        # Update last values
        self.last_position = (x, y)
        self.last_time = current_time
        self.last_velocity = (velocity_x, velocity_y)
    
    def start(self):
        """Start capturing mouse movements."""
        if self.is_running:
            return
            
        self.is_running = True
        self.listener = mouse.Listener(on_move=self._on_move)
        self.listener.start()
        
    def stop(self):
        """Stop capturing mouse movements."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.listener:
            self.listener.stop()
            self.listener = None
    
    def get_data(self) -> List[Dict]:
        """
        Get all captured movement data.
        
        Returns:
            List of movement records
        """
        with self._lock:
            return list(self.data)
    
    def get_recent_data(self, n: int = 10) -> List[Dict]:
        """
        Get the most recent N movement records.
        
        Args:
            n: Number of recent records to retrieve
            
        Returns:
            List of recent movement records
        """
        with self._lock:
            data_list = list(self.data)
            return data_list[-n:] if len(data_list) >= n else data_list
    
    def get_data_as_array(self) -> np.ndarray:
        """
        Get movement data as numpy array for ML processing.
        
        Returns:
            Numpy array with shape (n_samples, n_features)
            Features: [x, y, velocity_x, velocity_y, speed, acceleration]
        """
        with self._lock:
            if not self.data:
                return np.array([])
            
            data_list = list(self.data)
            features = []
            for record in data_list:
                features.append([
                    record['x'],
                    record['y'],
                    record['velocity_x'],
                    record['velocity_y'],
                    record['speed'],
                    record['acceleration']
                ])
            return np.array(features)
    
    def clear(self):
        """Clear all captured data."""
        with self._lock:
            self.data.clear()
            self.last_position = None
            self.last_time = None
            self.last_velocity = None
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about captured movements.
        
        Returns:
            Dictionary with movement statistics
        """
        with self._lock:
            if not self.data:
                return {
                    'total_samples': 0,
                    'avg_speed': 0,
                    'max_speed': 0,
                    'avg_acceleration': 0,
                    'total_distance': 0
                }
            
            data_list = list(self.data)
            speeds = [d['speed'] for d in data_list]
            accelerations = [d['acceleration'] for d in data_list]
            
            # Calculate total distance
            total_distance = sum(np.sqrt(d['dx']**2 + d['dy']**2) for d in data_list)
            
            return {
                'total_samples': len(data_list),
                'avg_speed': np.mean(speeds),
                'max_speed': np.max(speeds),
                'avg_acceleration': np.mean(accelerations),
                'total_distance': total_distance
            }


if __name__ == "__main__":
    # Test the logger
    print("Testing MouseLogger...")
    logger = MouseLogger(buffer_size=100)
    logger.start()
    print("Move your mouse for 5 seconds...")
    time.sleep(5)
    logger.stop()
    
    data = logger.get_data()
    stats = logger.get_statistics()
    
    print(f"\nCaptured {len(data)} movement events")
    print(f"Statistics: {stats}")
    
    if data:
        print(f"\nSample record: {data[0]}")
