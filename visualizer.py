"""
Trajectory Visualizer
Renders ghost trail and movement visualization.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Tuple
from collections import deque


class TrajectoryVisualizer:
    """
    Visualizes mouse movement and predicted trajectory with ghost trail effect.
    """
    
    def __init__(self, canvas_width: int = 800, canvas_height: int = 600):
        """
        Initialize the visualizer.
        
        Args:
            canvas_width: Width of visualization canvas
            canvas_height: Height of visualization canvas
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.trail_length = 30  # Number of past positions to show
        
    def create_visualization(
        self,
        current_position: Optional[Tuple[float, float]],
        recent_trail: List[Tuple[float, float]],
        predicted_positions: Optional[np.ndarray],
        accuracy_data: Optional[List[float]] = None
    ) -> go.Figure:
        """
        Create a plotly figure with movement visualization.
        
        Args:
            current_position: Current mouse position (x, y)
            recent_trail: List of recent positions for trail effect
            predicted_positions: Predicted future positions array (n, 2)
            accuracy_data: Optional list of accuracy metrics over time
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Set canvas dimensions
        fig.update_layout(
            width=self.canvas_width,
            height=self.canvas_height,
            xaxis=dict(
                range=[0, 1920],  # Typical screen width
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=False
            ),
            yaxis=dict(
                range=[0, 1080],  # Typical screen height
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='rgba(10, 10, 30, 1)',
            paper_bgcolor='rgba(10, 10, 30, 1)',
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(30, 30, 50, 0.8)',
                bordercolor='rgba(100, 100, 150, 0.5)',
                borderwidth=1,
                font=dict(color='white')
            ),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        # Plot recent trail (fading effect)
        if recent_trail and len(recent_trail) > 1:
            trail_x = [pos[0] for pos in recent_trail]
            trail_y = [pos[1] for pos in recent_trail]
            
            # Create fading trail effect
            n_points = len(trail_x)
            for i in range(n_points - 1):
                alpha = (i + 1) / n_points  # Fade from 0 to 1
                
                fig.add_trace(go.Scatter(
                    x=[trail_x[i], trail_x[i+1]],
                    y=[trail_y[i], trail_y[i+1]],
                    mode='lines',
                    line=dict(
                        color=f'rgba(100, 200, 255, {alpha * 0.6})',
                        width=2 + alpha * 2
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Plot current position
        if current_position:
            fig.add_trace(go.Scatter(
                x=[current_position[0]],
                y=[current_position[1]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='cyan',
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                name='Current',
                hovertemplate='<b>Current Position</b><br>X: %{x:.0f}<br>Y: %{y:.0f}<extra></extra>'
            ))
        
        # Plot predicted ghost trail
        if predicted_positions is not None and len(predicted_positions) > 0:
            pred_x = predicted_positions[:, 0]
            pred_y = predicted_positions[:, 1]
            
            n_pred = len(pred_x)
            
            # Create fading ghost trail
            for i in range(n_pred - 1):
                alpha = 1 - (i / n_pred)  # Fade from 1 to 0
                
                fig.add_trace(go.Scatter(
                    x=[pred_x[i], pred_x[i+1]],
                    y=[pred_y[i], pred_y[i+1]],
                    mode='lines',
                    line=dict(
                        color=f'rgba(255, 100, 255, {alpha * 0.7})',
                        width=3,
                        dash='dot'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add ghost markers
            fig.add_trace(go.Scatter(
                x=pred_x,
                y=pred_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color='magenta',
                    symbol='circle-open',
                    line=dict(width=2)
                ),
                name='Ghost Trail',
                hovertemplate='<b>Prediction</b><br>X: %{x:.0f}<br>Y: %{y:.0f}<extra></extra>'
            ))
        
        return fig
    
    def create_accuracy_chart(self, accuracy_history: List[float], error_history: List[float]) -> go.Figure:
        """
        Create accuracy metrics chart.
        
        Args:
            accuracy_history: List of timestamp-accuracy pairs
            error_history: List of error values over time
            
        Returns:
            Plotly figure with accuracy metrics
        """
        fig = go.Figure()
        
        if error_history and len(error_history) > 0:
            x_vals = list(range(len(error_history)))
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=error_history,
                mode='lines',
                line=dict(color='rgb(100, 200, 255)', width=2),
                fill='tozeroy',
                fillcolor='rgba(100, 200, 255, 0.2)',
                name='Prediction Error'
            ))
        
        fig.update_layout(
            title=dict(
                text='Prediction Accuracy Over Time',
                font=dict(color='white', size=16)
            ),
            xaxis=dict(
                title='Sample',
                gridcolor='rgba(128, 128, 128, 0.2)',
                color='white'
            ),
            yaxis=dict(
                title='Error (pixels)',
                gridcolor='rgba(128, 128, 128, 0.2)',
                color='white'
            ),
            plot_bgcolor='rgba(20, 20, 40, 1)',
            paper_bgcolor='rgba(20, 20, 40, 1)',
            height=300,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        
        return fig
    
    def create_statistics_display(self, stats: dict) -> str:
        """
        Create formatted statistics display.
        
        Args:
            stats: Dictionary with statistics
            
        Returns:
            Formatted HTML string
        """
        html = f"""
        <div style='background: linear-gradient(135deg, #1a1a3e 0%, #2a2a5e 100%); 
                    padding: 20px; 
                    border-radius: 10px; 
                    color: white;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);'>
            <h3 style='margin-top: 0; color: #64b5f6;'>📊 Movement Statistics</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                <div style='background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;'>
                    <div style='font-size: 12px; color: #b0b0b0;'>Total Samples</div>
                    <div style='font-size: 24px; font-weight: bold; color: #64b5f6;'>{stats.get('total_samples', 0)}</div>
                </div>
                <div style='background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;'>
                    <div style='font-size: 12px; color: #b0b0b0;'>Avg Speed</div>
                    <div style='font-size: 24px; font-weight: bold; color: #81c784;'>{stats.get('avg_speed', 0):.1f} px/s</div>
                </div>
                <div style='background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;'>
                    <div style='font-size: 12px; color: #b0b0b0;'>Max Speed</div>
                    <div style='font-size: 24px; font-weight: bold; color: #ffb74d;'>{stats.get('max_speed', 0):.1f} px/s</div>
                </div>
                <div style='background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px;'>
                    <div style='font-size: 12px; color: #b0b0b0;'>Total Distance</div>
                    <div style='font-size: 24px; font-weight: bold; color: #ba68c8;'>{stats.get('total_distance', 0):.0f} px</div>
                </div>
            </div>
        </div>
        """
        return html
    
    def create_model_info_display(self, model_info: dict, avg_error: float) -> str:
        """
        Create formatted model information display.
        
        Args:
            model_info: Dictionary with model information
            avg_error: Average prediction error
            
        Returns:
            Formatted HTML string
        """
        status_color = '#81c784' if model_info.get('is_trained', False) else '#e57373'
        status_text = '✓ Trained' if model_info.get('is_trained', False) else '✗ Not Trained'
        
        html = f"""
        <div style='background: linear-gradient(135deg, #1a3e1a 0%, #2a5e2a 100%); 
                    padding: 20px; 
                    border-radius: 10px; 
                    color: white;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);'>
            <h3 style='margin-top: 0; color: #81c784;'>🤖 Model Status</h3>
            <div style='margin-bottom: 15px;'>
                <span style='background: {status_color}; 
                             padding: 5px 15px; 
                             border-radius: 20px; 
                             font-weight: bold;'>{status_text}</span>
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;'>
                <div>
                    <div style='font-size: 12px; color: #b0b0b0;'>Training Samples</div>
                    <div style='font-size: 20px; font-weight: bold;'>{model_info.get('training_samples', 0)}</div>
                </div>
                <div>
                    <div style='font-size: 12px; color: #b0b0b0;'>Avg Error</div>
                    <div style='font-size: 20px; font-weight: bold;'>{avg_error:.1f} px</div>
                </div>
            </div>
        </div>
        """
        return html


if __name__ == "__main__":
    # Test the visualizer
    print("Testing TrajectoryVisualizer...")
    
    visualizer = TrajectoryVisualizer()
    
    # Create sample data
    current_pos = (500, 400)
    trail = [(400 + i*10, 350 + i*5) for i in range(20)]
    predictions = np.array([[520 + i*15, 410 + i*10] for i in range(5)])
    
    # Create visualization
    fig = visualizer.create_visualization(current_pos, trail, predictions)
    print("Visualization created successfully!")
    
    # Create stats display
    stats = {
        'total_samples': 150,
        'avg_speed': 245.5,
        'max_speed': 512.3,
        'total_distance': 15432
    }
    stats_html = visualizer.create_statistics_display(stats)
    print("\nStats display created!")
