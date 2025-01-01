"""
GhostPath — Predictive Cursor Trajectory Visualizer
Main Streamlit Application
"""

import streamlit as st
import time
import numpy as np
from collections import deque
import plotly.graph_objects as go

from mouse_logger import MouseLogger
from predictor import TrajectoryPredictor
from visualizer import TrajectoryVisualizer


# Page configuration
st.set_page_config(
    page_title="GhostPath — Cursor Trajectory Predictor",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0a1e 0%, #1a1a3e 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    h1 {
        color: #64b5f6;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0;
    }
    h2 {
        color: #81c784;
    }
    h3 {
        color: #ffb74d;
    }
    .subtitle {
        text-align: center;
        color: #b0b0b0;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logger' not in st.session_state:
    st.session_state.logger = MouseLogger(buffer_size=1000, sampling_rate=0.033)
    st.session_state.predictor = TrajectoryPredictor(lookback=10, predict_ahead=5)
    st.session_state.visualizer = TrajectoryVisualizer(canvas_width=900, canvas_height=600)
    st.session_state.is_tracking = False
    st.session_state.last_train_time = 0
    st.session_state.error_history = deque(maxlen=100)
    st.session_state.show_predictions = True
    st.session_state.last_prediction = None
    st.session_state.last_position = None

# Header
st.markdown("<h1>👻 GhostPath</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predictive Cursor Trajectory Visualizer</p>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("## 🎮 Controls")
    
    # Start/Stop tracking
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", key="start_btn", use_container_width=True):
            if not st.session_state.is_tracking:
                st.session_state.logger.start()
                st.session_state.is_tracking = True
                st.success("Tracking started!")
    
    with col2:
        if st.button("⏸️ Stop", key="stop_btn", use_container_width=True):
            if st.session_state.is_tracking:
                st.session_state.logger.stop()
                st.session_state.is_tracking = False
                st.info("Tracking stopped")
    
    # Reset button
    if st.button("🔄 Reset All Data", key="reset_btn", use_container_width=True):
        st.session_state.logger.clear()
        st.session_state.predictor = TrajectoryPredictor(lookback=10, predict_ahead=5)
        st.session_state.error_history.clear()
        st.session_state.last_prediction = None
        st.warning("All data cleared!")
    
    st.markdown("---")
    
    # Settings
    st.markdown("## ⚙️ Settings")
    
    st.session_state.show_predictions = st.checkbox(
        "Show Ghost Trail",
        value=st.session_state.show_predictions,
        help="Toggle prediction visualization"
    )
    
    lookback = st.slider(
        "Lookback Window",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of past positions to analyze"
    )
    
    predict_ahead = st.slider(
        "Prediction Steps",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of future positions to predict"
    )
    
    # Update predictor settings if changed
    if (st.session_state.predictor.lookback != lookback or 
        st.session_state.predictor.predict_ahead != predict_ahead):
        st.session_state.predictor = TrajectoryPredictor(
            lookback=lookback,
            predict_ahead=predict_ahead
        )
    
    st.markdown("---")
    
    # Status indicator
    st.markdown("## 📊 Status")
    status_color = "🟢" if st.session_state.is_tracking else "🔴"
    status_text = "ACTIVE" if st.session_state.is_tracking else "IDLE"
    st.markdown(f"### {status_color} {status_text}")

# Main content area
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("## 🎯 Live Trajectory View")
    
    # Visualization placeholder
    viz_placeholder = st.empty()
    
    # Get current data
    data = st.session_state.logger.get_data()
    
    if len(data) > 0:
        # Get recent trail
        recent_data = st.session_state.logger.get_recent_data(n=30)
        trail = [(d['x'], d['y']) for d in recent_data]
        
        # Current position
        current_pos = (data[-1]['x'], data[-1]['y']) if data else None
        st.session_state.last_position = current_pos
        
        # Train model periodically
        current_time = time.time()
        if current_time - st.session_state.last_train_time > 2.0:  # Train every 2 seconds
            movement_array = st.session_state.logger.get_data_as_array()
            if len(movement_array) > 15:
                st.session_state.predictor.partial_fit(movement_array)
                st.session_state.last_train_time = current_time
        
        # Make predictions
        predictions = None
        if st.session_state.show_predictions and st.session_state.predictor.is_trained:
            recent_array = st.session_state.logger.get_data_as_array()
            if len(recent_array) >= 10:
                predictions = st.session_state.predictor.predict(recent_array[-lookback:])
                st.session_state.last_prediction = predictions
                
                # Calculate accuracy if we have previous predictions
                if predictions is not None and len(recent_array) > predict_ahead:
                    actual = recent_array[-predict_ahead:, :2]
                    accuracy = st.session_state.predictor.calculate_accuracy(predictions, actual)
                    st.session_state.error_history.append(accuracy['mae'])
        
        # Create visualization
        fig = st.session_state.visualizer.create_visualization(
            current_position=current_pos,
            recent_trail=trail,
            predicted_positions=predictions
        )
        
        viz_placeholder.plotly_chart(fig, use_container_width=True, key=f"viz_{time.time()}")
    else:
        # Show placeholder
        st.info("👆 Click **Start** to begin tracking mouse movements")
        
        # Create empty visualization
        fig = go.Figure()
        fig.update_layout(
            width=900,
            height=600,
            xaxis=dict(range=[0, 1920], showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
            yaxis=dict(range=[0, 1080], showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
            plot_bgcolor='rgba(10, 10, 30, 1)',
            paper_bgcolor='rgba(10, 10, 30, 1)',
        )
        viz_placeholder.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("## 📈 Analytics")
    
    # Movement statistics
    stats = st.session_state.logger.get_statistics()
    stats_html = st.session_state.visualizer.create_statistics_display(stats)
    st.markdown(stats_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model information
    model_info = st.session_state.predictor.get_model_info()
    avg_error = st.session_state.predictor.get_average_error()
    model_html = st.session_state.visualizer.create_model_info_display(model_info, avg_error)
    st.markdown(model_html, unsafe_allow_html=True)

# Accuracy chart (full width below)
if len(st.session_state.error_history) > 0:
    st.markdown("---")
    st.markdown("## 📉 Prediction Accuracy Over Time")
    
    error_list = list(st.session_state.error_history)
    accuracy_fig = st.session_state.visualizer.create_accuracy_chart([], error_list)
    st.plotly_chart(accuracy_fig, use_container_width=True)

# Auto-refresh for live updates
if st.session_state.is_tracking:
    time.sleep(0.1)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>GhostPath v1.0 — Predictive Cursor Analysis</p>",
    unsafe_allow_html=True
)
