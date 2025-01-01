# GhostPath

**Predictive Cursor Trajectory Visualizer**

GhostPath is a machine learning-powered application that analyzes your mouse movement patterns in real-time and predicts where your cursor will move next, visualizing these predictions as a "ghost trail" effect.

## Overview

This application captures mouse movements, trains a lightweight ML model on your movement patterns, and displays predicted future cursor positions in real-time. The prediction accuracy improves continuously as the model learns your unique movement behavior.

## Features

### Real-Time Movement Tracking
- Captures mouse coordinates at approximately 30Hz
- Calculates velocity, acceleration, and movement deltas
- Efficient circular buffer for memory management
- Thread-safe data collection

### Machine Learning Prediction
- Lightweight Ridge regression model for trajectory prediction
- Online learning - continuously improves with usage
- Predicts 5-10 positions ahead based on recent patterns
- Physics-based constraints for realistic predictions
- Typical accuracy: 10-20 pixels MAE after training

### Interactive Visualization
- Beautiful fading gradient effect showing predicted positions
- Real-time overlay on movement canvas
- Color-coded accuracy indicators
- Smooth animations without flickering
- Dark theme optimized for extended viewing

### Analytics Dashboard
- Live movement statistics (speed, distance, samples)
- Prediction accuracy tracking over time
- Model training status and metrics
- Interactive graphs and charts

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/abhi3114-glitch/GhostPath.git
cd GhostPath
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

On Windows with Python 3.11:
```bash
py -3.11 -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Using the Interface

1. **Start Tracking**: Click the "Start" button to begin capturing mouse movements
2. **Move Your Mouse**: Move your cursor naturally around the screen
3. **Watch Predictions**: After approximately 10 seconds, the ML model will train and display predictions
4. **Adjust Settings**: Configure lookback window and prediction steps in the sidebar
5. **Reset Data**: Use the "Reset All Data" button to clear collected data and retrain

### Configuration Options

The sidebar provides several configuration options:

- **Lookback Window** (5-20 positions): Number of past positions used for prediction
- **Prediction Steps** (3-10 positions): How many future positions to predict
- **Show Ghost Trail**: Toggle visibility of predicted positions

## How It Works

### Architecture

The application consists of four main components:

1. **Mouse Logger** (`mouse_logger.py`): Captures cursor position and calculates movement features
2. **ML Predictor** (`predictor.py`): Uses Ridge regression to learn movement patterns
3. **Visualizer** (`visualizer.py`): Renders the ghost trail with fade effects
4. **Streamlit App** (`app.py`): Orchestrates components and provides the user interface

### Prediction Process

1. **Capture Phase**: Mouse movements are captured with timestamps and features (position, velocity, acceleration)
2. **Training Phase**: Ridge regression model trains on recent movement history every 2 seconds
3. **Prediction Phase**: Model predicts next 5-10 positions based on recent trajectory
4. **Visualization Phase**: Predictions rendered as fading ghost trail ahead of cursor
5. **Analytics Phase**: Accuracy metrics calculated by comparing predictions with actual movement

### Technical Details

**Movement Logger:**
- Sampling rate: 30Hz (configurable)
- Buffer size: 1000 samples (circular buffer)
- Features: x, y, dx, dy, velocity_x, velocity_y, speed, acceleration

**ML Predictor:**
- Algorithm: Ridge Regression (scikit-learn)
- Training: Online/incremental learning
- Inference time: <10ms per prediction
- Lookback: 10 positions (default)
- Horizon: 5 positions (default)

**Performance:**
- Memory usage: ~50-60MB
- CPU usage: 5-10% during tracking
- Latency: <50ms from movement to visualization

## Project Structure

```
GhostPath/
├── app.py              # Main Streamlit application
├── mouse_logger.py     # Mouse movement capture module
├── predictor.py        # ML prediction engine
├── visualizer.py       # Visualization rendering
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

- **pynput** - Mouse event capture
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning (Ridge regression)
- **streamlit** - Web UI framework
- **pandas** - Data manipulation
- **plotly** - Interactive visualizations
- **pillow** - Image processing

All dependencies are listed in `requirements.txt`.

## Testing

### Component Tests

Test individual components:

```bash
# Test mouse logger
python mouse_logger.py

# Test predictor
python predictor.py

# Test visualizer
python visualizer.py
```

### Expected Behavior

**Initial Usage (0-10 seconds):**
- Movement tracking begins
- Blue trail follows cursor
- Status shows "Not Trained" (data collection phase)

**After Training (10-30 seconds):**
- Model trains automatically
- Ghost trail appears ahead of cursor
- Status shows "Trained" with accuracy metrics

**Steady State (30+ seconds):**
- Prediction accuracy improves to <20px MAE
- Smooth ghost trail anticipates movements
- Accuracy graph shows improvement trend

## Performance Metrics

- **Initial accuracy**: ~30-50px MAE (first predictions)
- **Trained accuracy**: ~15-25px MAE (after 30 seconds)
- **Optimal accuracy**: ~10-20px MAE (after 60 seconds)

Note: Accuracy varies based on movement predictability. Linear movements are more predictable than erratic patterns.

## Troubleshooting

### Python Version Issues

If you encounter NumPy compatibility errors:
- Ensure you're using Python 3.11 or compatible version
- Use `py -3.11` on Windows to specify Python version

### Mouse Permission Issues

On Linux/macOS, you may need elevated permissions:
```bash
sudo python app.py  # Linux
```

On macOS, grant accessibility permissions in System Preferences.

### Ghost Trail Not Appearing

- Wait at least 10 seconds for initial training
- Ensure "Show Ghost Trail" is enabled in settings
- Verify you have collected at least 15-20 movement samples

### Performance Issues

- Reduce lookback window in settings
- Reduce prediction steps
- Close resource-intensive applications

## Privacy and Security

- **Local processing only**: All computation happens on your device
- **No data collection**: Mouse data is never sent anywhere
- **No screenshots**: Only cursor coordinates are tracked
- **No persistent storage**: Data clears when application closes

## Future Enhancements

Potential improvements:
- Export movement data and predictions to CSV
- Multiple ML models (Linear, Polynomial, Neural Networks)
- Movement pattern recognition and gesture detection
- Heatmap of frequently visited screen areas
- Multi-monitor support
- Confidence intervals for predictions
- Movement style classification

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome. Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## Author

Built to explore predictive machine learning applied to cursor dynamics.

## Acknowledgments

- **pynput** team for reliable mouse event capture
- **scikit-learn** for accessible ML tools
- **Streamlit** for rapid UI development
- **Plotly** for interactive visualizations
