# IMU-based Speed and Position Tracking


## Project Overview
This project provides a solution for estimating speed and position using Inertial Measurement Unit (IMU) sensors when GPS is unavailable. By leveraging accelerometer and gyroscope data, the system can track movement with reasonable accuracy even in GPS-denied environments.

### Why This Matters
GPS signals are not always available in environments such as:
- Indoor spaces
- Urban canyons (between tall buildings)
- Underground locations
- Areas with signal jamming
- During GPS outages

This project bridges the gap by providing continuous position and speed tracking using the IMU sensors that are present in most modern mobile devices and vehicles.

## Approaches Implemented

### 1. Basic Method
The basic approach uses physics principles to estimate speed and position:
- Low-pass filtering to clean sensor noise
- Gyroscope integration to estimate orientation (roll, pitch, yaw)
- Coordinate transformation from device to world frame
- High-pass filtering to remove acceleration drift
- Double integration of acceleration to get position

**Limitation:** This method suffers from integration drift, where small errors accumulate over time.

### 2. Machine Learning Method
To improve accuracy, a ML model is trained on periods when GPS data is available:
- Feature extraction from IMU sensor data (mean, std, max, min, peak-to-peak values)
- Random Forest Regression model to predict speed
- Position estimation using predicted speed and orientation
- Significantly reduced error compared to the basic method

## Data Format
The system works with CSV files containing the following columns:
```
time, ax, ay, az, wx, wy, wz, Latitude, Longitude, Speed (m/s), Altitude (m)
```
Where:
- `time`: Timestamp in seconds
- `ax, ay, az`: Accelerometer readings in m/s²
- `wx, wy, wz`: Gyroscope readings in rad/s
- `Latitude, Longitude`: GPS coordinates (used for training and validation)
- `Speed (m/s)`: Ground truth speed from GPS (used for training and validation)
- `Altitude (m)`: Elevation above sea level

## Results
Based on our test dataset:
- Basic Method: 2.54% relative error in distance estimation
- ML Method: 0.22% relative error in distance estimation

## How to Use

### Installation
```bash
git clone https://github.com/yourusername/imu-tracking.git
cd imu-tracking
pip install -r requirements.txt
```

### Training a Model
```python
from ml_model import process_ml_model
import pandas as pd

# Load your data
df = pd.read_csv("your_training_data.csv")

# Train the model
results = process_ml_model(df)

# Save the model
from master_analysis import save_model
save_model(results['model'], results['scaler'])
```

### Using the Trained Model
```python
from master_analysis import load_model, test_model
import pandas as pd

# Load new data
df = pd.read_csv("new_data.csv")

# Load your saved model
model, scaler = load_model()

# Make predictions
predictions, features_df = test_model(df, model, scaler)
```

### Analysis and Visualization
```python
from master_analysis import compare_methods

# Compare basic and ML methods and generate report
compare_methods("your_data.csv")
```

## Project Structure
- `basic_methodv2.py`: Implements the physics-based approach for speed and position estimation
- `ml_model.py`: Contains code for feature extraction and machine learning model training
- `master_analysis.py`: Combines and compares both methods, generates analysis reports

## Requirements
- Python 3.6+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- reportlab (for PDF report generation)

## Future Improvements
- Implementation of sensor fusion techniques like Kalman filtering
- Support for additional sensors (magnetometer, barometer)
- Real-time processing capabilities
- Mobile application integration
