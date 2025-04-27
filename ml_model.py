import numpy as np
import pandas as pd
import math
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_features(df, fs=150, window_size=1.0):
    """Extract features from raw sensor data"""
    # Apply filtering
    df['time_sec'] = df['time']
    filtered_data = apply_filters(df, fs)
    
    # Segment data
    segments = segment_data(filtered_data, window_size, fs)
    
    # Extract features
    feature_list = []
    label_list = []
    time_list = []
    segment_durations = []
    
    for seg in segments:
        feats = extract_features_from_segment(seg)
        feature_list.append(feats)
        label_list.append(seg['Speed (m/s)'].mean())
        time_list.append(seg['time_sec'].iloc[len(seg)//2])
        segment_durations.append(seg['time_sec'].iloc[-1] - seg['time_sec'].iloc[0])
    
    features_df = pd.DataFrame(feature_list)
    features_df['label'] = label_list
    features_df['time'] = time_list
    features_df['duration'] = segment_durations
    
    return features_df

def train_model_pipeline(features_df):
    """Train the ML model and return model and scaler"""
    X = features_df.drop(columns=['label', 'time', 'duration'])
    y = features_df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    return grid_search.best_estimator_, scaler

def process_ml_model(df):
    """Process data using ML model and return all results with intermediate values"""
    # Apply filters and compute orientations
    df = df.copy()
    df['time_diff'] = df['time'].diff().fillna(0)
    filtered_data = apply_filters(df, fs=150)
    
    # Compute orientations
    roll = np.zeros(len(df))
    pitch = np.zeros(len(df))
    yaw = np.zeros(len(df))
    
    for i in range(1, len(df)):
        dt = df['time_diff'].iloc[i]
        roll[i] = roll[i-1] + filtered_data['wx_filtered'].iloc[i] * dt
        pitch[i] = pitch[i-1] + filtered_data['wy_filtered'].iloc[i] * dt
        yaw[i] = yaw[i-1] + filtered_data['wz_filtered'].iloc[i] * dt
    
    filtered_data['roll'] = roll
    filtered_data['pitch'] = pitch
    filtered_data['yaw'] = yaw
    
    # Extract features and train model
    features_df = extract_features(filtered_data)
    X = features_df.drop(columns=['label', 'time', 'duration'])
    model, scaler = train_model_pipeline(features_df)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    
    # Pass orientation data to features_df
    features_df['roll'] = np.interp(features_df['time'], filtered_data['time'], roll)
    features_df['pitch'] = np.interp(features_df['time'], filtered_data['time'], pitch)
    features_df['yaw'] = np.interp(features_df['time'], filtered_data['time'], yaw)
    
    return {
        'predictions': predictions,
        'actual_speed': features_df['label'].values,
        'time': features_df['time'].values,
        'distance': (predictions * features_df['duration'].values).cumsum(),
        'model': model,
        'scaler': scaler,
        'features_df': features_df,
        'filtered_data': filtered_data
    }

# Helper functions
def apply_filters(df, fs):
    """Apply Butterworth filters to sensor data"""
    def lowpass_filter(data, cutoff=5, fs=150, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    df['ax_filtered'] = lowpass_filter(df['ax'].values, cutoff=5, fs=fs)
    df['ay_filtered'] = lowpass_filter(df['ay'].values, cutoff=5, fs=fs)
    df['az_filtered'] = lowpass_filter(df['az'].values, cutoff=5, fs=fs)
    df['wx_filtered'] = lowpass_filter(df['wx'].values, cutoff=5, fs=fs)
    df['wy_filtered'] = lowpass_filter(df['wy'].values, cutoff=5, fs=fs)
    df['wz_filtered'] = lowpass_filter(df['wz'].values, cutoff=5, fs=fs)
    return df

def segment_data(df, window_size, fs):
    """Segment the DataFrame into windows"""
    samples_per_window = int(window_size * fs)
    segments = []
    for start in range(0, len(df), samples_per_window):
        end = start + samples_per_window
        if end <= len(df):
            segments.append(df.iloc[start:end])
    return segments

def extract_features_from_segment(segment):
    """Extract features from a single segment"""
    features = {}
    features.update(extract_acc_features(segment))
    features.update(extract_gyro_features(segment))
    return features

def extract_acc_features(segment):
    """
    Extract time-domain features from the filtered accelerometer data.
    Uses ax_filtered and ay_filtered as example features.
    """
    features = {}
    # For ax
    acc_x = segment['ax_filtered'].values
    features['ax_mean'] = np.mean(acc_x)
    features['ax_std'] = np.std(acc_x)
    features['ax_max'] = np.max(acc_x)
    features['ax_min'] = np.min(acc_x)
    features['ax_ptp'] = np.ptp(acc_x)  # peak-to-peak amplitude

    # For ay
    acc_y = segment['ay_filtered'].values
    features['ay_mean'] = np.mean(acc_y)
    features['ay_std'] = np.std(acc_y)
    features['ay_max'] = np.max(acc_y)
    features['ay_min'] = np.min(acc_y)
    features['ay_ptp'] = np.ptp(acc_y)
    
    return features

def extract_gyro_features(segment):
    """
    Extract time-domain features from the filtered gyroscope data.
    Uses wx_filtered, wy_filtered, wz_filtered.
    """
    features = {}
    # For wx
    gyro_x = segment['wx_filtered'].values
    features['wx_mean'] = np.mean(gyro_x)
    features['wx_std'] = np.std(gyro_x)
    features['wx_max'] = np.max(gyro_x)
    features['wx_min'] = np.min(gyro_x)
    
    # For wy
    gyro_y = segment['wy_filtered'].values
    features['wy_mean'] = np.mean(gyro_y)
    features['wy_std'] = np.std(gyro_y)
    features['wy_max'] = np.max(gyro_y)
    features['wy_min'] = np.min(gyro_y)
    
    # For wz
    gyro_z = segment['wz_filtered'].values
    features['wz_mean'] = np.mean(gyro_z)
    features['wz_std'] = np.std(gyro_z)
    features['wz_max'] = np.max(gyro_z)
    features['wz_min'] = np.min(gyro_z)
    
    return features

if __name__ == "__main__":
    # -----------------------------
    # PART 1: Data Loading and Pre-Processing
    # -----------------------------
    # Load CSV data (adjust file path as needed)
    file_path = "round_around_campus.csv"  # Replace with your actual file path
    df = pd.read_csv(file_path)

    # Ensure proper column names: 
    # 'time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Latitude', 'Longitude', 'Speed (m/s)', 'Altitude (m)'
    print("Columns:", df.columns)

    # Convert time column to seconds (assuming 'time' is in seconds already, if not, adjust accordingly)
    # Here we assume 'time' is already numeric.
    df['time_sec'] = df['time']

    # -----------------------------
    # PART 2: Segmentation and Feature Extraction
    # -----------------------------
    features_df = extract_features(df)

    # -----------------------------
    # PART 3: Feature Normalization and Train-Test Split
    # -----------------------------
    # Separate features and target
    X = features_df.drop(columns=['label', 'time', 'duration'])
    y = features_df['label']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test, time_train, time_test, duration_train, duration_test = train_test_split(
        X_scaled, 
        y, 
        features_df['time'], 
        features_df['duration'],
        test_size=0.2, 
        random_state=42
    )

    # -----------------------------
    # PART 4: Regression Model Training with GridSearchCV
    # -----------------------------
    # We'll use RandomForestRegressor and perform grid search to tune hyperparameters.
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=rf,
                            param_grid=param_grid,
                            scoring='neg_mean_squared_error',
                            cv=5,
                            n_jobs=-1,
                            verbose=1)

    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    print("Best CV MSE:", -grid_search.best_score_)

    # Evaluate on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(f"Test Mean Squared Error: {test_mse:.3f}")

    # -----------------------------
    # PART 5: Plot Predictions vs. Actual Labels
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Speed (m/s)")
    plt.ylabel("Predicted Speed (m/s)")
    plt.title("Actual vs Predicted Speed")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.grid(True)
    plt.show()

    # -----------------------------
    # PART 6: Distance Calculation
    # -----------------------------

    # Create a DataFrame to store test results with time information
    test_results = pd.DataFrame({
        'time': time_test.values,
        'duration': duration_test.values,
        'actual_speed': y_test.values,
        'predicted_speed': y_pred
    })

    # Sort by time to ensure chronological order
    test_results = test_results.sort_values('time').reset_index(drop=True)

    # Calculate distance for each segment (speed * duration)
    test_results['actual_distance_segment'] = test_results['actual_speed'] * test_results['duration']
    test_results['predicted_distance_segment'] = test_results['predicted_speed'] * test_results['duration']

    # Calculate cumulative distance
    test_results['actual_distance_cumulative'] = test_results['actual_distance_segment'].cumsum()
    test_results['predicted_distance_cumulative'] = test_results['predicted_distance_segment'].cumsum()

    # Calculate total distance
    total_actual_distance = test_results['actual_distance_segment'].sum()
    total_predicted_distance = test_results['predicted_distance_segment'].sum()

    print(f"Total Actual Distance (GPS): {total_actual_distance:.2f} meters")
    print(f"Total Predicted Distance (IMU): {total_predicted_distance:.2f} meters")
    print(f"Distance Error: {abs(total_actual_distance - total_predicted_distance):.2f} meters ({abs(total_actual_distance - total_predicted_distance)/total_actual_distance*100:.2f}%)")

    # Plot cumulative distance comparison
    plt.figure(figsize=(10, 6))
    plt.plot(test_results['time'], test_results['actual_distance_cumulative'], 'b-', label='Actual Distance (GPS)')
    plt.plot(test_results['time'], test_results['predicted_distance_cumulative'], 'r--', label='Predicted Distance (IMU)')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Distance (m)')
    plt.title('Cumulative Distance: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot speed vs time
    plt.figure(figsize=(10, 6))
    plt.plot(test_results['time'], test_results['actual_speed'], 'b-', label='Actual Speed (GPS)')
    plt.plot(test_results['time'], test_results['predicted_speed'], 'r--', label='Predicted Speed (IMU)')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed over Time: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate distance for all available data (training + test)
    # Apply the model to all data
    X_all = scaler.transform(features_df.drop(columns=['label', 'time', 'duration']))
    all_predictions = best_model.predict(X_all)

    all_results = pd.DataFrame({
        'time': features_df['time'].values,
        'duration': features_df['duration'].values,
        'actual_speed': features_df['label'].values,
        'predicted_speed': all_predictions
    })

    # Sort by time
    all_results = all_results.sort_values('time').reset_index(drop=True)

    # Calculate distance for each segment
    all_results['actual_distance_segment'] = all_results['actual_speed'] * all_results['duration']
    all_results['predicted_distance_segment'] = all_results['predicted_speed'] * all_results['duration']

    # Calculate cumulative distance
    all_results['actual_distance_cumulative'] = all_results['actual_distance_segment'].cumsum()
    all_results['predicted_distance_cumulative'] = all_results['predicted_distance_segment'].cumsum()

    # Calculate total distance for entire dataset
    total_actual_distance_all = all_results['actual_distance_segment'].sum()
    total_predicted_distance_all = all_results['predicted_distance_segment'].sum()

    print("\nDistance calculations for entire dataset:")
    print(f"Total Actual Distance (GPS): {total_actual_distance_all:.2f} meters")
    print(f"Total Predicted Distance (IMU): {total_predicted_distance_all:.2f} meters")
    print(f"Distance Error: {abs(total_actual_distance_all - total_predicted_distance_all):.2f} meters ({abs(total_actual_distance_all - total_predicted_distance_all)/total_actual_distance_all*100:.2f}%)")

    # Plot cumulative distance comparison for all data
    plt.figure(figsize=(10, 6))
    plt.plot(all_results['time'], all_results['actual_distance_cumulative'], 'b-', label='Actual Distance (GPS)')
    plt.plot(all_results['time'], all_results['predicted_distance_cumulative'], 'r--', label='Predicted Distance (IMU)')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Distance (m)')
    plt.title('Cumulative Distance (All Data): Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()