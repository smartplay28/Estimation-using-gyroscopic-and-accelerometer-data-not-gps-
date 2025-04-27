import numpy as np
import pandas as pd
import joblib
import os
from ml_model import process_ml_model, extract_features
from basic_methodv2 import process_basic_method
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from sklearn.metrics import mean_squared_error
import io
from reportlab.lib.enums import TA_CENTER
from scipy.signal import butter, filtfilt

def save_model(model, scaler, filepath='model/'):
    """Save the trained model and scaler"""
    os.makedirs(filepath, exist_ok=True)
    joblib.dump(model, filepath + 'speed_predictor.joblib')
    joblib.dump(scaler, filepath + 'scaler.joblib')
    print(f"Model saved to {filepath}")

def load_model(filepath='model/'):
    """Load the trained model and scaler"""
    model = joblib.load(filepath + 'speed_predictor.joblib')
    scaler = joblib.load(filepath + 'scaler.joblib')
    return model, scaler

def train_model(data_path):
    """Train the model and save it"""
    df = pd.read_csv(data_path)
    results = process_ml_model(df)
    save_model(results['model'], results['scaler'])
    return results['model'], results['scaler']

def test_model(data_path, model, scaler):
    """Test the model on new data"""
    df = pd.read_csv(data_path)
    features_df = extract_features(df)
    X = features_df.drop(columns=['label', 'time', 'duration'])
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions, features_df

def ensure_yaw_column(df):
    """Ensure the DataFrame has a 'yaw' column, computing it if necessary."""
    if 'yaw' not in df.columns:
        # Try to compute yaw from gyroscope data if possible
        required_cols = {'wx', 'wy', 'wz', 'time'}
        if required_cols.issubset(df.columns):
            df = df.copy()
            df['time_diff'] = df['time'].diff().fillna(0)
            yaw = np.zeros(len(df))
            for i in range(1, len(df)):
                dt = df['time_diff'].iloc[i]
                yaw[i] = yaw[i-1] + df['wz'].iloc[i] * dt
            df['yaw'] = yaw
        else:
            raise KeyError("Column 'yaw' not found and cannot be computed (missing wx, wy, wz, or time).")
    return df

def calculate_gps_path(df):
    """Calculate GPS-based path from speed and heading (yaw)."""
    df = ensure_yaw_column(df)
    df = df.copy()
    if 'Speed (m/s)' not in df.columns:
        raise KeyError("Column 'Speed (m/s)' not found in DataFrame.")
    df['time_diff'] = df['time'].diff().fillna(0)
    df["displacement"] = df["Speed (m/s)"] * df["time_diff"]
    df["dx_gps"] = df["displacement"] * np.cos(df["yaw"])
    df["dy_gps"] = df["displacement"] * np.sin(df["yaw"])
    df["x_pos_gps"] = np.cumsum(df["dx_gps"])
    df["y_pos_gps"] = np.cumsum(df["dy_gps"])
    return df

def calculate_ml_path(features_df, ml_predictions):
    """Calculate ML-based path using predicted velocities and orientation"""
    if 'yaw' not in features_df.columns:
        print("Warning: No yaw data in features_df. Path may be inaccurate.")
        yaw = np.zeros(len(features_df))
    else:
        yaw = features_df['yaw'].values

    dx_ml = ml_predictions * np.cos(yaw) * features_df['duration'].values
    dy_ml = ml_predictions * np.sin(yaw) * features_df['duration'].values
    x_pos_ml = np.cumsum(dx_ml)
    y_pos_ml = np.cumsum(dy_ml)
    return x_pos_ml, y_pos_ml

def save_plot(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def generate_all_plots(df, features_df, ml_predictions, basic_results, output_prefix="report_fig"):
    """Generate and save all relevant plots, return their filenames."""
    plot_files = []

    # 1. Raw and filtered accelerometer magnitude
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(df['time'], np.sqrt(df['ax']**2 + df['ay']**2), 'b-')
    axs[0].set_title('Raw Accelerometer Magnitude')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Accel (m/s²)')
    axs[0].grid(True)
    axs[1].plot(df['time'], np.sqrt(df['ax_filtered']**2 + df['ay_filtered']**2), 'r-')
    axs[1].set_title('Filtered Accelerometer Magnitude')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Accel (m/s²)')
    axs[1].grid(True)
    plt.tight_layout()
    fname1 = f"{output_prefix}_accel.png"
    save_plot(fig, fname1)
    plot_files.append((fname1, "Raw and Filtered Accelerometer Magnitude"))

    # 2. Speed comparison
    fig = plt.figure(figsize=(10, 5))
    plt.plot(features_df['time'], features_df['label'], 'g-', label='GPS Ground Truth')
    plt.plot(features_df['time'], ml_predictions, 'b--', label='ML Prediction')
    plt.plot(df['time'], basic_results['v_magnitude_integrated'], 'r--', label='Basic Method')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed Estimation Comparison')
    plt.legend()
    plt.grid(True)
    fname2 = f"{output_prefix}_speed.png"
    save_plot(fig, fname2)
    plot_files.append((fname2, "Speed Estimation Comparison"))

    # 3. Path comparison (GPS, Basic, ML)
    x_ml, y_ml = calculate_ml_path(features_df, ml_predictions)
    fig = plt.figure(figsize=(8, 8))
    plt.plot(df['x_pos_gps'], df['y_pos_gps'], 'g-', label='GPS Path')
    plt.plot(basic_results['x_pos_accel'], basic_results['y_pos_accel'], 'r--', label='Basic Method Path')
    plt.plot(x_ml, y_ml, 'b-.', label='ML Path')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Path Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    fname3 = f"{output_prefix}_paths.png"
    save_plot(fig, fname3)
    plot_files.append((fname3, "2D Path Comparison (GPS, Basic, ML)"))

    return plot_files, (x_ml, y_ml)

def generate_report(df, ml_results, basic_results, output_path='analysis_report.pdf'):
    """Generate comprehensive PDF report with signal processing analysis and visuals"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    story.append(Paragraph("IMU-based Speed and Distance Estimation Analysis", title_style))
    story.append(PageBreak())

    # 1. Data Overview
    story.append(Paragraph("1. Data Overview", styles['Heading2']))
    sampling_rate = 1 / df['time'].diff().mean()
    data_duration = df['time'].max() - df['time'].min()
    overview_text = f"""
    <b>Sampling Rate:</b> {sampling_rate:.2f} Hz<br/>
    <b>Total Duration:</b> {data_duration:.2f} seconds<br/>
    <b>Number of Samples:</b> {len(df)}<br/>
    <b>Nyquist Frequency:</b> {sampling_rate/2:.2f} Hz<br/>
    """
    story.append(Paragraph(overview_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # 2. Signal Processing Pipeline
    story.append(Paragraph("2. Signal Processing Pipeline", styles['Heading2']))
    pipeline_text = """
    <b>2.1 Pre-processing:</b><br/>
    &bull; <b>Butterworth Low-pass Filter</b> (IMU data)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;Removes high-frequency noise, preserves motion dynamics.<br/>
    &bull; <b>Nyquist Theorem:</b> Sampling rate must be at least twice the highest frequency of interest.<br/>
    <br/>
    <b>2.2 Orientation Estimation:</b><br/>
    &bull; <b>Gyroscope Integration (Euler angles):</b> Simple numerical integration to estimate roll, pitch, yaw.<br/>
    <br/>
    <b>2.3 Coordinate Transformation:</b><br/>
    &bull; <b>Body to World Frame:</b> 3D rotation matrix compensates for device orientation.<br/>
    <br/>
    <b>2.4 Acceleration Processing:</b><br/>
    &bull; <b>High-pass Filter:</b> Removes integration drift.<br/>
    """
    story.append(Paragraph(pipeline_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Insert raw/filtered accel plot
    story.append(Paragraph("2.5 Visual: Raw and Filtered Accelerometer Magnitude", styles['Heading3']))
    plot_files, (x_ml, y_ml) = generate_all_plots(df, ml_results['features_df'], ml_results['predictions'], basic_results)
    story.append(Image(plot_files[0][0], width=5*inch, height=3*inch))
    story.append(Paragraph("Figure: Shows the effect of low-pass filtering on the accelerometer signal. Filtering is crucial for removing noise before integration.", styles['Normal']))
    story.append(PageBreak())

    # 3. Machine Learning Approach
    story.append(Paragraph("3. Machine Learning Analysis", styles['Heading2']))
    try:
        rmse = np.sqrt(mean_squared_error(ml_results['actual_speed'], ml_results['predictions']))
    except:
        rmse = 0.0
    ml_text = f"""
    <b>3.1 Feature Extraction:</b><br/>
    &bull; Window Size: 1.0 seconds<br/>
    &bull; Features: Mean, Std, Max, Min, Peak-to-peak of IMU signals<br/>
    <br/>
    <b>3.2 Model:</b> Random Forest Regressor<br/>
    <b>3.3 Performance:</b> RMSE = {rmse:.3f} m/s<br/>
    """
    story.append(Paragraph(ml_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Insert speed comparison plot
    story.append(Paragraph("3.4 Visual: Speed Estimation Comparison", styles['Heading3']))
    story.append(Image(plot_files[1][0], width=5*inch, height=3*inch))
    story.append(Paragraph("Figure: Comparison of speed estimated by GPS, ML, and basic integration. ML can learn to correct for drift and noise.", styles['Normal']))
    story.append(PageBreak())

    # 4. Comparison of Methods
    story.append(Paragraph("4. Methods Comparison", styles['Heading2']))
    ml_distance = ml_results['distance'][-1]
    basic_distance = basic_results['distance'][-1]
    gps_distance = df['Speed (m/s)'].mean() * data_duration
    data = [
        ['Method', 'Total Distance (m)', 'Relative Error (%)'],
        ['GPS (Ground Truth)', f"{gps_distance:.2f}", "0.00"],
        ['ML Method', f"{ml_distance:.2f}", f"{abs(ml_distance-gps_distance)/gps_distance*100:.2f}"],
        ['Basic Method', f"{basic_distance:.2f}", f"{abs(basic_distance-gps_distance)/gps_distance*100:.2f}"]
    ]
    table = Table(data)
    table.setStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ])
    story.append(table)
    story.append(Spacer(1, 12))

    # Insert path comparison plot
    story.append(Paragraph("4.1 Visual: 2D Path Comparison (GPS, Basic, ML)", styles['Heading3']))
    story.append(Image(plot_files[2][0], width=5*inch, height=5*inch))
    story.append(Paragraph("Figure: Comparison of reconstructed paths using GPS, basic integration, and ML-predicted velocities. ML can reduce drift and error compared to basic integration.", styles['Normal']))
    story.append(PageBreak())

    # 5. Error Analysis and Signal Processing Concepts
    story.append(Paragraph("5. Error Analysis & Signal Processing Concepts", styles['Heading2']))
    error_text = """
    <b>5.1 Sources of Error:</b><br/>
    &bull; <b>Integration Drift:</b> Small errors in acceleration accumulate over time.<br/>
    &bull; <b>Sensor Noise:</b> IMU noise is filtered using Butterworth filters.<br/>
    &bull; <b>Orientation Errors:</b> Gyroscope drift affects coordinate transformation.<br/>
    <br/>
    <b>5.2 Signal Processing Concepts:</b><br/>
    &bull; <b>Filtering:</b> Butterworth filters are used for both low-pass (noise removal) and high-pass (drift removal).<br/>
    &bull; <b>Sampling Theorem:</b> Ensures no aliasing in digital signals.<br/>
    &bull; <b>Numerical Integration:</b> Trapezoidal rule is used for velocity and position estimation.<br/>
    &bull; <b>Feature Engineering:</b> Statistical features from IMU signals are used for ML.<br/>
    """
    story.append(Paragraph(error_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    print(f"Report generated: {output_path}")

def ensure_filtered_columns(df):
    """Ensure filtered columns exist in DataFrame, apply filtering if missing."""
    required = ['ax_filtered', 'ay_filtered', 'az_filtered', 'wx_filtered', 'wy_filtered', 'wz_filtered']
    if not all(col in df.columns for col in required):
        # Estimate sampling rate
        time_diffs = df['time'].diff().dropna()
        sampling_rate = 1 / time_diffs.mean()
        def butter_lowpass_filter(data, cutoff, fs, order=2):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)
        df = df.copy()
        df['ax_filtered'] = butter_lowpass_filter(df['ax'], cutoff=0.85, fs=sampling_rate)
        df['ay_filtered'] = butter_lowpass_filter(df['ay'], cutoff=1, fs=sampling_rate)
        df['az_filtered'] = butter_lowpass_filter(df['az'], cutoff=0.5, fs=sampling_rate)
        df['wx_filtered'] = butter_lowpass_filter(df['wx'], cutoff=2.0, fs=sampling_rate)
        df['wy_filtered'] = butter_lowpass_filter(df['wy'], cutoff=2.0, fs=sampling_rate)
        df['wz_filtered'] = butter_lowpass_filter(df['wz'], cutoff=2.0, fs=sampling_rate)
    return df

def compare_methods(data_path):
    """Compare ML model vs basic method and generate report"""
    df = pd.read_csv(data_path)
    df = calculate_gps_path(df)
    basic_results = process_basic_method(df)
    
    try:
        model, scaler = load_model()
        predictions, features_df = test_model(data_path, model, scaler)
        # Ensure yaw data is available
        features_df['yaw'] = np.interp(features_df['time'], 
                                     basic_results['time'],
                                     basic_results['yaw'])
    except FileNotFoundError:
        print("No trained model found. Training new model...")
        ml_results = process_ml_model(df)
        predictions = ml_results['predictions']
        features_df = ml_results['features_df']
    
    ml_results = {
        'predictions': predictions,
        'actual_speed': features_df['label'].values,
        'time': features_df['time'].values,
        'distance': (predictions * features_df['duration'].values).cumsum(),
        'features_df': features_df
    }
    
    # Update DataFrame with all required columns
    for key in ['ax_filtered', 'ay_filtered', 'az_filtered', 'roll', 'pitch', 'yaw', 'x_pos_gps']:
        if key not in df.columns:
            df[key] = basic_results[key]
    
    try:
        generate_report(df, ml_results, basic_results)
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        missing = set(['ax_filtered', 'ay_filtered', 'az_filtered', 
                      'roll', 'pitch', 'yaw']) - set(df.columns)
        if missing:
            print("Missing columns:", missing)

def plot_comparisons(df, ml_predictions, features_df, basic_results):
    """Plot comparison graphs"""
    # Defensive: check for required columns before plotting
    for col in ['x_pos_gps', 'y_pos_gps']:
        if col not in df.columns:
            print(f"Warning: '{col}' not found in DataFrame. Skipping GPS path plot.")
            return
    for col in ['x_pos_accel', 'y_pos_accel']:
        if col not in basic_results:
            print(f"Warning: '{col}' not found in basic_results. Skipping Basic Method path plot.")
            return

    plt.figure(figsize=(15, 10))
    
    # Speed comparison
    plt.subplot(2, 1, 1)
    plt.plot(features_df['time'], features_df['label'], 'g-', label='GPS Ground Truth')
    plt.plot(features_df['time'], ml_predictions, 'b--', label='ML Prediction')
    plt.plot(df['time'], basic_results['v_magnitude_integrated'], 'r--', label='Basic Method')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed Estimation Comparison')
    plt.legend()
    plt.grid(True)
    
    # Path comparison
    plt.subplot(2, 1, 2)
    plt.plot(df['x_pos_gps'], df['y_pos_gps'], 'g-', label='GPS Path')
    plt.plot(basic_results['x_pos_accel'], basic_results['y_pos_accel'], 'r--', label='Basic Method Path')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Path Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    while True:
        print("\n1. Train new model")
        print("2. Test existing model")
        print("3. Compare methods")
        print("4. Exit")
        print("5. Generate Analysis Report")
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            data_path = input("Enter training data path: ")
            train_model(data_path)
        
        elif choice == '2':
            data_path = input("Enter test data path: ")
            try:
                model, scaler = load_model()
                predictions, features_df = test_model(data_path, model, scaler)
                plot_comparisons(pd.read_csv(data_path), predictions, features_df, process_basic_method(pd.read_csv(data_path)))
            except FileNotFoundError:
                print("No trained model found. Please train a model first.")
        
        elif choice == '3':
            data_path = input("Enter data path for comparison: ")
            compare_methods(data_path)
        
        elif choice == '4':
            break
        
        elif choice == '5':
            data_path = input("Enter data path for report generation: ")
            compare_methods(data_path)
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
