import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Apply low-pass filter to accelerometer and gyroscope data to reduce noise
def butter_lowpass_filter(data, cutoff=3.0, fs=50, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Perform high-pass filtering to remove drift in accelerations
def butter_highpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Manual implementation of cumulative trapezoidal integration
def cumulative_trapezoid(y, x):
    dx = np.diff(x)
    cumsum = np.zeros(len(y))
    for i in range(1, len(y)):
        cumsum[i] = cumsum[i-1] + 0.5 * (y[i] + y[i-1]) * dx[i-1]
    return cumsum

def process_basic_method(df):
    """Process data using basic method and return all computed results"""
    # Calculate sampling rate
    time_diffs = df['time'].diff().dropna()
    sampling_rate = 1 / time_diffs.mean()
    
    # Apply filters and calculate initial values
    df = df.copy()  # Create a copy to avoid modifying original
    df['time_diff'] = df['time'].diff().fillna(0)
    
    # Apply lowpass filter
    df['ax_filtered'] = butter_lowpass_filter(df['ax'], cutoff=0.85, fs=sampling_rate)
    df['ay_filtered'] = butter_lowpass_filter(df['ay'], cutoff=1, fs=sampling_rate)
    df['az_filtered'] = butter_lowpass_filter(df['az'], cutoff=0.5, fs=sampling_rate)
    df['wx_filtered'] = butter_lowpass_filter(df['wx'], cutoff=2.0, fs=sampling_rate)
    df['wy_filtered'] = butter_lowpass_filter(df['wy'], cutoff=2.0, fs=sampling_rate)
    df['wz_filtered'] = butter_lowpass_filter(df['wz'], cutoff=2.0, fs=sampling_rate)
    
    # Compute orientation
    roll = np.zeros(len(df))
    pitch = np.zeros(len(df))
    yaw = np.zeros(len(df))
    
    for i in range(1, len(df)):
        dt = df['time_diff'].iloc[i]
        roll[i] = roll[i-1] + df['wx_filtered'].iloc[i] * dt
        pitch[i] = pitch[i-1] + df['wy_filtered'].iloc[i] * dt
        yaw[i] = yaw[i-1] + df['wz_filtered'].iloc[i] * dt
    
    df['roll'], df['pitch'], df['yaw'] = roll, pitch, yaw
    
    # Transform to world frame
    a_world_x, a_world_y, a_world_z = transform_to_world_frame(df)
    df['ax_world'] = a_world_x
    df['ay_world'] = a_world_y
    df['az_world'] = a_world_z
    
    # Apply high-pass filter
    df['ax_world_filtered'] = butter_highpass_filter(df['ax_world'], cutoff=0.06, fs=sampling_rate)
    df['ay_world_filtered'] = butter_highpass_filter(df['ay_world'], cutoff=0.06, fs=sampling_rate)
    
    # Calculate velocities and positions
    df['vx_integrated'] = cumulative_trapezoid(df['ax_world_filtered'].values, df['time'].values)
    df['vy_integrated'] = cumulative_trapezoid(df['ay_world_filtered'].values, df['time'].values)
    
    df['v_magnitude_integrated'] = np.sqrt(df['vx_integrated']**2 + df['vy_integrated']**2)
    
    df['d_dist'] = df['v_magnitude_integrated'] * df['time_diff']
    df['dx_accel'] = df['d_dist'] * np.cos(df['yaw'])
    df['dy_accel'] = df['d_dist'] * np.sin(df['yaw'])
    df['x_pos_accel'] = np.cumsum(df['dx_accel'])
    df['y_pos_accel'] = np.cumsum(df['dy_accel'])
    
    return {
        'v_magnitude_integrated': df['v_magnitude_integrated'].values,
        'x_pos_accel': df['x_pos_accel'].values,
        'y_pos_accel': df['y_pos_accel'].values,
        'time': df['time'].values,
        'distance': (df['v_magnitude_integrated'] * df['time_diff']).cumsum().values,
        'ax_filtered': df['ax_filtered'].values,
        'ay_filtered': df['ay_filtered'].values,
        'az_filtered': df['az_filtered'].values,
        'wx_filtered': df['wx_filtered'].values,
        'wy_filtered': df['wy_filtered'].values,
        'wz_filtered': df['wz_filtered'].values,
        'roll': df['roll'].values,
        'pitch': df['pitch'].values,
        'yaw': df['yaw'].values,
        'a_mag': df['a_mag'].values if 'a_mag' in df.columns else None,
        'a_filtered_mag': df['a_filtered_mag'].values if 'a_filtered_mag' in df.columns else None
    }

def transform_to_world_frame(df):
    """Transform accelerometer readings from body frame to world frame"""
    a_world_x = np.zeros(len(df))
    a_world_y = np.zeros(len(df))
    a_world_z = np.zeros(len(df))
    
    for i in range(len(df)):
        cos_roll = np.cos(df['roll'].iloc[i])
        sin_roll = np.sin(df['roll'].iloc[i])
        cos_pitch = np.cos(df['pitch'].iloc[i])
        sin_pitch = np.sin(df['pitch'].iloc[i])
        cos_yaw = np.cos(df['yaw'].iloc[i])
        sin_yaw = np.sin(df['yaw'].iloc[i])
        
        ax_body = df['ax_filtered'].iloc[i]
        ay_body = df['ay_filtered'].iloc[i]
        az_body = df['az_filtered'].iloc[i]
        
        a_world_x[i] = (cos_yaw * cos_pitch) * ax_body + \
                      (cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll) * ay_body + \
                      (cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll) * az_body
        
        a_world_y[i] = (sin_yaw * cos_pitch) * ax_body + \
                      (sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll) * ay_body + \
                      (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll) * az_body
        
        a_world_z[i] = (-sin_pitch) * ax_body + \
                      (cos_pitch * sin_roll) * ay_body + \
                      (cos_pitch * cos_roll) * az_body
    
    return a_world_x, a_world_y, a_world_z

if __name__ == "__main__":
    # Move the main execution code here
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt

    # Load CSV data
    # file_path = "bhaskara_to_ground.csv"
    file_path = "round_around_campus.csv"
    # file_path = "round_around_ground.csv"
    df = pd.read_csv(file_path)

    # Estimate sampling frequency from time column
    time_diffs = df['time'].diff().dropna()
    sampling_rate = 1 / time_diffs.mean()
    print(f"Estimated sampling rate: {sampling_rate:.2f} Hz")

    # Apply lowpass filter to accelerometer and gyroscope data
    df['ax_filtered'] = butter_lowpass_filter(df['ax'], cutoff=0.85, fs=sampling_rate)
    df['ay_filtered'] = butter_lowpass_filter(df['ay'], cutoff=1, fs=sampling_rate)
    df['az_filtered'] = butter_lowpass_filter(df['az'], cutoff=0.5, fs=sampling_rate)
    df['wx_filtered'] = butter_lowpass_filter(df['wx'], cutoff=2.0, fs=sampling_rate)
    df['wy_filtered'] = butter_lowpass_filter(df['wy'], cutoff=2.0, fs=sampling_rate)
    df['wz_filtered'] = butter_lowpass_filter(df['wz'], cutoff=2.0, fs=sampling_rate)
    df['a_mag'] = np.sqrt(df['ax']**2 + df['ay']**2)
    df['a_filtered_mag'] = np.sqrt(df['ax_filtered']**2 + df['ay_filtered']**2)

    # Calculate time differences
    df['time_diff'] = df['time'].diff().fillna(0)

    # Compute orientation using gyroscope integration (simple Euler integration)
    # Initialize arrays for roll, pitch, yaw angles
    roll = np.zeros(len(df))
    pitch = np.zeros(len(df))
    yaw = np.zeros(len(df))

    # Integrate gyroscope data to get orientation
    for i in range(1, len(df)):
        dt = df['time_diff'].iloc[i]
        roll[i] = roll[i-1] + df['wx_filtered'].iloc[i] * dt
        pitch[i] = pitch[i-1] + df['wy_filtered'].iloc[i] * dt
        yaw[i] = yaw[i-1] + df['wz_filtered'].iloc[i] * dt

    df['roll'] = roll
    df['pitch'] = pitch
    df['yaw'] = yaw


    # Transform accelerometer readings from body frame to world frame
    # Create empty arrays for world frame accelerations
    a_world_x = np.zeros(len(df))
    a_world_y = np.zeros(len(df))
    a_world_z = np.zeros(len(df))

    for i in range(len(df)):
        # Full rotation matrix using Euler angles (3-2-1 sequence: yaw, pitch, roll)
        cos_roll = np.cos(df['roll'].iloc[i])
        sin_roll = np.sin(df['roll'].iloc[i])
        cos_pitch = np.cos(df['pitch'].iloc[i])
        sin_pitch = np.sin(df['pitch'].iloc[i])
        cos_yaw = np.cos(df['yaw'].iloc[i])
        sin_yaw = np.sin(df['yaw'].iloc[i])
        
        # Get body frame accelerations
        ax_body = df['ax_filtered'].iloc[i]
        ay_body = df['ay_filtered'].iloc[i]
        az_body = df['az_filtered'].iloc[i]
        
        # Apply rotation matrix to convert to world frame
        a_world_x[i] = (cos_yaw * cos_pitch) * ax_body + \
                    (cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll) * ay_body + \
                    (cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll) * az_body
        
        a_world_y[i] = (sin_yaw * cos_pitch) * ax_body + \
                    (sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll) * ay_body + \
                    (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll) * az_body
        
        a_world_z[i] = (-sin_pitch) * ax_body + \
                    (cos_pitch * sin_roll) * ay_body + \
                    (cos_pitch * cos_roll) * az_body

    # Store world frame accelerations
    df['ax_world'] = a_world_x
    df['ay_world'] = a_world_y
    df['az_world'] = a_world_z


    # Apply high-pass filter to remove drift from accelerations
    df['ax_world_filtered'] = butter_highpass_filter(df['ax_world'], cutoff=0.06, fs=sampling_rate)
    df['ay_world_filtered'] = butter_highpass_filter(df['ay_world'], cutoff=0.06, fs=sampling_rate)
    # df['ax_world_filtered'] = df['ax_world']                                      # I used these to check how much drift was in the data
    # df['ay_world_filtered'] = df['ay_world']
    # df['az_world_filtered'] = butter_highpass_filter(df['az_world'], cutoff=0.0935, fs=sampling_rate)
    df['a_mag_world_filtered'] = np.sqrt(df['ax_world_filtered']**2 + df['ay_world_filtered']**2)

    # Integrate accelerations to get velocities using cumulative trapezoidal rule
    df['vx_integrated'] = cumulative_trapezoid(df['ax_world_filtered'].values, df['time'].values)
    df['vy_integrated'] = cumulative_trapezoid(df['ay_world_filtered'].values, df['time'].values)
    # df['vz_integrated'] = cumulative_trapezoid(df['az_world_filtered'].values, df['time'].values)

    # Calculate total velocity magnitude from integrated accelerations
    df['v_magnitude_integrated'] = np.sqrt(
        # df['vx_integrated']**2 + df['vy_integrated']**2 + df['vz_integrated']**2
        df['vx_integrated']**2 + df['vy_integrated']**2
    )

    # Integrate velocities to get positions
    df['x_pos_integrated'] = cumulative_trapezoid(df['vx_integrated'].values, df['time'].values)
    df['y_pos_integrated'] = cumulative_trapezoid(df['vy_integrated'].values, df['time'].values)
    df['d_dist'] = df['v_magnitude_integrated'] * df['time_diff']
    df['dx_accel'] = df['d_dist'] * np.cos(df['yaw'])
    df['dy_accel'] = df['d_dist'] * np.sin(df['yaw'])
    df['x_pos_accel'] = np.cumsum(df['dx_accel'])
    df['y_pos_accel'] = np.cumsum(df['dy_accel'])

    # df['z_pos_integrated'] = cumulative_trapezoid(df['vz_integrated'].values, df['time'].values)

    # Calculate position using GPS speed (original approach)
    df["displacement"] = df["Speed (m/s)"] * df["time_diff"]
    df["dx_gps"] = df["displacement"] * np.cos(df["yaw"])
    df["dy_gps"] = df["displacement"] * np.sin(df["yaw"])
    df["x_pos_gps"] = np.cumsum(df["dx_gps"])
    df["y_pos_gps"] = np.cumsum(df["dy_gps"])

    # Calculate total distance traveled - Method 1: GPS Speed
    # Sum of displacement = sum of speed * dt
    total_distance_gps = df["displacement"].sum()

    # Calculate total distance traveled - Method 2: Integrated Accelerometer
    # Calculate incremental distances between consecutive positions
    df['dx_acc'] = df['x_pos_integrated'].diff().fillna(0)
    df['dy_acc'] = df['y_pos_integrated'].diff().fillna(0)
    # df['dz_acc'] = df['z_pos_integrated'].diff().fillna(0)

    # Calculate 3D distance increments
    df['distance_increment_acc'] = np.sqrt(df['dx_acc']**2 + df['dy_acc']**2)

    # Sum up all increments to get total distance
    total_distance_acc = df['distance_increment_acc'].sum()

    # Calculate total distance traveled - Method 3: Velocity Integration
    # Integrate the speed magnitude over time
    df['distance_from_speed'] = df['v_magnitude_integrated'] * df['time_diff']
    total_distance_velocity = df['distance_from_speed'].sum()

    # Print the results
    print(f"Total distance traveled (GPS method): {total_distance_gps:.2f} meters")
    print(f"Total distance traveled (Velocity integration): {total_distance_velocity:.2f} meters")

    # Plot 1: Raw and filtered accelerometer data
    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df['a_mag'], 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.ylim(0,7)
    plt.title('Raw Accelerometer Data (Magnitude)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df['a_filtered_mag'], 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.ylim(0,7)
    plt.title('Filtered Accelerometer Data (Magnitude)')
    plt.grid(True)
    plt.show()


    # Plot 2: 2D Path from Accelerometer
    plt.plot(df['x_pos_accel'], df['y_pos_accel'], 'b-')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Path From Accelerometer')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # 2-D Path Comparison
    plt.plot(df['x_pos_accel'], df['y_pos_accel'], 'b-', label='Accelerometer + Gyro Path')
    plt.plot(df['x_pos_gps'], df['y_pos_gps'], 'r--', label='GPS + Gyro Path')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Path Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # # Additional analysis: Calculate drift between methods
    # distance_between_methods = np.sqrt(
    #     (df['x_pos_integrated'] - df['x_pos_gps'])**2 + 
    #     (df['y_pos_integrated'] - df['y_pos_gps'])**2
    # )

    # # Calculate and print error metrics
    # vel_rmse = np.sqrt(np.mean((df['v_magnitude_integrated'] - df['Speed (m/s)'])**2))
    # final_position_error = distance_between_methods.iloc[-1]

    # print(f"Velocity RMSE: {vel_rmse:.3f} m/s")
    # print(f"Final position difference: {final_position_error:.3f} m")

    # # Normalizing velocity by dividing by mean velocity and calculating distance traveled
    # df['normalized_velocity'] = df['v_magnitude_integrated'] / np.mean(df['v_magnitude_integrated'])
    # df['normalized_dist'] = df['normalized_velocity'] * df['time_diff']
    # final_distance = df['normalized_dist'].sum()

    # print(f"Final distance by accelerometer: {final_distance} m")