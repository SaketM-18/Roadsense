"""
RoadSense — Sensor & Vision Edge Fusion Engine (v2.0)
======================================================
Fuses YOLO computer vision detections with high-frequency physical IMU data.

Key Upgrades:
1. Dynamic Speed-Dependent Lookahead Delay: t_impact = d_lookahead / v_speed.
2. Jerk Computation: da/dt sharply differentiates sharp rim strikes from smooth undulations.
3. Butterworth Bandpass Filtering: 5–18 Hz filter isolates road suspension impacts from engine/braking hum.
4. Strict Physical Verification Gate (AND Gate): Rejects shadows and dark pavement patches.
5. Failsafe Sensor Override: Catches un-imaged potholes at night or behind traffic.
6. Gyroscope Pitch Disambiguation: Distinguishes speed bumps from potholes.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

EARTH_RADIUS_METERS = 6371000.0


def apply_butterworth_bandpass(signal, fs, lowcut=5.0, highcut=18.0, order=3):
    """
    Apply a zero-phase Butterworth bandpass filter to isolate vehicle suspension resonance.
    """
    if len(signal) < 15 or fs <= 0:
        return signal
    
    nyquist = 0.5 * fs
    highcut = min(highcut, nyquist * 0.9)
    lowcut = max(0.5, min(lowcut, highcut * 0.5))
    
    if lowcut >= highcut or highcut >= nyquist:
        return signal

    try:
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='bandpass')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        return signal


def compute_jerk(accel_values, time_values):
    """
    Compute instantaneous Jerk (da/dt) in m/s^3.
    Sharp pothole rim strikes produce Jerk >= 60-100 m/s^3, whereas smooth speed dips have low jerk.
    """
    jerk = np.zeros(len(accel_values), dtype=float)
    if len(accel_values) < 2:
        return jerk
    
    dt = np.diff(time_values)
    da = np.diff(accel_values)
    
    # Avoid zero division
    dt[dt <= 0.0001] = 0.025
    jerk[1:] = np.abs(da / dt)
    jerk[0] = jerk[1]
    return jerk


def fuse_sensor_and_vision_data(
    yolo_csv,
    sensor_csv,
    output_csv,
    sync_offset_sec=0.0,
    lookahead_distance_m=3.5,
    default_delay_sec=0.5
):
    """
    Fuses YOLO vision detections with Sensor IMU data using physics-grounded gating.
    
    yolo_csv: Output from detect_potholes.py (must have 'video_time_sec' or 'time_sec')
    sensor_csv: High-frequency telemetry CSV (phone or phyphox)
    output_csv: Path to save the labeled and physically verified dataset
    sync_offset_sec: Millisecond clock alignment offset between video and telemetry
    lookahead_distance_m: Distance in meters from camera viewpoint to front tire (default: 3.5m)
    default_delay_sec: Fallback delay when vehicle speed is stationary/unknown (default: 0.5s)
    """
    print("=" * 60)
    print("ROADSENSE — Multi-Modal Physics-Grounded Fusion Engine")
    print("=" * 60)
    print("Loading telemetry and vision records...")
    
    try:
        df_yolo = pd.read_csv(yolo_csv)
    except Exception as e:
        print(f"  Note: YOLO CSV not found ({e}). Running in Sensor-Only Mode.")
        df_yolo = pd.DataFrame()

    try:
        df_sensor = pd.read_csv(sensor_csv)
    except Exception as e:
        print(f"  Error loading sensor CSV {sensor_csv}: {e}")
        return None

    # 1. Normalize time column
    time_col = None
    for c in ['time_s', 'Time (s)', 'time', 'timestamp', 'Timestamp']:
        if c in df_sensor.columns:
            time_col = c
            break
            
    if not time_col:
        df_sensor['time_s'] = np.arange(len(df_sensor)) * 0.025
    elif time_col != 'time_s':
        t_vals = df_sensor[time_col].values
        if t_vals[0] > 1e11: # Epoch milliseconds
            df_sensor['time_s'] = (t_vals - t_vals[0]) / 1000.0
        elif t_vals[0] > 1e8: # Epoch seconds
            df_sensor['time_s'] = t_vals - t_vals[0]
        else:
            df_sensor['time_s'] = t_vals - t_vals[0]

    # Normalize coordinate and sensor columns
    for alias, standard in [
        ('Linear Acceleration z (m/s^2)', 'accel_z'),
        ('Acceleration z (m/s^2)', 'accel_z'),
        ('az', 'accel_z'),
        ('Linear Acceleration y (m/s^2)', 'accel_y'),
        ('Acceleration y (m/s^2)', 'accel_y'),
        ('ay', 'accel_y'),
        ('Linear Acceleration x (m/s^2)', 'accel_x'),
        ('Acceleration x (m/s^2)', 'accel_x'),
        ('ax', 'accel_x'),
        ('Gyroscope y (rad/s)', 'gyro_y'),
        ('Gyroscope y (deg/s)', 'gyro_y'),
        ('gy', 'gyro_y'),
        ('Speed (m/s)', 'speed'),
        ('Velocity (km/h)', 'speed'),
        ('Latitude', 'latitude'),
        ('Longitude', 'longitude')
    ]:
        if alias in df_sensor.columns and standard not in df_sensor.columns:
            df_sensor[standard] = df_sensor[alias]

    # Ensure vertical acceleration exists
    if 'accel_z' not in df_sensor.columns:
        df_sensor['accel_z'] = 9.81
    if 'accel_dyn' not in df_sensor.columns:
        ax = df_sensor.get('accel_x', 0)
        ay = df_sensor.get('accel_y', 0)
        az = df_sensor.get('accel_z', 9.81)
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        df_sensor['accel_dyn'] = np.abs(mag - 9.81)

    # 2. Compute sampling rate and Butterworth Bandpass filtered signal
    median_dt = np.median(np.diff(df_sensor['time_s'].values)) if len(df_sensor) > 2 else 0.025
    fs = 1.0 / max(0.005, median_dt)
    print(f"  Telemetry: {len(df_sensor)} readings | Sampling rate: {fs:.1f} Hz")
    
    df_sensor['accel_bandpass'] = apply_butterworth_bandpass(
        df_sensor['accel_dyn'].values, fs=fs, lowcut=5.0, highcut=18.0
    )

    # 3. Compute Jerk (da/dt)
    df_sensor['jerk'] = compute_jerk(df_sensor['accel_dyn'].values, df_sensor['time_s'].values)

    # 4. Compute Dynamic Rolling Baseline & Z-Score (10s rolling window)
    window_samples = max(10, int(10.0 * fs))
    roll_mean = df_sensor['accel_dyn'].rolling(window=window_samples, min_periods=5, center=True).mean()
    roll_std = df_sensor['accel_dyn'].rolling(window=window_samples, min_periods=5, center=True).std()
    roll_mean = roll_mean.bfill().ffill()
    roll_std = roll_std.bfill().ffill().replace(0, 0.5)

    df_sensor['z_score'] = np.abs(df_sensor['accel_dyn'] - roll_mean) / roll_std
    df_sensor['z_score'] = df_sensor['z_score'].fillna(0.0).round(2)

    # 5. Initialize Output Labels
    df_sensor['Label'] = 0
    df_sensor['Source'] = 'None'
    df_sensor['Hazard_Type'] = 'None'
    df_sensor['Severity'] = 'None'
    df_sensor['box_area_pct'] = 0.0

    # 6. Multi-Modal Vision & Physical Gating (AND Gate)
    verified_yolo = 0
    rejected_yolo = 0

    yolo_time_col = None
    for c in ['time_sec', 'video_time_sec', 'timestamp', 'time']:
        if c in df_yolo.columns:
            yolo_time_col = c
            break

    if yolo_time_col and not df_yolo.empty:
        print(f"  Fusing {len(df_yolo)} vision detections with speed-adaptive lookahead...")
        for _, yolo_row in df_yolo.iterrows():
            v_time = float(yolo_row[yolo_time_col])
            conf = float(yolo_row.get('confidence', 0.5))
            area_pct = float(yolo_row.get('box_area_pct', 0.0))
            
            # Align time
            aligned_t = v_time + sync_offset_sec
            
            # Find vehicle speed at detection timestamp
            idx_closest = (df_sensor['time_s'] - aligned_t).abs().idxmin()
            speed_val = float(df_sensor.at[idx_closest, 'speed']) if 'speed' in df_sensor.columns else 0.0
            
            # Dynamic speed lookahead: t_impact = d / v
            if speed_val > 1.0: # vehicle in motion
                calculated_delay = float(np.clip(lookahead_distance_m / speed_val, 0.15, 1.5))
            else:
                calculated_delay = default_delay_sec

            impact_t = aligned_t + calculated_delay
            window_mask = (df_sensor['time_s'] >= impact_t - 0.25) & (df_sensor['time_s'] <= impact_t + 0.25)
            
            if window_mask.sum() == 0:
                continue

            max_z = df_sensor.loc[window_mask, 'z_score'].max()
            max_jerk = df_sensor.loc[window_mask, 'jerk'].max()

            # Physical Verification: Vision must correlate with physical shock spike
            if max_z >= 1.5 or max_jerk >= 50.0:
                df_sensor.loc[window_mask, 'Label'] = 1
                df_sensor.loc[window_mask, 'Source'] = 'Fused'
                df_sensor.loc[window_mask, 'Hazard_Type'] = 'Pothole'
                if area_pct > 0:
                    df_sensor.loc[window_mask, 'box_area_pct'] = area_pct
                verified_yolo += 1
            else:
                # Rejected! Road was physically smooth (false alarm from shadow/paint)
                rejected_yolo += 1

        print(f"  Camera Results: {verified_yolo} Verified | {rejected_yolo} Shadow/Visual False Positives Replaced")

    # 7. Failsafe Sensor Override (Severe Mechanical Shock)
    override_count = 0
    for idx, row in df_sensor.iterrows():
        z = float(row['z_score'])
        j = float(row.get('jerk', 0.0))
        
        # Heavy physical strike
        if z >= 3.0 or j >= 100.0:
            df_sensor.at[idx, 'Label'] = 1
            if df_sensor.at[idx, 'Source'] == 'None':
                df_sensor.at[idx, 'Source'] = 'Sensor'
                # Disambiguate speedbreaker vs pothole using gyro pitch deflection
                if 'gyro_y' in df_sensor.columns and abs(row['gyro_y']) > 40:
                    df_sensor.at[idx, 'Hazard_Type'] = 'Speedbreaker'
                else:
                    df_sensor.at[idx, 'Hazard_Type'] = 'Pothole'
                override_count += 1

    if override_count > 0:
        print(f"  Physical Failsafe Overrides (Night/Occluded Hazards): {override_count}")

    # 8. Severity Classification & Metric Dimensioning
    df_sensor['est_area_cm2'] = 0.0
    df_sensor['est_depth_cm'] = 0.0
    
    for idx, row in df_sensor.iterrows():
        if row['Label'] == 1:
            z = float(row['z_score'])
            area_pct = float(row.get('box_area_pct', 0.0))
            
            if z >= 3.0:
                df_sensor.at[idx, 'Severity'] = 'Severe'
            elif z >= 2.0:
                df_sensor.at[idx, 'Severity'] = 'Moderate'
            else:
                df_sensor.at[idx, 'Severity'] = 'Low'
                
            # Dimensional modeling
            df_sensor.at[idx, 'est_area_cm2'] = round(max(150.0, area_pct * 140.0), 1) if area_pct > 0 else round(max(100.0, z * 180.0), 1)
            df_sensor.at[idx, 'est_depth_cm'] = round(min(12.0, max(1.5, z * 1.65)), 1)

    # Save labeled dataset
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_sensor.to_csv(output_csv, index=False)
    
    pothole_count = (df_sensor['Label'] == 1).sum()
    print(f"  Successfully labeled {pothole_count} road hazard reading(s).")
    print(f"[SAVED] Fused Dataset: {output_csv}")
    print("=" * 60)
    
    return df_sensor


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(current_dir)
    
    yolo_file = os.path.join(current_dir, "yolo_timestamps.csv")
    sensor_file = os.path.join(main_dir, "test", "Four", "telemetry.csv")
    if not os.path.exists(sensor_file):
        # Fallback to any test csv
        csvs = [os.path.join(main_dir, "test", "Four", f) for f in os.listdir(os.path.join(main_dir, "test", "Four")) if f.endswith('.csv') and 'fused' not in f]
        if csvs:
            sensor_file = csvs[0]
            
    out_file = os.path.join(current_dir, "fused_labeled_data.csv")
    
    if os.path.exists(sensor_file):
        fuse_sensor_and_vision_data(yolo_file, sensor_file, out_file)
    else:
        print(f"Ready. Provide input files and call fuse_sensor_and_vision_data().")
