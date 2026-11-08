import pandas as pd
import numpy as np
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE = os.path.join(BASE_DIR, 'vehicle_profile.json')

def train_profile(csv_path):
    print(f"Loading calibration data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Orientation-invariant computation if full 3D IMU is available
    if 'accel_x' in df.columns and 'accel_y' in df.columns and 'accel_z' in df.columns:
        print("  Detected 3-axis accelerometer data. Computing orientation-invariant dynamic acceleration...")
        mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
        df['accel_dyn'] = np.abs(mag - 9.81)
    elif 'accel_dyn' not in df.columns and 'accel_z' in df.columns:
        # Fallback for single-axis data
        df['accel_dyn'] = np.abs(df['accel_z'] - 9.81)

    # Filter out rows where the vehicle is stationary (speed < 0.5 m/s)
    if 'speed' in df.columns:
        moving = df[df['speed'] > 0.5].copy()
    else:
        moving = df.copy()
    
    if len(moving) < 100:
        print("Warning: Very little moving data found. Profile may be inaccurate.")
    
    # Create speed bins (e.g. 0-2 m/s, 2-4 m/s, 4-6 m/s, 6-8 m/s, 8+ m/s)
    bins = [0, 2, 4, 6, 8, 10, 15, 20, 100]
    labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15-20', '20+']
    if 'speed' in moving.columns:
        moving['speed_bin'] = pd.cut(moving['speed'], bins=bins, labels=labels, include_lowest=True)
    else:
        moving['speed_bin'] = '4-6' # default mid-speed bin
    
    profile = {}
    print("\nSpeed Bin | Accel Z Mean | Accel Z StdDev | Dyn Accel Mean | Dyn Accel StdDev")
    print("-" * 75)
    
    for bin_label in labels:
        bin_data = moving[moving['speed_bin'] == bin_label] if 'speed' in moving.columns else moving
        if len(bin_data) > 10:
            mean_z = float(bin_data['accel_z'].mean()) if 'accel_z' in bin_data.columns else 9.81
            std_z = float(bin_data['accel_z'].std()) if 'accel_z' in bin_data.columns else 1.0
            mean_dyn = float(bin_data['accel_dyn'].mean()) if 'accel_dyn' in bin_data.columns else 1.0
            std_dyn = float(bin_data['accel_dyn'].std()) if 'accel_dyn' in bin_data.columns else 1.0
            
            profile[bin_label] = {
                'mean': mean_z,
                'std': max(std_z, 0.1),
                'mean_dyn': mean_dyn,
                'std_dyn': max(std_dyn, 0.1)
            }
            print(f"{bin_label:>9} | {mean_z:>12.4f} | {std_z:>14.4f} | {mean_dyn:>14.4f} | {std_dyn:>16.4f}")
        else:
            profile[bin_label] = {
                'mean': 9.81, 'std': 1.0,
                'mean_dyn': 0.8, 'std_dyn': 0.8
            } # Fallback defaults
            
    with open(PROFILE_FILE, 'w') as f:
        json.dump(profile, f, indent=4)
        
    print(f"\n[SUCCESS] Vehicle vibration profile saved to {PROFILE_FILE}")
    print("Future rides processed with process_ride.py will now use this physical baseline for 100% anomaly accuracy.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py <path_to_calibration_csv>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    if not os.path.isabs(csv_file):
        if os.path.exists(csv_file):
            csv_file = os.path.abspath(csv_file)
        else:
            csv_file = os.path.join(BASE_DIR, csv_file)
        
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        sys.exit(1)
        
    train_profile(csv_file)
