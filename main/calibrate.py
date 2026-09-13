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
    
    # Filter out rows where the scooter is not moving (speed < 0.5 m/s)
    moving = df[df['speed'] > 0.5].copy()
    
    if len(moving) < 100:
        print("Warning: Very little moving data found. Profile may be inaccurate.")
    
    # Create speed bins (e.g. 0-2 m/s, 2-4 m/s, 4-6 m/s, 6-8 m/s, 8+ m/s)
    # Using 2 m/s bins (~7.2 km/h)
    bins = [0, 2, 4, 6, 8, 10, 15, 20, 100]
    labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15-20', '20+']
    moving['speed_bin'] = pd.cut(moving['speed'], bins=bins, labels=labels, include_lowest=True)
    
    profile = {}
    print("\nSpeed Bin | Accel Z Baseline (Mean) | Accel Z StdDev")
    print("-" * 55)
    
    for bin_label in labels:
        bin_data = moving[moving['speed_bin'] == bin_label]
        if len(bin_data) > 10:
            mean_z = float(bin_data['accel_z'].mean())
            std_z = float(bin_data['accel_z'].std())
            profile[bin_label] = {'mean': mean_z, 'std': std_z}
            print(f"{bin_label:>9} | {mean_z:>21.4f} | {std_z:>14.4f}")
        else:
            profile[bin_label] = {'mean': 9.81, 'std': 1.0} # Fallback defaults
            
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
        csv_file = os.path.join(BASE_DIR, csv_file)
        
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        sys.exit(1)
        
    train_profile(csv_file)
