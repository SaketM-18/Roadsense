import pandas as pd
import numpy as np
import os

def fuse_sensor_and_vision_data(yolo_csv, sensor_csv, output_csv, sync_offset_sec, camera_lookahead_sec=0.5):
    """
    Fuses YOLO vision detections with Sensor IMU data.
    
    yolo_csv: Output from detect_potholes.py
    sensor_csv: The CSV from your phone (phyphox)
    output_csv: Where to save the perfectly labeled data
    sync_offset_sec: The "Clap Sync" offset. (Sensor Time - Video Time at the clap)
                     If you clapped at Video 00:05.00 and the sensor spiked at 00:07.00, 
                     the offset is +2.0 seconds.
    camera_lookahead_sec: The time delay between the camera seeing the pothole and 
                          the front tire physically hitting it. (Default: 0.5s)
    """
    print("Loading data...")
    try:
        df_yolo = pd.read_csv(yolo_csv)
        df_sensor = pd.read_csv(sensor_csv)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Loaded {len(df_yolo)} camera detections and {len(df_sensor)} sensor readings.")
    
    # Create a new label column, default to 0 (Normal Road)
    df_sensor['Label'] = 0
    
    potholes_labeled = 0

    print(f"Applying sync offset: {sync_offset_sec}s and camera delay: {camera_lookahead_sec}s...")
    for idx, row in df_yolo.iterrows():
        video_time = row['video_time_sec']
        
        # 1. Align the clocks
        aligned_sensor_time = video_time + sync_offset_sec
        
        # 2. Add the physical delay (tire hits pothole after camera sees it)
        impact_time = aligned_sensor_time + camera_lookahead_sec
        
        # 3. Find the sensor rows within a 1-second window around the impact
        window_start = impact_time - 0.5
        window_end = impact_time + 0.5
        
        # 4. Label those exact rows as 1 (Pothole)
        mask = (df_sensor['Time (s)'] >= window_start) & (df_sensor['Time (s)'] <= window_end)
        df_sensor.loc[mask, 'Label'] = 1
        
        if mask.sum() > 0:
            potholes_labeled += 1

    print(f"Successfully matched {potholes_labeled} camera detections to physical sensor impacts.")
    
    # Save the fused, labeled dataset
    df_sensor.to_csv(output_csv, index=False)
    print(f"Saved fused training data to: {output_csv}")
    print("You can now feed this CSV directly into the Machine Learning trainer!")

if __name__ == "__main__":
    base_dir = r"d:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health"
    
    yolo_file = os.path.join(base_dir, r"cv_pipeline\yolo_timestamps.csv")
    sensor_file = os.path.join(base_dir, r"test\sample_ride_sensors.csv")
    out_file = os.path.join(base_dir, r"cv_pipeline\fused_labeled_data.csv")
    
    # IMPORTANT: Since you are now using the Unified PWA, the video and sensor CSV 
    # start at the exact same millisecond! No more clapping required.
    clap_sync_offset = 0.0
    
    # Uncomment the line below to run the fusion once you have the files
    # fuse_sensor_and_vision_data(yolo_file, sensor_file, out_file, clap_sync_offset)
    print("Fusion script ready. Provide the files and the clap sync offset, then uncomment the function call.")
