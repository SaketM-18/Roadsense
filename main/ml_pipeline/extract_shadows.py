import os
import cv2
import json
import pandas as pd
import numpy as np
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = os.path.join(BASE_DIR, 'vehicle_profile.json')
DATASET_DIR = os.path.join(BASE_DIR, 'ml_pipeline', 'shadow_dataset')

def setup_dataset():
    """Create YOLO background image dataset structure."""
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        
    os.makedirs(os.path.join(DATASET_DIR, 'images', 'train'))
    os.makedirs(os.path.join(DATASET_DIR, 'labels', 'train'))
    
    yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/train

names:
  0: pothole
"""
    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

def extract_shadows(folder_path):
    print(f"Extracting shadows from {folder_path}...")
    csv_path = os.path.join(folder_path, 'fused_labeled.csv')
    video_path = os.path.join(folder_path, 'stabilized.mp4')
    
    if not os.path.exists(csv_path) or not os.path.exists(video_path):
        print("Error: fused_labeled.csv or stabilized.mp4 not found.")
        return
        
    if not os.path.exists(PROFILE_PATH):
        print("Error: vehicle_profile.json not found.")
        return
        
    df = pd.read_csv(csv_path)
    with open(PROFILE_PATH, 'r') as f:
        profile = json.load(f)
        
    # Re-calculate Z-Scores
    bins = [0, 2, 4, 6, 8, 10, 15, 20, 100]
    labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15-20', '20+']
    df['speed_bin'] = pd.cut(df['speed'], bins=bins, labels=labels, include_lowest=True)
    
    false_positive_times = []
    
    for idx, row in df.iterrows():
        if row['Source'] == 'YOLO' and row['speed'] > 0.5:
            bin_label = str(row['speed_bin'])
            if bin_label in profile:
                mean_z = profile[bin_label]['mean']
                std_z = profile[bin_label]['std']
                z_score = abs(row['accel_z'] - mean_z) / (std_z + 1e-6)
                
                # If YOLO detected a pothole, but Z-score < 1.0 (smooth road)
                if z_score < 1.0:
                    false_positive_times.append(row['time_s'])

    if not false_positive_times:
        print("No hard negative shadows found.")
        return
        
    print(f"Found {len(false_positive_times)} false positive frames (Shadows/Flat roads). Extracting...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
        
    extracted_count = 0
    for time_s in false_positive_times:
        frame_idx = int(time_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Apply CLAHE just like YOLO does in process_ride
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            img_name = f"shadow_{extracted_count}.jpg"
            img_path = os.path.join(DATASET_DIR, 'images', 'train', img_name)
            cv2.imwrite(img_path, final_frame)
            
            # Create empty text file (Background Image / Hard Negative)
            txt_path = os.path.join(DATASET_DIR, 'labels', 'train', img_name.replace('.jpg', '.txt'))
            open(txt_path, 'a').close()
            
            extracted_count += 1
            
    cap.release()
    print(f"Successfully extracted {extracted_count} shadow frames into {DATASET_DIR}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_shadows.py <folder_path>")
        sys.exit(1)
        
    folder = sys.argv[1]
    if not os.path.isabs(folder):
        folder = os.path.join(BASE_DIR, folder)
        
    setup_dataset()
    extract_shadows(folder)
