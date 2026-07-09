"""
Road Health — One-Command Data Pipeline
========================================
Usage:
    python process_ride.py <folder_path> [--retrain]

Examples:
    python process_ride.py "Videos/Video3"
    python process_ride.py "Videos/Video3" --retrain

This script will:
1. Find the .mp4 and .csv files in the folder
2. Stabilize the shaky video
3. Run YOLOv8 pothole detection on the stabilized video
4. Fuse the YOLO timestamps with the sensor CSV
5. Append the new labeled data to the master training dataset
6. Optionally retrain the ML model (with --retrain flag)
"""
import cv2
import os
import sys
import numpy as np
import pandas as pd
from ultralytics import YOLO

BASE_DIR = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health'
MODEL_FILE = os.path.join(BASE_DIR, 'cv_pipeline', 'pothole_yolov8.pt')
MASTER_CSV = os.path.join(BASE_DIR, 'cv_pipeline', 'master_training_data.csv')


def find_files(folder):
    """Find the video (.mp4/.webm) and CSV files in the folder."""
    files = os.listdir(folder)
    video = None
    csv_file = None
    
    for f in files:
        lower = f.lower()
        if lower.endswith(('.mp4', '.webm', '.avi', '.mov')) and 'stabilized' not in lower:
            video = os.path.join(folder, f)
        if lower.endswith('.csv') and 'fused' not in lower and 'stabilized' not in lower:
            csv_file = os.path.join(folder, f)
    
    return video, csv_file


def stabilize_video(input_path, output_path):
    """Apply optical-flow based video stabilization."""
    print("\n[STEP 1/4] Stabilizing video...")
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Input: {total} frames, {w}x{h}, {fps:.1f} FPS")
    
    # Pass 1: Read frames and compute transforms
    frames = []
    ret, prev = cap.read()
    frames.append(prev)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    transforms = []
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if pts is not None:
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
            idx = np.where(status == 1)[0]
            if len(idx) >= 3:
                m, _ = cv2.estimateAffinePartial2D(pts[idx], pts_next[idx])
                if m is not None:
                    transforms.append((m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0])))
                else:
                    transforms.append((0, 0, 0))
            else:
                transforms.append((0, 0, 0))
        else:
            transforms.append((0, 0, 0))
        
        frames.append(curr)
        prev_gray = curr_gray
        
        if len(frames) % 500 == 0:
            print(f"    ... {len(frames)}/{total} frames analyzed")
    
    cap.release()
    
    # Pass 2: Smooth trajectory
    trajectory = np.cumsum(transforms, axis=0)
    RADIUS = 30
    smoothed = np.copy(trajectory)
    for i in range(trajectory.shape[1]):
        kernel = np.ones(2 * RADIUS + 1) / (2 * RADIUS + 1)
        padded = np.pad(trajectory[:, i], RADIUS, mode='edge')
        smoothed[:, i] = np.convolve(padded, kernel, mode='valid')
    
    corrections = smoothed - trajectory
    transforms_corrected = np.array(transforms) + corrections
    
    # Pass 3: Write stabilized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    out.write(frames[0])
    
    for i in range(len(transforms_corrected)):
        dx, dy, da = transforms_corrected[i]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy]
        ], dtype=np.float64)
        out.write(cv2.warpAffine(frames[i + 1], m, (w, h)))
    
    out.release()
    print(f"  Stabilized video saved: {output_path}")


def run_yolo(video_path, conf_threshold=0.5):
    """Run YOLOv8 pothole detection on the video."""
    print("\n[STEP 2/4] Running YOLO pothole detection...")
    model = YOLO(MODEL_FILE)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    mask_y = int(h * 0.75) # Mask out bottom 25% (scooter)
    
    detections = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 15 == 0:  # Sample every 15th frame (~2/sec)
            masked = frame.copy()
            cv2.rectangle(masked, (0, mask_y), (w, h), (0, 0, 0), -1)
            results = model.predict(masked, conf=conf_threshold, verbose=False)
            for r in results:
                for box in r.boxes:
                    y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                    if y_center < mask_y:
                        conf = float(box.conf[0])
                        time_sec = frame_idx / fps
                        detections.append((time_sec, conf))
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    ... {frame_idx}/{total} frames scanned")
    
    cap.release()
    print(f"  Found {len(detections)} pothole detections (conf > {conf_threshold})")
    return detections


def fuse_data(detections, csv_path, output_path, camera_delay=0.5):
    """Fuse YOLO timestamps with sensor CSV to create labeled training data."""
    print("\n[STEP 3/4] Fusing camera detections with sensor data...")
    df = pd.read_csv(csv_path)
    df['Label'] = 0
    
    for t, conf in detections:
        impact_time = t + camera_delay
        mask = (df['time_s'].astype(float) >= impact_time - 0.5) & \
               (df['time_s'].astype(float) <= impact_time + 0.5)
        df.loc[mask, 'Label'] = 1
    
    df.to_csv(output_path, index=False)
    
    pothole_rows = (df['Label'] == 1).sum()
    normal_rows = (df['Label'] == 0).sum()
    print(f"  Pothole rows: {pothole_rows}")
    print(f"  Normal rows:  {normal_rows}")
    print(f"  Saved: {output_path}")
    
    return df


def append_to_master(new_df):
    """Append the new labeled data to the master training dataset."""
    print("\n[STEP 4/4] Appending to master training dataset...")
    
    if os.path.exists(MASTER_CSV):
        master = pd.read_csv(MASTER_CSV)
        combined = pd.concat([master, new_df], ignore_index=True)
    else:
        combined = new_df
    
    combined.to_csv(MASTER_CSV, index=False)
    
    total_p = (combined['Label'] == 1).sum()
    total_n = (combined['Label'] == 0).sum()
    print(f"  Master dataset now has {len(combined)} total rows")
    print(f"  Pothole: {total_p} ({total_p/len(combined)*100:.1f}%)")
    print(f"  Normal:  {total_n} ({total_n/len(combined)*100:.1f}%)")


def retrain_model():
    """Retrain the ML model on the updated master dataset."""
    print("\n[BONUS] Retraining ML model...")
    train_script = os.path.join(BASE_DIR, 'ml_pipeline', 'train_fusion_model.py')
    os.system(f'python "{train_script}"')


# ============ MAIN ============
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_ride.py <folder_path> [--retrain]")
        print("Example: python process_ride.py Videos/Video3")
        sys.exit(1)
    
    folder = sys.argv[1]
    # Make path absolute if relative
    if not os.path.isabs(folder):
        folder = os.path.join(BASE_DIR, folder)
    
    do_retrain = '--retrain' in sys.argv
    
    print("=" * 60)
    print("ROAD HEALTH — Automated Data Pipeline")
    print("=" * 60)
    print(f"Processing folder: {folder}")
    
    # Find files
    video_file, csv_file = find_files(folder)
    if not video_file:
        print("ERROR: No video file (.mp4/.webm) found in folder!")
        sys.exit(1)
    if not csv_file:
        print("ERROR: No CSV file found in folder!")
        sys.exit(1)
    
    print(f"  Video: {os.path.basename(video_file)}")
    print(f"  CSV:   {os.path.basename(csv_file)}")
    
    # Run pipeline
    stab_path = os.path.join(folder, 'stabilized.mp4')
    stabilize_video(video_file, stab_path)
    
    detections = run_yolo(stab_path)
    
    fused_path = os.path.join(folder, 'fused_labeled.csv')
    new_data = fuse_data(detections, csv_file, fused_path)
    
    append_to_master(new_data)
    
    if do_retrain:
        retrain_model()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    if not do_retrain:
        print("Tip: Add --retrain flag to also retrain the model:")
        print(f'  python process_ride.py "{sys.argv[1]}" --retrain')
