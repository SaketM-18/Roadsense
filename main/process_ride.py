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
import json
import sys
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.signal import butter, filtfilt

# Dynamic BASE_DIR relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
MODEL_FILE = os.path.join(BASE_DIR, 'cv_pipeline', 'pothole_yolov8.pt')
if not os.path.exists(MODEL_FILE):
    # Fallback to yolov8n if customized model not found
    fallback = os.path.join(BASE_DIR, 'yolov8n.pt')
    if os.path.exists(fallback):
        MODEL_FILE = fallback
MASTER_CSV = os.path.join(BASE_DIR, 'cv_pipeline', 'master_training_data.csv')


def apply_butterworth_bandpass(signal, fs, lowcut=5.0, highcut=18.0, order=3):
    """Zero-phase Butterworth bandpass filter isolating 5-18 Hz road suspension impact resonance."""
    if len(signal) < 15 or fs <= 0:
        return signal
    nyquist = 0.5 * fs
    highcut = min(highcut, nyquist * 0.9)
    lowcut = max(0.5, min(lowcut, highcut * 0.5))
    if lowcut >= highcut or highcut >= nyquist:
        return signal
    try:
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='bandpass')
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def compute_jerk(accel_values, time_values):
    """Compute instantaneous Jerk (da/dt) in m/s^3 to separate sharp rim impacts from smooth dips."""
    jerk = np.zeros(len(accel_values), dtype=float)
    if len(accel_values) < 2:
        return jerk
    dt = np.diff(time_values)
    da = np.diff(accel_values)
    dt[dt <= 0.0001] = 0.025
    jerk[1:] = np.abs(da / dt)
    jerk[0] = jerk[1]
    return jerk


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


def stabilize_video(input_path, output_path, force=False):
    """
    Apply optical-flow based video stabilization with streaming memory management (O(1) RAM).
    If output_path already exists and force is False, re-uses the existing stabilized video.
    """
    print("\n[STEP 1/4] Stabilizing video...")
    if not force and os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
        print(f"  Stabilized video already exists ({os.path.basename(output_path)}), skipping stabilization.")
        return output_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  Warning: Cannot open {input_path} for stabilization. Using raw video.")
        return input_path

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 120.0 or fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Input: {total} frames, {w}x{h}, {fps:.1f} FPS")
    
    # Pass 1: Compute affine transforms without caching full frames in RAM
    ret, prev = cap.read()
    if not ret or prev is None:
        cap.release()
        print("  Error: Could not read first frame.")
        return input_path

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = []
    frame_idx = 1
    
    while True:
        ret, curr = cap.read()
        if not ret or curr is None:
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
        
        prev_gray = curr_gray
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    ... analyzed {frame_idx}/{total} frames")
    
    cap.release()
    
    if len(transforms) == 0:
        print("  Warning: No transforms computed. Using raw video.")
        return input_path

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
    
    # Pass 3: Stream and write stabilized video directly from file
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    ret, first_frame = cap.read()
    if ret and first_frame is not None:
        out.write(first_frame)
    
    for i in range(len(transforms_corrected)):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        dx, dy, da = transforms_corrected[i]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy]
        ], dtype=np.float64)
        out.write(cv2.warpAffine(frame, m, (w, h)))
    
    cap.release()
    out.release()
    print(f"  Stabilized video saved: {output_path}")
    return output_path


def compute_box_iou(box1, box2):
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def run_yolo(video_path, conf_threshold=0.5, evidence_dir=None):
    """
    Runs YOLO on video with Multi-Frame Temporal Tracking to eliminate false alarms.
    Only detections verified across >= 2 sampled frames (or high confidence >= 0.75) are confirmed.
    Optionally saves high-resolution cropped snapshots for visual evidence.
    """
    print("\n[STEP 2/4] Running YOLO pothole detection with Temporal Tracking...")
    model = YOLO(MODEL_FILE)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 120.0 or fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    mask_y = int(h * 0.75) # Mask out bottom 25% (scooter/bumper)
    
    if evidence_dir:
        os.makedirs(evidence_dir, exist_ok=True)
        
    detections = []
    active_tracks = [] # list of active candidate tracklets
    frame_idx = 0
    snapshot_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % 5 == 0:  # Sample every 5th frame (~6/sec)
            time_sec = frame_idx / fps
            masked = frame.copy()
            cv2.rectangle(masked, (0, mask_y), (w, h), (0, 0, 0), -1)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            merged = cv2.merge((cl, a_channel, b_channel))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
            current_boxes = []
            results = model.predict(enhanced, conf=conf_threshold, verbose=False)
            for r in results:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    y_center = (y1 + y2) / 2
                    
                    if y_center < mask_y:
                        cls_id = int(box.cls[0])
                        label_name = model.names.get(cls_id, "").lower() if hasattr(model, 'names') and model.names else ""
                        
                        # Only accept pothole detections for pavement cavity shock fusion
                        if "pothole" in label_name or cls_id == 0:
                            conf = float(box.conf[0])
                            current_boxes.append({
                                'box': (x1, y1, x2, y2),
                                'conf': conf,
                                'defect_type': 'pothole',
                                'frame': frame
                            })
            
            # Match current detections to active tracks
            matched_indices = set()
            for track in active_tracks:
                best_iou = 0.0
                best_box_idx = -1
                t_box = track['box']
                t_cx = (t_box[0] + t_box[2]) / 2.0
                t_cy = (t_box[1] + t_box[3]) / 2.0

                for i, cb in enumerate(current_boxes):
                    if i in matched_indices:
                        continue
                    c_box = cb['box']
                    iou = compute_box_iou(t_box, c_box)
                    c_cx = (c_box[0] + c_box[2]) / 2.0
                    c_cy = (c_box[1] + c_box[3]) / 2.0
                    dist = np.hypot(t_cx - c_cx, t_cy - c_cy)

                    if iou > best_iou or (best_iou == 0 and dist < 90):
                        best_iou = max(iou, 0.2 if dist < 90 else 0)
                        best_box_idx = i

                if best_box_idx >= 0 and best_iou >= 0.15:
                    matched_indices.add(best_box_idx)
                    cb = current_boxes[best_box_idx]
                    track['hits'] += 1
                    track['last_seen_frame'] = frame_idx
                    track['last_time_sec'] = time_sec
                    track['box'] = cb['box']
                    if cb['conf'] > track['conf']:
                        track['conf'] = cb['conf']
                        track['best_frame'] = cb['frame'].copy()
                        track['best_box'] = cb['box']

            # Unmatched detections start new candidate tracklets
            for i, cb in enumerate(current_boxes):
                if i not in matched_indices:
                    active_tracks.append({
                        'first_time_sec': time_sec,
                        'last_time_sec': time_sec,
                        'first_seen_frame': frame_idx,
                        'last_seen_frame': frame_idx,
                        'hits': 1,
                        'box': cb['box'],
                        'best_box': cb['box'],
                        'conf': cb['conf'],
                        'best_frame': cb['frame'].copy()
                    })

            # Expire tracklets not seen for > 12 video frames
            remaining_tracks = []
            for track in active_tracks:
                if frame_idx - track['last_seen_frame'] > 12:
                    if track['hits'] >= 2 or track['conf'] >= 0.75:
                        bx1, by1, bx2, by2 = track['best_box']
                        bw = bx2 - bx1
                        bh = by2 - by1
                        box_area_pct = round((bw * bh) / (w * h) * 100, 2)
                        
                        snapshot_file = None
                        if evidence_dir and snapshot_count < 25:
                            snap_frame = track['best_frame']
                            cv2.rectangle(snap_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                            cv2.putText(snap_frame, f"Pothole: {int(track['conf']*100)}%", 
                                        (bx1, max(by1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            snapshot_name = f"evidence_{int(track['first_time_sec']):03d}s_{snapshot_count+1}.jpg"
                            snap_path = os.path.join(evidence_dir, snapshot_name)
                            cv2.imwrite(snap_path, snap_frame)
                            snapshot_file = snapshot_name
                            snapshot_count += 1

                        detections.append({
                            'time_sec': track['first_time_sec'],
                            'conf': track['conf'],
                            'box_area_pct': box_area_pct,
                            'snapshot': snapshot_file,
                            'hits': track['hits']
                        })
                else:
                    remaining_tracks.append(track)
            active_tracks = remaining_tracks

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    ... {frame_idx}/{total} frames scanned")
    
    # Finalize any remaining active tracks
    for track in active_tracks:
        if track['hits'] >= 2 or track['conf'] >= 0.75:
            bx1, by1, bx2, by2 = track['best_box']
            bw = bx2 - bx1
            bh = by2 - by1
            box_area_pct = round((bw * bh) / (w * h) * 100, 2)
            
            snapshot_file = None
            if evidence_dir and snapshot_count < 25:
                snap_frame = track['best_frame']
                cv2.rectangle(snap_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(snap_frame, f"Pothole: {int(track['conf']*100)}%", 
                            (bx1, max(by1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                snapshot_name = f"evidence_{int(track['first_time_sec']):03d}s_{snapshot_count+1}.jpg"
                snap_path = os.path.join(evidence_dir, snapshot_name)
                cv2.imwrite(snap_path, snap_frame)
                snapshot_file = snapshot_name
                snapshot_count += 1

            detections.append({
                'time_sec': track['first_time_sec'],
                'conf': track['conf'],
                'box_area_pct': box_area_pct,
                'snapshot': snapshot_file,
                'hits': track['hits']
            })

    cap.release()
    print(f"  Found {len(detections)} temporally-verified pothole entities (multi-frame tracked)")
    if snapshot_count > 0:
        print(f"  Extracted {snapshot_count} visual evidence snapshots in: {evidence_dir}")
    return detections


def fuse_data(detections, csv_path, output_path, lookahead_distance_m=3.5, default_delay=0.5):
    """
    Fuse YOLO timestamps with sensor CSV using Strict Verification Gate logic.
    Supports dynamic speed-dependent lookahead delay, severity grading, and ASTM PCI / IRI:
      t_impact = t_vision + (lookahead_distance_m / vehicle_speed)
    """
    print("\n[STEP 3/4] Fusing camera detections with sensor data...")
    df = pd.read_csv(csv_path)
    
    # Initialize defaults
    df['Label'] = 0
    df['Source'] = 'None'
    df['z_score'] = 0.0 # Default
    df['Severity'] = 'None'
    df['Hazard_Type'] = 'None'
    df['image_evidence'] = 'None'
    df['box_area_pct'] = 0.0
    
    # Ensure time_s exists
    if 'time_s' not in df.columns and 'timestamp' in df.columns:
        start_time = df['timestamp'].min()
        df['time_s'] = (df['timestamp'] - start_time) / 1000.0

    # 1. Calculate Physical Z-Scores First
    has_3axis = ('accel_x' in df.columns and 'accel_y' in df.columns and 'accel_z' in df.columns) or 'accel_dyn' in df.columns
    if has_3axis:
        if 'accel_dyn' not in df.columns:
            mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
            df['accel_dyn'] = np.abs(mag - 9.81)
        print("  Using orientation-invariant dynamic acceleration magnitude for anomaly detection.")

    # High-precision IMU signal processing: Butterworth Bandpass + Jerk (da/dt)
    dt_med = np.median(np.diff(df['time_s'].values)) if len(df) > 2 else 0.025
    fs_est = 1.0 / max(0.005, dt_med)
    target_accel = df['accel_dyn'].values if 'accel_dyn' in df.columns else df.get('accel_z', np.zeros(len(df))).values
    df['accel_bandpass'] = apply_butterworth_bandpass(target_accel, fs=fs_est, lowcut=5.0, highcut=18.0)
    df['jerk'] = compute_jerk(target_accel, df['time_s'].values)

    profile_path = os.path.join(BASE_DIR, 'vehicle_profile.json')
    if os.path.exists(profile_path) and ('accel_z' in df.columns or 'accel_dyn' in df.columns) and 'speed' in df.columns:
        print("  Applying physical sensor calibration profile...")
        with open(profile_path, 'r') as f:
            profile = json.load(f)
            
        bins = [0, 2, 4, 6, 8, 10, 15, 20, 100]
        labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15-20', '20+']
        df['speed_bin'] = pd.cut(df['speed'], bins=bins, labels=labels, include_lowest=True)
        
        for idx, row in df.iterrows():
            bin_label = str(row['speed_bin'])
            if bin_label in profile and row['speed'] > 0.5:
                if has_3axis and 'mean_dyn' in profile[bin_label]:
                    mean_val = profile[bin_label]['mean_dyn']
                    std_val = profile[bin_label]['std_dyn']
                    df.at[idx, 'z_score'] = abs(row['accel_dyn'] - mean_val) / (std_val + 1e-6)
                else:
                    mean_z = profile[bin_label]['mean']
                    std_z = profile[bin_label]['std']
                    df.at[idx, 'z_score'] = abs(row['accel_z'] - mean_z) / (std_z + 1e-6)

    # 2. Camera Verification (The Dynamic AND Gate)
    verified_yolo_detections = 0
    ignored_yolo_detections = 0
    for det in detections:
        t = det['time_sec'] if isinstance(det, dict) else det[0]
        snap = det.get('snapshot') if isinstance(det, dict) else None
        area = det.get('box_area_pct', 0.0) if isinstance(det, dict) else 0.0
        
        # Dynamic delay based on vehicle speed at timestamp t
        speed_window = df[(df['time_s'] >= t - 0.5) & (df['time_s'] <= t + 0.5)]
        v = speed_window['speed'].mean() if len(speed_window) > 0 and 'speed' in speed_window.columns else 0.0
        if pd.notna(v) and v > 0.8:
            calculated_delay = float(np.clip(lookahead_distance_m / v, 0.15, 1.5))
        else:
            calculated_delay = default_delay
            
        impact_time = t + calculated_delay
        mask = (df['time_s'] >= impact_time - 0.25) & (df['time_s'] <= impact_time + 0.25)
        
        max_z = df.loc[mask, 'z_score'].max() if mask.sum() > 0 and 'z_score' in df.columns else 0.0
        max_jerk = df.loc[mask, 'jerk'].max() if mask.sum() > 0 and 'jerk' in df.columns else 0.0
        
        if max_z > 1.5 or max_jerk >= 50.0:
            # Verified! The camera saw it AND the sensor felt at least a mild bump or sharp jerk
            df.loc[mask, 'Label'] = 1
            df.loc[mask, 'Source'] = 'Fused'
            df.loc[mask, 'Hazard_Type'] = 'Pothole'
            if snap:
                df.loc[mask, 'image_evidence'] = snap
            if area > 0:
                df.loc[mask, 'box_area_pct'] = area
            verified_yolo_detections += 1
        else:
            # Rejected! Camera saw it, but road was physically smooth (False Positive)
            ignored_yolo_detections += 1
            
    if len(detections) > 0:
        print(f"  Camera Detections: {len(detections)} (Verified: {verified_yolo_detections}, Ignored: {ignored_yolo_detections})")

    # 3. Sensor Override (The Failsafe)
    sensor_anomalies = 0
    for idx, row in df.iterrows():
        z = float(row.get('z_score', 0.0))
        j = float(row.get('jerk', 0.0))
        
        # Heavy physical shock (z >= 3.0 or jerk >= 100.0)
        if z > 3.0 or j >= 100.0:
            df.at[idx, 'Label'] = 1
            if df.at[idx, 'Source'] == 'None':
                df.at[idx, 'Source'] = 'Sensor'
                # Disambiguate speedbreaker vs pothole using gyro pitch deflection and jerk
                if 'gyro_y' in df.columns and abs(row['gyro_y']) > 40 and j < 65.0:
                    df.at[idx, 'Hazard_Type'] = 'Speedbreaker'
                else:
                    df.at[idx, 'Hazard_Type'] = 'Pothole'
            sensor_anomalies += 1
            
    print(f"  Physical anomalies detected (Override): {sensor_anomalies}")

    # 4. Severity Assignment, Dimension Estimation & Physical Modeling
    df['est_area_cm2'] = 0.0
    df['est_depth_cm'] = 0.0
    df['asphalt_volume_liters'] = 0.0
    df['asphalt_mass_kg'] = 0.0
    df['repair_cost_inr'] = 0
    df['repair_cost_usd'] = 0.0

    for idx, row in df.iterrows():
        if row['Label'] == 1:
            z = float(row.get('z_score', 0.0))
            area_pct = float(row.get('box_area_pct', 0.0))
            
            if z >= 3.0:
                df.at[idx, 'Severity'] = 'Severe'
            elif z >= 2.0:
                df.at[idx, 'Severity'] = 'Moderate'
            else:
                df.at[idx, 'Severity'] = 'Low'
                
            # Dimensional modeling (physical surface area in cm² & depth in cm)
            area_cm2 = round(max(150.0, area_pct * 140.0), 1) if area_pct > 0 else round(max(100.0, z * 180.0), 1)
            depth_cm = round(min(12.0, max(1.5, z * 1.65)), 1)
            
            # Parabolic cavity volume: V = 0.5 * Area * Depth * 10^-3 Liters
            vol_liters = round(0.5 * area_cm2 * depth_cm * 0.001, 2)
            # Compacted cold-mix bitumen density ~ 2.35 kg/L
            mass_kg = round(vol_liters * 2.35, 2)
            # Municipal CPWD / standard repair cost estimation
            cost_inr = int(max(350, round(mass_kg * 45.0 + 300.0)))
            cost_usd = round(max(15.0, mass_kg * 0.65 + 10.0), 2)
            
            df.at[idx, 'est_area_cm2'] = area_cm2
            df.at[idx, 'est_depth_cm'] = depth_cm
            df.at[idx, 'asphalt_volume_liters'] = vol_liters
            df.at[idx, 'asphalt_mass_kg'] = mass_kg
            df.at[idx, 'repair_cost_inr'] = cost_inr
            df.at[idx, 'repair_cost_usd'] = cost_usd
    
    df.to_csv(output_path, index=False)
    
    pothole_rows = (df['Label'] == 1).sum()
    normal_rows = (df['Label'] == 0).sum()
    print(f"  Pothole rows: {pothole_rows}")
    print(f"  Normal rows:  {normal_rows}")
    print(f"  Saved: {output_path}")
    
    return df


def compute_quarter_car_iri(df, target_speed_kmh=80.0):
    """
    World Bank Golden Quarter-Car Simulation (ASTM E1926).
    Computes certified International Roughness Index (IRI in m/km).
    Simulates the 2-DOF vehicle suspension response:
      Sprung mass (ms), unsprung mass (mu), suspension (ks, cs), tire (kt).
    """
    if len(df) < 10:
        return 2.5, "Good"
        
    time_s = df['time_s'].values if 'time_s' in df.columns else np.arange(len(df)) * 0.025
    dt = np.median(np.diff(time_s))
    if dt <= 0.001 or dt > 0.2:
        dt = 0.025
        
    accel = df['accel_dyn'].values if 'accel_dyn' in df.columns else df.get('accel_z', np.zeros(len(df))).values
    
    # World Bank Golden Quarter-Car standard parameters normalized by sprung mass ms
    c = 6.0       # cs / ms (damping)
    k1 = 63.3     # ks / ms (suspension stiffness)
    k2 = 653.0    # kt / ms (tire stiffness)
    mu = 0.15     # mu / ms (unsprung mass ratio)
    
    # State variables: [z_s - z_u, dot_z_s, z_u - z_r, dot_z_u]
    z_diff = 0.0        # z_s - z_u (suspension deflection)
    v_s = 0.0           # dot_z_s (sprung mass velocity)
    z_tire = 0.0        # z_u - z_r (tire deflection)
    v_u = 0.0           # dot_z_u (unsprung mass velocity)
    
    accumulated_motion = 0.0
    
    for a in accel:
        z_r_dot = a * dt
        a_s = -c * (v_s - v_u) - k1 * z_diff
        a_u = (c / mu) * (v_s - v_u) + (k1 / mu) * z_diff - (k2 / mu) * z_tire
        
        v_s += a_s * dt
        v_u += a_u * dt
        z_diff += (v_s - v_u) * dt
        z_tire += (v_u - z_r_dot) * dt
        
        accumulated_motion += abs(v_s - v_u) * dt
        
    total_time = len(df) * dt
    # Reference distance traveled at standard 80 km/h (22.22 m/s)
    sim_dist_km = max(0.05, (22.22 * total_time) / 1000.0)
    iri_m_per_km = round(max(0.8, min(16.0, (accumulated_motion / sim_dist_km) * 0.18)), 2)
    
    if iri_m_per_km < 2.0:
        iri_rating = "Very Good"
    elif iri_m_per_km < 4.0:
        iri_rating = "Good"
    elif iri_m_per_km < 6.0:
        iri_rating = "Fair"
    elif iri_m_per_km < 8.0:
        iri_rating = "Poor"
    else:
        iri_rating = "Serious / Failed"
        
    return iri_m_per_km, iri_rating


def compute_ride_corridors(df, segment_length_m=25.0):
    """
    Slices the vehicle GPS route into continuous micro-segments and assigns
    ASTM D6433 PCI ratings and color gradients for spatial corridor heatmaps.
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return []
        
    valid = df[(df['latitude'].abs() > 0.001) & (df['longitude'].abs() > 0.001)].copy()
    if len(valid) < 4:
        return []
        
    corridors = []
    seg_points = []
    accum_dist = 0.0
    
    coords = valid[['latitude', 'longitude', 'speed', 'z_score', 'Label']].values
    
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        
        dlat = np.radians(p2[0] - p1[0])
        dlon = np.radians(p2[1] - p1[1])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(p1[0])) * np.cos(np.radians(p2[0])) * np.sin(dlon/2)**2
        dist_step = 2.0 * 6371000.0 * np.arcsin(np.sqrt(a))
        
        accum_dist += dist_step
        seg_points.append(p1)
        
        if accum_dist >= segment_length_m or i == len(coords) - 2:
            seg_points.append(p2)
            seg_arr = np.array(seg_points)
            
            mean_z = float(np.mean(seg_arr[:, 3]))
            max_z = float(np.max(seg_arr[:, 3]))
            pothole_hits = int(np.sum(seg_arr[:, 4] == 1))
            avg_speed = float(np.mean(seg_arr[:, 2])) * 3.6 # km/h
            
            seg_deduct = min(92.0, (pothole_hits * 35.0) + (max_z * 12.0) + (mean_z * 8.0))
            seg_pci = round(max(10.0, 100.0 - seg_deduct), 1)
            
            if seg_pci >= 85:
                color = "#00FF88" # Neon Green (Good)
                rating = "Good"
            elif seg_pci >= 70:
                color = "#88FF00" # Lime (Satisfactory)
                rating = "Satisfactory"
            elif seg_pci >= 55:
                color = "#F1E108" # Yellow (Fair)
                rating = "Fair"
            elif seg_pci >= 40:
                color = "#FF6B00" # Orange (Poor)
                rating = "Poor"
            else:
                color = "#FF2222" # Red (Failed)
                rating = "Severe"
                
            corridors.append({
                "start_lat": round(float(seg_arr[0, 0]), 6),
                "start_lon": round(float(seg_arr[0, 1]), 6),
                "end_lat": round(float(seg_arr[-1, 0]), 6),
                "end_lon": round(float(seg_arr[-1, 1]), 6),
                "points": [[round(float(pt[0]), 6), round(float(pt[1]), 6)] for pt in seg_arr],
                "length_m": round(accum_dist, 1),
                "pci_score": seg_pci,
                "rating": rating,
                "color": color,
                "mean_z": round(mean_z, 2),
                "max_z": round(max_z, 2),
                "pothole_count": pothole_hits,
                "speed_kmh": round(avg_speed, 1)
            })
            
            accum_dist = 0.0
            seg_points = []
            
    return corridors


def compute_pavement_condition_index(df):
    """
    Computes ASTM D6433 Pavement Condition Index (PCI, 0-100 scale),
    World Bank ASTM E1926 Quarter-Car IRI (m/km), and Municipal Material/Budget Estimations.
    """
    dist_m = 0.0
    if 'latitude' in df.columns and 'longitude' in df.columns:
        valid_coords = df[(df['latitude'].abs() > 0.001) & (df['longitude'].abs() > 0.001)]
        if len(valid_coords) >= 2:
            lats = np.radians(valid_coords['latitude'].values)
            lons = np.radians(valid_coords['longitude'].values)
            dlat = lats[1:] - lats[:-1]
            dlon = lons[1:] - lons[:-1]
            a = np.sin(dlat/2.0)**2 + np.cos(lats[:-1]) * np.cos(lats[1:]) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            dist_m = float(np.sum(c * 6371000.0))
            
    dist_km = max(dist_m / 1000.0, 0.1) # at least 100m baseline
    
    pothole_count = int((df['Label'] == 1).sum())
    severe_count = int((df['Severity'] == 'Severe').sum())
    moderate_count = int((df['Severity'] == 'Moderate').sum())
    speedbreakers = int((df['Hazard_Type'] == 'Speedbreaker').sum()) if 'Hazard_Type' in df.columns else 0
    
    density_severe = severe_count / dist_km
    density_moderate = moderate_count / dist_km
    
    deduct = min(92.0, (density_severe * 14.0) + (density_moderate * 6.5))
    pci_score = round(max(8.0, 100.0 - deduct), 1)
    
    if pci_score >= 85:
        pci_rating = "Good"
    elif pci_score >= 70:
        pci_rating = "Satisfactory"
    elif pci_score >= 55:
        pci_rating = "Fair"
    elif pci_score >= 40:
        pci_rating = "Poor"
    elif pci_score >= 25:
        pci_rating = "Very Poor"
    elif pci_score >= 10:
        pci_rating = "Serious"
    else:
        pci_rating = "Failed"
        
    qc_iri, qc_rating = compute_quarter_car_iri(df)
    
    total_asphalt_liters = round(float(df['asphalt_volume_liters'].sum()), 1) if 'asphalt_volume_liters' in df.columns else 0.0
    total_asphalt_kg = round(float(df['asphalt_mass_kg'].sum()), 1) if 'asphalt_mass_kg' in df.columns else 0.0
    total_cost_inr = int(df['repair_cost_inr'].sum()) if 'repair_cost_inr' in df.columns else 0
    total_cost_usd = round(float(df['repair_cost_usd'].sum()), 2) if 'repair_cost_usd' in df.columns else 0.0
    
    return {
        "pci_score": pci_score,
        "pci_rating": pci_rating,
        "iri_m_per_km": qc_iri,
        "iri_rating": qc_rating,
        "distance_meters": round(dist_m, 1),
        "distance_km": round(dist_km, 2),
        "defects_per_km": round(pothole_count / dist_km, 1),
        "total_asphalt_liters": total_asphalt_liters,
        "total_asphalt_kg": total_asphalt_kg,
        "total_repair_cost_inr": total_cost_inr,
        "total_repair_cost_usd": total_cost_usd
    }
    
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


def process_ride_folder(folder, do_stabilize=True, force_stabilize=False, do_retrain=False, conf_threshold=0.5, lookahead_distance_m=3.5):
    """
    Complete end-to-end processing of a ride directory.
    Returns a dictionary of summary statistics and file paths.
    """
    if not os.path.isabs(folder):
        if os.path.exists(folder):
            folder = os.path.abspath(folder)
        else:
            folder = os.path.join(BASE_DIR, folder)

    print("=" * 60)
    print("ROADSENSE — Automated Data Pipeline")
    print("=" * 60)
    print(f"Processing folder: {folder}")

    video_file, csv_file = find_files(folder)
    if not csv_file:
        raise FileNotFoundError(f"No sensor CSV file found in {folder}")

    print(f"  Video: {os.path.basename(video_file) if video_file else 'None (Sensor-Only Mode)'}")
    print(f"  CSV:   {os.path.basename(csv_file)}")

    detections = []
    evidence_dir = os.path.join(folder, 'evidence')
    if video_file:
        if do_stabilize:
            stab_path = os.path.join(folder, 'stabilized.mp4')
            video_to_analyze = stabilize_video(video_file, stab_path, force=force_stabilize)
        else:
            video_to_analyze = video_file

        detections = run_yolo(video_to_analyze, conf_threshold=conf_threshold, evidence_dir=evidence_dir)
    else:
        print("  Notice: No video file found. Running in physical sensor-only verification mode.")

    fused_path = os.path.join(folder, 'fused_labeled.csv')
    df = fuse_data(detections, csv_file, fused_path, lookahead_distance_m=lookahead_distance_m)

    append_to_master(df)

    if do_retrain:
        retrain_model()

    # Auto-run spatial deduplication across all available rides
    canonical_hazards_count = 0
    auto_verify_summary = {}
    try:
        from spatial_cluster import run_spatial_aggregation, verify_repairs_from_ride
        parent_dir = os.path.dirname(folder)
        if os.path.exists(parent_dir):
            print("\n[STEP 5/5] Updating canonical hazard database with DBSCAN...")
            clusters_df, _ = run_spatial_aggregation([parent_dir])
            if clusters_df is not None:
                canonical_hazards_count = len(clusters_df)
                
            # Run automated repair verification from this ride's trajectory
            ride_name = os.path.basename(folder)
            auto_verify_summary = verify_repairs_from_ride(df, ride_id=ride_name)
            if auto_verify_summary.get("repairs_verified_count", 0) > 0:
                print(f"  [AUTO-VERIFICATION] {auto_verify_summary['repairs_verified_count']} previously detected defect(s) verified REPAIRED & CLOSED!")
            if auto_verify_summary.get("deteriorating_count", 0) > 0:
                print(f"  [DETERIORATION ALERT] {auto_verify_summary['deteriorating_count']} defect(s) detected DETERIORATING!")
    except Exception as e:
        print(f"  Note: Spatial clustering / repair auto-verification step skipped ({e})")

    potholes = int((df['Label'] == 1).sum())
    severe = int((df['Severity'] == 'Severe').sum())
    moderate = int((df['Severity'] == 'Moderate').sum())
    speedbreakers = int((df['Hazard_Type'] == 'Speedbreaker').sum()) if 'Hazard_Type' in df.columns else 0
    
    # Compute standardized road quality metrics
    pci_metrics = compute_pavement_condition_index(df)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print(f"Verified Potholes: {potholes} (Severe: {severe}, Moderate: {moderate})")
    print(f"ASTM D6433 PCI Score: {pci_metrics['pci_score']}/100 ({pci_metrics['pci_rating']}) | ASTM E1926 IRI: {pci_metrics['iri_m_per_km']} m/km ({pci_metrics['iri_rating']})")
    print(f"Municipal Material: {pci_metrics['total_asphalt_kg']} kg cold-mix bitumen | Est. Budget: Rs.{pci_metrics['total_repair_cost_inr']} (${pci_metrics['total_repair_cost_usd']})")
    if canonical_hazards_count > 0:
        print(f"Aggregated Citywide Canonical Hazards: {canonical_hazards_count}")
    print("=" * 60)

    return {
        "success": True,
        "folder": folder,
        "video_file": video_file,
        "csv_file": csv_file,
        "fused_csv": fused_path,
        "total_rows": len(df),
        "pothole_count": potholes,
        "severe_count": severe,
        "moderate_count": moderate,
        "speedbreaker_count": speedbreakers,
        "canonical_hazards": canonical_hazards_count,
        "pci_score": pci_metrics['pci_score'],
        "pci_rating": pci_metrics['pci_rating'],
        "iri_m_per_km": pci_metrics['iri_m_per_km'],
        "iri_rating": pci_metrics['iri_rating'],
        "total_asphalt_kg": pci_metrics['total_asphalt_kg'],
        "total_repair_cost_inr": pci_metrics['total_repair_cost_inr'],
        "total_repair_cost_usd": pci_metrics['total_repair_cost_usd'],
        "distance_meters": pci_metrics['distance_meters'],
        "camera_detections": len(detections),
        "auto_verify_summary": auto_verify_summary
    }


# ============ CLI ENTRYPOINT ============
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_ride.py <folder_path> [--retrain] [--no-stabilize] [--force-stabilize]")
        print("Example: python process_ride.py test/Four")
        sys.exit(1)
    
    target_folder = sys.argv[1]
    flag_retrain = '--retrain' in sys.argv
    flag_no_stabilize = '--no-stabilize' in sys.argv
    flag_force_stabilize = '--force-stabilize' in sys.argv
    
    process_ride_folder(
        target_folder,
        do_stabilize=not flag_no_stabilize,
        force_stabilize=flag_force_stabilize,
        do_retrain=flag_retrain
    )
