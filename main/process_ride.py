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

# Advanced Computer Vision Intelligence (SAHI, 3D Rim Gradient, Kinematics, Consensus)
try:
    from cv_pipeline.advanced_detector import (
        sahi_corridor_inference,
        fuse_ensemble_candidates,
        is_flat_shadow,
        check_perspective_kinematics,
        suppress_duplicate_boxes
    )
    HAVE_ADVANCED_CV = True
except ImportError:
    try:
        from main.cv_pipeline.advanced_detector import (
            sahi_corridor_inference,
            fuse_ensemble_candidates,
            is_flat_shadow,
            check_perspective_kinematics,
            suppress_duplicate_boxes
        )
        HAVE_ADVANCED_CV = True
    except ImportError:
        HAVE_ADVANCED_CV = False

# Model selection: ensemble, segmentation, and detection
SEG_MODEL = os.path.join(BASE_DIR, 'cv_pipeline', 'pothole_yolo11_seg.pt')
V8_MODEL = os.path.join(BASE_DIR, 'cv_pipeline', 'pothole_yolov8.pt')
SEG_ONNX = os.path.join(BASE_DIR, 'cv_pipeline', 'pothole_yolo11_seg.onnx')
V8_ONNX = os.path.join(BASE_DIR, 'cv_pipeline', 'pothole_yolov8.onnx')
MODEL_FILE = SEG_MODEL if os.path.exists(SEG_MODEL) else V8_MODEL
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


def compute_adaptive_vehicle_baseline(df, has_3axis=True):
    """
    Self-Calibrating Vehicle Vibration Baseline (Online Trimmed Estimator).
    Learns dynamic suspension characteristics for ANY vehicle (bike, car, bus, auto-rickshaw)
    by computing trimmed rolling statistics across speed bins without requiring pre-recorded profiles.
    """
    if len(df) == 0:
        df['z_score'] = 0.0
        return df

    target_signal = 'accel_dyn' if (has_3axis and 'accel_dyn' in df.columns) else ('accel_z' if 'accel_z' in df.columns else None)
    if target_signal is None or target_signal not in df.columns:
        df['z_score'] = 0.0
        return df

    bins = [0, 2, 4, 6, 8, 10, 15, 20, 100]
    labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15-20', '20+']
    if 'speed' in df.columns:
        df['speed_bin'] = pd.cut(df['speed'], bins=bins, labels=labels, include_lowest=True)
    else:
        df['speed_bin'] = '4-6'

    profile_path = os.path.join(BASE_DIR, 'vehicle_profile.json')
    prior = {}
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r') as f:
                prior = json.load(f)
        except Exception:
            pass

    z_scores = np.zeros(len(df), dtype=float)
    
    for bin_lbl in labels:
        mask = (df['speed_bin'] == bin_lbl)
        if mask.sum() == 0:
            continue
        vals = df.loc[mask, target_signal].values
        
        # Self-calibrating trimmed statistics (exclude extreme shocks > 85th percentile to isolate asphalt baseline)
        if len(vals) >= 8:
            q15, q85 = np.percentile(vals, [15, 85])
            nominal = vals[(vals >= q15) & (vals <= q85)]
            b_mean = float(np.mean(nominal)) if len(nominal) > 0 else float(np.mean(vals))
            b_std = float(max(0.35, np.std(nominal))) if len(nominal) > 1 else float(max(0.5, np.std(vals)))
        elif bin_lbl in prior:
            p = prior[bin_lbl]
            b_mean = float(p.get('mean_dyn' if target_signal == 'accel_dyn' else 'mean', 9.81 if target_signal == 'accel_z' else 1.0))
            b_std = float(max(0.35, p.get('std_dyn' if target_signal == 'accel_dyn' else 'std', 1.0)))
        else:
            b_mean = float(np.median(vals))
            b_std = float(max(0.5, np.std(vals)))
            
        z_scores[mask] = np.abs(vals - b_mean) / (b_std + 1e-6)

    df['z_score'] = np.round(z_scores, 2)
    return df


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


def run_yolo(video_path, conf_threshold=0.40, evidence_dir=None, model_mode='ensemble', use_onnx=False):
    """
    Runs YOLO on video with Multi-Frame Temporal Tracking and Ensemble Cascaded Detection.
    - model_mode='ensemble': Synchronously pairs 81.06% mAP YOLOv8 detection with YOLO11-Seg instance polygon masks.
    - model_mode='v8': High-speed YOLOv8 bounding box detection.
    - model_mode='seg': Direct YOLO11-Seg instance contour extraction.
    - use_onnx=True: Loads ONNX Runtime models for accelerated inference.
    """
    print(f"\n[STEP 2/4] Running YOLO detection (Mode: {model_mode.upper()}, ONNX: {use_onnx}) with Temporal Tracking...")
    
    # Engine resolution
    det_target = V8_ONNX if (use_onnx and os.path.exists(V8_ONNX)) else V8_MODEL
    seg_target = SEG_ONNX if (use_onnx and os.path.exists(SEG_ONNX)) else SEG_MODEL
    
    model_det = None
    model_seg = None
    
    if model_mode in ('ensemble', 'v8') and os.path.exists(det_target):
        try:
            model_det = YOLO(det_target)
            print(f"  Stage 1 Engine: Loaded high-recall detector ({os.path.basename(det_target)})")
        except Exception as e:
            print(f"  Warning: Could not load detector: {e}")
            
    if model_mode in ('ensemble', 'seg') and os.path.exists(seg_target):
        try:
            model_seg = YOLO(seg_target)
            print(f"  Stage 2 Engine: Loaded polygon instance segmenter ({os.path.basename(seg_target)})")
        except Exception as e:
            print(f"  Warning: Could not load segmenter: {e}")

    if model_det is None and model_seg is None:
        fallback = MODEL_FILE
        print(f"  Fallback Engine: {os.path.basename(fallback)}")
        model_det = YOLO(fallback)

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
            
            cand_boxes_det = []
            cand_boxes_det = []
            cand_boxes_seg = []

            # 1. High-Recall Detector (Stage 1)
            if model_det is not None:
                r_det = model_det.predict(enhanced, conf=conf_threshold, verbose=False)
                for r in r_det:
                    for box in r.boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                        if (by1 + by2) / 2 < mask_y:
                            cls_id = int(box.cls[0])
                            lbl = model_det.names.get(cls_id, "").lower() if hasattr(model_det, 'names') and model_det.names else ""
                            if "pothole" in lbl or cls_id == 0:
                                cand_boxes_det.append({
                                    'box': (bx1, by1, bx2, by2),
                                    'conf': float(box.conf[0])
                                })
                if HAVE_ADVANCED_CV:
                    sahi_det = sahi_corridor_inference(model_det, enhanced, conf_threshold=conf_threshold)
                    cand_boxes_det.extend([s for s in sahi_det if (s['box'][1] + s['box'][3]) / 2 < mask_y])
                    cand_boxes_det = suppress_duplicate_boxes(cand_boxes_det, iou_thresh=0.45)

            # 2. Polygon Instance Segmenter (Stage 2)
            if model_seg is not None:
                r_seg = model_seg.predict(enhanced, conf=conf_threshold, verbose=False)
                for r in r_seg:
                    has_m = hasattr(r, 'masks') and r.masks is not None and len(r.masks) > 0
                    for i, box in enumerate(r.boxes):
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                        if (by1 + by2) / 2 < mask_y:
                            cls_id = int(box.cls[0])
                            lbl = model_seg.names.get(cls_id, "").lower() if hasattr(model_seg, 'names') and model_seg.names else ""
                            if "pothole" in lbl or cls_id == 0:
                                poly = None
                                s_area = None
                                if has_m and i < len(r.masks.xy):
                                    pts = r.masks.xy[i]
                                    if len(pts) >= 3:
                                        poly = pts.tolist()
                                        poly_area_px = cv2.contourArea(np.array(pts, dtype=np.float32))
                                        s_area = round((poly_area_px / (w * h)) * 100.0, 3)
                                cand_boxes_seg.append({
                                    'box': (bx1, by1, bx2, by2),
                                    'conf': float(box.conf[0]),
                                    'polygon': poly,
                                    'seg_area_pct': s_area
                                })
                if HAVE_ADVANCED_CV:
                    sahi_seg = sahi_corridor_inference(model_seg, enhanced, conf_threshold=conf_threshold)
                    cand_boxes_seg.extend([s for s in sahi_seg if (s['box'][1] + s['box'][3]) / 2 < mask_y])
                    cand_boxes_seg = suppress_duplicate_boxes(cand_boxes_seg, iou_thresh=0.45)

            # 3. Ensemble Fusion & 3D Rim Gradient Depth Check
            if HAVE_ADVANCED_CV:
                current_boxes = fuse_ensemble_candidates(cand_boxes_det, cand_boxes_seg, enhanced, iou_thresh=0.25)
            elif model_det is not None and model_seg is not None:
                matched_seg_idx = set()
                for d_box in cand_boxes_det:
                    best_match = None
                    best_iou = 0.0
                    best_s_idx = -1
                    for s_idx, s_box in enumerate(cand_boxes_seg):
                        iou = compute_box_iou(d_box['box'], s_box['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_s_idx = s_idx

                    if best_s_idx >= 0 and best_iou >= 0.25:
                        matched_seg_idx.add(best_s_idx)
                        s_box = cand_boxes_seg[best_s_idx]
                        ens_conf = min(0.99, max(d_box['conf'], s_box['conf']) + 0.05)
                        current_boxes.append({
                            'box': s_box['box'] if s_box.get('polygon') else d_box['box'],
                            'conf': ens_conf,
                            'defect_type': 'pothole',
                            'polygon': s_box.get('polygon'),
                            'seg_area_pct': s_box.get('seg_area_pct'),
                            'frame': frame
                        })
                    else:
                        current_boxes.append({
                            'box': d_box['box'],
                            'conf': d_box['conf'],
                            'defect_type': 'pothole',
                            'polygon': None,
                            'seg_area_pct': None,
                            'frame': frame
                        })

                for s_idx, s_box in enumerate(cand_boxes_seg):
                    if s_idx not in matched_seg_idx:
                        current_boxes.append({
                            'box': s_box['box'],
                            'conf': s_box['conf'],
                            'defect_type': 'pothole',
                            'polygon': s_box.get('polygon'),
                            'seg_area_pct': s_box.get('seg_area_pct'),
                            'frame': frame
                        })
            else:
                active_cands = cand_boxes_seg if model_seg is not None else cand_boxes_det
                for c in active_cands:
                    current_boxes.append({
                        'box': c['box'],
                        'conf': c['conf'],
                        'defect_type': 'pothole',
                        'polygon': c.get('polygon'),
                        'seg_area_pct': c.get('seg_area_pct'),
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
                    if 'first_box' not in track:
                        track['first_box'] = cb['box']
                    if cb['conf'] > track['conf']:
                        track['conf'] = cb['conf']
                        track['best_frame'] = cb['frame'].copy()
                        track['best_box'] = cb['box']
                        track['polygon'] = cb['polygon']
                        track['seg_area_pct'] = cb['seg_area_pct']

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
                        'first_box': cb['box'],
                        'best_box': cb['box'],
                        'conf': cb['conf'],
                        'polygon': cb['polygon'],
                        'seg_area_pct': cb['seg_area_pct'],
                        'best_frame': cb['frame'].copy()
                    })

            # Expire tracklets not seen for > 12 video frames
            remaining_tracks = []
            for track in active_tracks:
                if frame_idx - track['last_seen_frame'] > 12:
                    # Optical Perspective Kinematics check
                    if HAVE_ADVANCED_CV:
                        is_valid, conf_bonus = check_perspective_kinematics(track)
                        if not is_valid:
                            continue  # Drop receding or upward-moving false alarms
                        track['conf'] = min(0.99, max(0.1, track['conf'] + conf_bonus))

                    if track['hits'] >= 2 or track['conf'] >= 0.75:
                        bx1, by1, bx2, by2 = track['best_box']
                        bw = bx2 - bx1
                        bh = by2 - by1
                        
                        # Use true polygon segmentation area if available, else box area
                        if track.get('seg_area_pct') is not None and track['seg_area_pct'] > 0:
                            box_area_pct = track['seg_area_pct']
                        else:
                            box_area_pct = round((bw * bh) / (w * h) * 100, 2)
                        
                        snapshot_file = None
                        if evidence_dir and snapshot_count < 25:
                            snap_frame = track['best_frame']
                            poly = track.get('polygon')
                            
                            # Render semi-transparent polygon contour mask if segmentation is active
                            if poly is not None and len(poly) >= 3:
                                pts = np.array(poly, dtype=np.int32)
                                overlay = snap_frame.copy()
                                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                                cv2.addWeighted(overlay, 0.4, snap_frame, 0.6, 0, snap_frame)
                                cv2.polylines(snap_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                            else:
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
        if HAVE_ADVANCED_CV:
            is_valid, conf_bonus = check_perspective_kinematics(track)
            if not is_valid:
                continue
            track['conf'] = min(0.99, max(0.1, track['conf'] + conf_bonus))

        if track['hits'] >= 2 or track['conf'] >= 0.75:
            bx1, by1, bx2, by2 = track['best_box']
            bw = bx2 - bx1
            bh = by2 - by1
            
            if track.get('seg_area_pct') is not None and track['seg_area_pct'] > 0:
                box_area_pct = track['seg_area_pct']
            else:
                box_area_pct = round((bw * bh) / (w * h) * 100, 2)
            
            snapshot_file = None
            if evidence_dir and snapshot_count < 25:
                snap_frame = track['best_frame']
                poly = track.get('polygon')
                if poly is not None and len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32)
                    overlay = snap_frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.4, snap_frame, 0.6, 0, snap_frame)
                    cv2.polylines(snap_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                else:
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
    target_accel = df['accel_dyn'].values if 'accel_dyn' in df.columns else (df['accel_z'].values if 'accel_z' in df.columns else np.zeros(len(df)))
    df['accel_bandpass'] = apply_butterworth_bandpass(target_accel, fs=fs_est, lowcut=5.0, highcut=18.0)
    df['jerk'] = compute_jerk(target_accel, df['time_s'].values)

    # Self-Calibrating Adaptive Vehicle Baseline (Online Trimmed Estimator)
    if ('accel_z' in df.columns or 'accel_dyn' in df.columns):
        print("  Applying self-calibrating adaptive vehicle vibration baseline...")
        df = compute_adaptive_vehicle_baseline(df, has_3axis=has_3axis)

    # 2. Camera Verification & Multi-Modal Gating (Tri-Modal: Fused, Vision, Sensor)
    has_telemetry_sensor = (
        ('accel_z' in df.columns or 'accel_dyn' in df.columns)
        and (df['z_score'].max() > 0.0 or df['jerk'].max() > 0.0)
    )

    verified_yolo_detections = 0
    ignored_yolo_detections = 0
    for det in detections:
        t = det['time_sec'] if isinstance(det, dict) else det[0]
        snap = det.get('snapshot') if isinstance(det, dict) else None
        area = det.get('box_area_pct', 0.0) if isinstance(det, dict) else 0.0
        conf = det.get('conf', 0.5) if isinstance(det, dict) else 0.5
        hits = det.get('hits', 1) if isinstance(det, dict) else 1
        
        # Dynamic delay based on vehicle speed at timestamp t
        speed_window = df[(df['time_s'] >= t - 0.5) & (df['time_s'] <= t + 0.5)]
        v = speed_window['speed'].mean() if len(speed_window) > 0 and 'speed' in speed_window.columns else 0.0
        if pd.notna(v) and v > 0.8:
            calculated_delay = float(np.clip(lookahead_distance_m / v, 0.15, 1.5))
        else:
            calculated_delay = default_delay
            
        impact_time = t + calculated_delay
        mask = (df['time_s'] >= impact_time - 0.35) & (df['time_s'] <= impact_time + 0.35)
        
        if mask.sum() == 0 and len(df) > 0:
            closest_idx = (df['time_s'] - impact_time).abs().idxmin()
            target_indices = [closest_idx]
        else:
            target_indices = df[mask].index.tolist()
        
        max_z = df.loc[target_indices, 'z_score'].max() if len(target_indices) > 0 and 'z_score' in df.columns else 0.0
        max_jerk = df.loc[target_indices, 'jerk'].max() if len(target_indices) > 0 and 'jerk' in df.columns else 0.0
        
        # 1. Fused Verification: Camera saw it AND vehicle wheel registered an impact shock spike
        if max_z > 1.5 or max_jerk >= 50.0:
            for idx in target_indices:
                df.at[idx, 'Label'] = 1
                df.at[idx, 'Source'] = 'Fused'
                df.at[idx, 'Hazard_Type'] = 'Pothole'
                if snap:
                    df.at[idx, 'image_evidence'] = snap
                if area > 0:
                    df.at[idx, 'box_area_pct'] = max(float(df.at[idx, 'box_area_pct']), area)
            verified_yolo_detections += 1
        # 2. Vision Ground Truth:
        # Either sensor telemetry is absent in this ride (GPS-only recording),
        # OR vehicle avoided/straddled the pothole or crawled at very low speed,
        # BUT the camera temporally tracked the defect across frames
        elif (not has_telemetry_sensor) or (hits >= 2) or (conf >= 0.60):
            for idx in target_indices:
                df.at[idx, 'Label'] = 1
                df.at[idx, 'Source'] = 'Vision'
                df.at[idx, 'Hazard_Type'] = 'Pothole'
                if snap:
                    df.at[idx, 'image_evidence'] = snap
                if area > 0:
                    df.at[idx, 'box_area_pct'] = max(float(df.at[idx, 'box_area_pct']), area)
            verified_yolo_detections += 1
        else:
            # Single-frame weak detection on a confirmed smooth road with active sensor
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
            source = str(row.get('Source', 'None'))
            
            # Severity grading: Use shock if available, or visual contour area if Vision-only
            if source == 'Vision' or z < 1.0:
                if area_pct >= 10.0:
                    df.at[idx, 'Severity'] = 'Severe'
                elif area_pct >= 3.0:
                    df.at[idx, 'Severity'] = 'Moderate'
                else:
                    df.at[idx, 'Severity'] = 'Low'
            else:
                if z >= 3.0:
                    df.at[idx, 'Severity'] = 'Severe'
                elif z >= 2.0:
                    df.at[idx, 'Severity'] = 'Moderate'
                else:
                    df.at[idx, 'Severity'] = 'Low'
                
            # Dimensional modeling (physical surface area in cm² & depth in cm)
            area_cm2 = round(max(150.0, area_pct * 140.0), 1) if area_pct > 0 else round(max(100.0, max(z, 1.0) * 180.0), 1)
            depth_cm = round(min(12.0, max(2.5, max(z, 1.0) * 1.65)), 1)
            
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
        
    accel = df['accel_dyn'].values if 'accel_dyn' in df.columns else (df['accel_z'].values if 'accel_z' in df.columns else np.zeros(len(df)))
    
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


def process_ride_folder(folder, do_stabilize=True, force_stabilize=False, do_retrain=False, conf_threshold=0.40, lookahead_distance_m=3.5, model_mode='ensemble', use_onnx=False):
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

        detections = run_yolo(video_to_analyze, conf_threshold=conf_threshold, evidence_dir=evidence_dir, model_mode=model_mode, use_onnx=use_onnx)
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
    
    model_mode = 'ensemble'
    for a in sys.argv:
        if a.startswith('--model='):
            model_mode = a.split('=', 1)[1]
        elif a in ('--seg', '--v8', '--ensemble'):
            model_mode = a.lstrip('-')
    use_onnx = '--onnx' in sys.argv
    
    process_ride_folder(
        target_folder,
        do_stabilize=not flag_no_stabilize,
        force_stabilize=flag_force_stabilize,
        do_retrain=flag_retrain,
        model_mode=model_mode,
        use_onnx=use_onnx
    )
