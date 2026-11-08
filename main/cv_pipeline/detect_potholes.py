"""
RoadSense — High-Precision Computer Vision Detection Engine (v2.0)
==================================================================
Performs deep computer vision pothole detection on dashcam footage.

Key Upgrades:
1. CLAHE Contrast Normalization: Equalizes harsh road lighting, direct sunlight, and tree shadows.
2. Multi-Frame Temporal Tracklet Association: Rejects single-frame optical flashes and noise.
3. Box Area Scaling: Computes normalized bounding box area (%) for defect dimension modeling.
4. Export Schema: video_time_sec, confidence, box_area_pct, track_hits.
"""

import os
import sys
import csv
import cv2
import numpy as np
from ultralytics import YOLO


def apply_clahe(frame):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the Luminance (L) channel.
    Normalizes deep shadows and asphalt glare without altering natural chrominance.
    """
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def analyze_video(
    video_path,
    model_path,
    output_csv,
    conf_threshold=0.45,
    use_clahe=True,
    min_hits=2,
    evidence_dir=None
):
    """
    Analyzes video with CLAHE enhancement and multi-frame temporal tracklet filtering.
    """
    print("=" * 60)
    print("ROADSENSE — Computer Vision Detection Pipeline (v2.0)")
    print("=" * 60)
    print(f"Loading YOLO model: {os.path.basename(model_path)}...")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return []

    print(f"Opening video: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    print(f"Video Stats: {total_frames} frames | {w}x{h} | {fps:.1f} FPS | CLAHE: {use_clahe}")

    if evidence_dir:
        os.makedirs(evidence_dir, exist_ok=True)

    active_tracks = []
    final_detections = []
    frame_idx = 0
    snapshot_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_idx / fps

        # Apply CLAHE to enhance asphalt texture under shadows
        proc_frame = apply_clahe(frame) if use_clahe else frame

        # Run inference (stream=False to get standard Results object)
        results = model.predict(proc_frame, conf=conf_threshold, verbose=False)
        
        current_detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls_id, "").lower()
                
                # Extract class and confidence
                label_name = model.names.get(cls_id, "").lower() if hasattr(model, 'names') and model.names else ""
                
                # Distinguish real-world road elements
                if "pothole" in label_name or cls_id == 0:
                    defect_type = "pothole"
                elif "crack" in label_name or cls_id == 1:
                    defect_type = "crack"
                elif "manhole" in label_name or cls_id == 2:
                    defect_type = "manhole"
                else:
                    defect_type = label_name or "hazard"

                # Filter or categorize detection (potholes prioritized, cracks/manholes recognized)
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                current_detections.append({
                    'box': xyxy,
                    'conf': conf,
                    'defect_type': defect_type,
                    'frame': frame
                })

        # Match detections to active tracklets
        matched_curr = set()
        for track in active_tracks:
            t_box = track['box']
            t_cx = (t_box[0] + t_box[2]) / 2.0
            t_cy = (t_box[1] + t_box[3]) / 2.0

            best_iou = 0.0
            best_idx = -1

            for i, cd in enumerate(current_detections):
                if i in matched_curr:
                    continue
                c_box = cd['box']
                iou = compute_iou(t_box, c_box)
                c_cx = (c_box[0] + c_box[2]) / 2.0
                c_cy = (c_box[1] + c_box[3]) / 2.0
                dist = np.hypot(t_cx - c_cx, t_cy - c_cy)

                if iou > best_iou or (best_iou == 0 and dist < 85):
                    best_iou = max(iou, 0.2 if dist < 85 else 0)
                    best_idx = i

            if best_idx >= 0 and (best_iou >= 0.15 or best_iou > 0):
                matched_curr.add(best_idx)
                cd = current_detections[best_idx]
                track['hits'] += 1
                track['last_seen_frame'] = frame_idx
                track['last_time_sec'] = current_time_sec
                track['box'] = cd['box']
                if cd['conf'] > track['best_conf']:
                    track['best_conf'] = cd['conf']
                    track['defect_type'] = cd['defect_type']
                    track['best_frame'] = cd['frame'].copy()
                    track['best_box'] = cd['box']

        # Start new tracklets for unmatched detections
        for i, cd in enumerate(current_detections):
            if i not in matched_curr:
                active_tracks.append({
                    'first_time_sec': current_time_sec,
                    'last_time_sec': current_time_sec,
                    'first_seen_frame': frame_idx,
                    'last_seen_frame': frame_idx,
                    'hits': 1,
                    'box': cd['box'],
                    'best_box': cd['box'],
                    'best_conf': cd['conf'],
                    'defect_type': cd['defect_type'],
                    'best_frame': cd['frame'].copy()
                })

        # Expire tracklets not seen for > 10 frames
        surviving_tracks = []
        for track in active_tracks:
            if frame_idx - track['last_seen_frame'] > 10:
                # Filter: must persist for min_hits or have high single-frame confidence
                if track['hits'] >= min_hits or track['best_conf'] >= 0.75:
                    bx1, by1, bx2, by2 = track['best_box']
                    box_w = bx2 - bx1
                    box_h = by2 - by1
                    area_pct = round((box_w * box_h) / (w * h) * 100.0, 2)
                    d_type = track.get('defect_type', 'pothole')

                    snapshot_path = None
                    if evidence_dir and snapshot_count < 30:
                        snap = track['best_frame']
                        # Color coding: Green for pothole, Orange for crack, Cyan for manhole
                        color = (0, 255, 0) if d_type == "pothole" else ((0, 165, 255) if d_type == "crack" else (255, 255, 0))
                        cv2.rectangle(snap, (bx1, by1), (bx2, by2), color, 2)
                        cv2.putText(snap, f"{d_type.capitalize()}: {int(track['best_conf']*100)}%", 
                                    (bx1, max(by1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        snap_name = f"evidence_{int(track['first_time_sec']):03d}s_{snapshot_count+1}.jpg"
                        cv2.imwrite(os.path.join(evidence_dir, snap_name), snap)
                        snapshot_path = snap_name
                        snapshot_count += 1

                    det_record = {
                        "video_time_sec": round(track['first_time_sec'], 3),
                        "confidence": round(track['best_conf'], 2),
                        "box_area_pct": area_pct,
                        "track_hits": track['hits'],
                        "snapshot": snapshot_path or "",
                        "defect_type": d_type
                    }
                    final_detections.append(det_record)
                    print(f"  [{d_type.upper()}] {det_record['video_time_sec']}s (Conf: {det_record['confidence']:.2f}, Hits: {det_record['track_hits']}, Area: {det_record['box_area_pct']}%)")
            else:
                surviving_tracks.append(track)
        active_tracks = surviving_tracks

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  ... {frame_idx}/{total_frames} frames processed")

    # Flush remaining active tracklets
    for track in active_tracks:
        if track['hits'] >= min_hits or track['best_conf'] >= 0.75:
            bx1, by1, bx2, by2 = track['best_box']
            box_w = bx2 - bx1
            box_h = by2 - by1
            area_pct = round((box_w * box_h) / (w * h) * 100.0, 2)
            d_type = track.get('defect_type', 'pothole')
            final_detections.append({
                "video_time_sec": round(track['first_time_sec'], 3),
                "confidence": round(track['best_conf'], 2),
                "box_area_pct": area_pct,
                "track_hits": track['hits'],
                "snapshot": "",
                "defect_type": d_type
            })

    cap.release()

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, mode='w', newline='') as f:
        fieldnames = ["video_time_sec", "confidence", "box_area_pct", "track_hits", "snapshot", "defect_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_detections)

    print(f"\n[COMPLETE] Extracted {len(final_detections)} validated hazard tracklet(s).")
    print(f"[SAVED] Results: {output_csv}")
    print("=" * 60)
    return final_detections


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(current_dir)
    
    # Check default test ride video
    test_video = None
    for folder in ['Four', 'Three', 'Five', 'two', 'one']:
        cand = os.path.join(main_dir, 'test', folder, f'{folder}.mp4')
        if os.path.exists(cand):
            test_video = cand
            break
            
    my_model = os.path.join(current_dir, "pothole_yolov8.pt")
    out_csv = os.path.join(current_dir, "yolo_timestamps.csv")
    
    if test_video and os.path.exists(my_model):
        analyze_video(test_video, my_model, out_csv)
    else:
        print("Script ready. Call analyze_video(video_path, model_path, output_csv).")
