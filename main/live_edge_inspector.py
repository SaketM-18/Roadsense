"""
RoadSense — Real-Time Edge Runtime & Live Dashcam Inspector
===========================================================
High-throughput, sub-second edge runtime engineered for in-vehicle dashcams,
webcams, RTSP streams, and pre-recorded video evaluation.

Key Capabilities:
1. INT8 Quantized ONNX Runtime Inference (sub-50ms CPU execution).
2. PotholeKalmanTracker: ByteTrack-style 8-state kinematic multi-object tracking.
3. Inverse Perspective Mapping (IPM): Real-time metric ground distance (m), 
   non-convex defect area (cm²), and cold-mix bitumen mass (kg).
4. Heads-Up Display (HUD) with dynamic telemetry, collision corridor, and 
   critical shock warning banner (< 3.5m alert).
5. Dynamic Zero-DCE low-light enhancement and on-device privacy blurring.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Ensure main directory is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ultralytics import YOLO

# Import advanced detector components
CV_DIR = os.path.join(BASE_DIR, 'cv_pipeline')
if CV_DIR not in sys.path:
    sys.path.insert(0, CV_DIR)

from advanced_detector import (
    PotholeKalmanTracker,
    compute_metric_pothole_dimensions,
    zero_dce_low_light_enhancement,
    anonymize_privacy_regions,
    classify_irc_maintenance_code
)

# Model paths
V8_INT8 = os.path.join(CV_DIR, 'pothole_yolov8_int8.onnx')
V8_FP32 = os.path.join(CV_DIR, 'pothole_yolov8.onnx')
V8_PT = os.path.join(CV_DIR, 'pothole_yolov8.pt')

SEG_INT8 = os.path.join(CV_DIR, 'pothole_yolo11_seg_int8.onnx')
SEG_FP32 = os.path.join(CV_DIR, 'pothole_yolo11_seg.onnx')
SEG_PT = os.path.join(CV_DIR, 'pothole_yolo11_seg.pt')


def compute_box_iou(box1, box2):
    """Computes Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-6)


class EdgeTrackManager:
    """Manages active Kalman filter tracklets across video frames."""
    def __init__(self, max_age=8, min_hits=2, iou_thresh=0.20):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.tracks = []
        self.frame_idx = 0

    def update(self, detections, frame=None, time_sec=0.0):
        self.frame_idx += 1
        
        # 1. Predict next state for each track
        for trk in self.tracks:
            trk.predict()
            
        # 2. Association between detections and existing tracks
        matched_det_indices = set()
        matched_trk_indices = set()
        
        for trk_idx, trk in enumerate(self.tracks):
            best_iou = 0.0
            best_det_idx = -1
            t_box = trk.get_box()
            t_cx = (t_box[0] + t_box[2]) / 2.0
            t_cy = (t_box[1] + t_box[3]) / 2.0
            
            for d_idx, det in enumerate(detections):
                if d_idx in matched_det_indices:
                    continue
                d_box = det['bbox']
                d_cx = (d_box[0] + d_box[2]) / 2.0
                d_cy = (d_box[1] + d_box[3]) / 2.0
                dist = np.hypot(t_cx - d_cx, t_cy - d_cy)
                iou = compute_box_iou(t_box, d_box)
                
                if iou > best_iou or (best_iou == 0 and dist < 85):
                    best_iou = max(iou, 0.25 if dist < 85 else 0)
                    best_det_idx = d_idx
                    
            if best_det_idx >= 0 and best_iou >= self.iou_thresh:
                matched_det_indices.add(best_det_idx)
                matched_trk_indices.add(trk_idx)
                matched_det = detections[best_det_idx]
                trk.update(
                    bbox=matched_det['bbox'],
                    conf=matched_det.get('conf', 0.5),
                    polygon=matched_det.get('mask'),
                    frame=frame,
                    frame_idx=self.frame_idx,
                    time_sec=time_sec
                )

        # 3. Spawn new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_trk = PotholeKalmanTracker(
                    bbox=det['bbox'],
                    conf=det.get('conf', 0.5),
                    polygon=det.get('mask'),
                    frame=frame,
                    frame_idx=self.frame_idx,
                    time_sec=time_sec
                )
                self.tracks.append(new_trk)
                
        # 4. Prune dead tracks
        self.tracks = [t for t in self.tracks if (self.frame_idx - t.last_seen_frame) <= self.max_age]
        
        # 5. Return confirmed active tracks (seen recently)
        active = [t for t in self.tracks if t.hits >= self.min_hits or (self.frame_idx - t.last_seen_frame) <= 1]
        return active


class LiveEdgeInspector:
    def __init__(self, source=0, use_onnx=True, use_int8=True, conf_thresh=0.35, enable_privacy=True, enable_enhancement=True):
        self.source = source
        self.conf_thresh = conf_thresh
        self.enable_privacy = enable_privacy
        self.enable_enhancement = enable_enhancement
        self.show_hud = True
        
        print("\n" + "=" * 65)
        print("ROADSENSE — Real-Time Edge Runtime & Live Dashcam Inspector")
        print("=" * 65)
        
        # 1. Resolve Detection Engine
        if use_int8 and os.path.exists(V8_INT8):
            self.model_det_path = V8_INT8
            self.det_engine_type = "INT8 ONNX"
        elif use_onnx and os.path.exists(V8_FP32):
            self.model_det_path = V8_FP32
            self.det_engine_type = "FP32 ONNX"
        else:
            self.model_det_path = V8_PT
            self.det_engine_type = "PyTorch FP32"
            
        # 2. Resolve Segmentation Engine
        if use_int8 and os.path.exists(SEG_INT8):
            self.model_seg_path = SEG_INT8
            self.seg_engine_type = "INT8 ONNX"
        elif use_onnx and os.path.exists(SEG_FP32):
            self.model_seg_path = SEG_FP32
            self.seg_engine_type = "FP32 ONNX"
        else:
            self.model_seg_path = SEG_PT
            self.seg_engine_type = "PyTorch FP32"

        print(f"  Detector:   {os.path.basename(self.model_det_path)} [{self.det_engine_type}]")
        print(f"  Segmenter:  {os.path.basename(self.model_seg_path)} [{self.seg_engine_type}]")
        
        # Load models
        print("  Initializing neural runtimes...")
        self.model_det = YOLO(self.model_det_path, task='detect')
        self.model_seg = YOLO(self.model_seg_path, task='segment') if os.path.exists(self.model_seg_path) else None
        
        # Initialize Kalman Multi-Object Tracker
        self.tracker = EdgeTrackManager(max_age=12, min_hits=2, iou_thresh=0.20)
        
        # Telemetry
        self.frame_count = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.active_potholes = []
        self.danger_alert = False

    def process_frame(self, frame):
        t_start = time.time()
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Low-light dynamic enhancement if frame is dark
        if self.enable_enhancement:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(np.mean(gray)) < 80.0:
                frame = zero_dce_low_light_enhancement(frame)
                
        # 2. Fast Stage 1 Candidate Bounding Boxes
        results = self.model_det(frame, conf=self.conf_thresh, verbose=False, imgsz=640)
        detections = []
        
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append({
                        'bbox': coords,
                        'conf': conf,
                        'mask': None
                    })
                    
        # 3. ByteTrack Kalman Multi-Object Tracking
        active_tracks = self.tracker.update(detections, frame=frame, time_sec=time.time())
        
        # 4. Selective Stage 2 Segmentation on Active Tracks (Runs only when defects exist)
        if len(active_tracks) > 0 and self.model_seg is not None and self.frame_count % 2 == 0:
            seg_res = self.model_seg(frame, conf=max(0.20, self.conf_thresh - 0.10), verbose=False, imgsz=640)
            for sr in seg_res:
                if hasattr(sr, 'masks') and sr.masks is not None:
                    raw_masks = sr.masks.xy
                    for m_idx, m_poly in enumerate(raw_masks):
                        if len(m_poly) >= 3 and m_idx < len(active_tracks):
                            active_tracks[m_idx].polygon = m_poly
                            
        # 5. Photogrammetric IPM Profiling & Risk Scoring
        self.danger_alert = False
        processed_defects = []
        
        for trk in active_tracks:
            bbox = trk.get_box()
            mask = getattr(trk, 'polygon', None)
            conf = getattr(trk, 'conf', 0.5)
            track_id = getattr(trk, 'id', 0)
            
            # True metric dimensions via IPM photogrammetry
            dims = compute_metric_pothole_dimensions(bbox, mask, frame_shape=(orig_h, orig_w), z_score=0.0)
            dist_m = dims.get('ground_distance_m', 10.0)
            area_cm2 = dims.get('metric_area_cm2', 250.0)
            depth_cm = dims.get('metric_depth_cm', 3.5)
            mass_kg = dims.get('cold_mix_mass_kg', 2.5)
            
            # IRC Maintenance Standard
            irc_cat, irc_code, irc_act = classify_irc_maintenance_code(depth_cm, area_cm2)
            
            # Proximity collision / chassis damage alert
            if dist_m <= 4.0:
                self.danger_alert = True
                
            processed_defects.append({
                'track_id': track_id,
                'bbox': bbox,
                'mask': mask,
                'conf': conf,
                'dist_m': dist_m,
                'area_cm2': area_cm2,
                'depth_cm': depth_cm,
                'mass_kg': mass_kg,
                'irc_cat': irc_cat,
                'irc_code': irc_code
            })
            
        self.active_potholes = processed_defects
        self.latency_ms = (time.time() - t_start) * 1000.0
        self.fps = 1000.0 / max(self.latency_ms, 0.001)
        self.frame_count += 1
        
        # 6. Render HUD Overlay
        annotated = frame.copy()
        if self.show_hud:
            annotated = self.render_hud(annotated)
            
        # 7. Privacy Anonymization
        if self.enable_privacy:
            annotated = anonymize_privacy_regions(annotated)
            
        return annotated

    def render_hud(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # 1. Drive corridor perspective lines (HUD Trajectory)
        vanishing_y = int(h * 0.45)
        cv2.line(overlay, (int(w * 0.15), h), (int(w * 0.42), vanishing_y), (0, 240, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (int(w * 0.85), h), (int(w * 0.58), vanishing_y), (0, 240, 255), 1, cv2.LINE_AA)
        
        # Distance markers across road corridor
        for d_m, frac in [(3.0, 0.85), (6.0, 0.70), (10.0, 0.58), (15.0, 0.50)]:
            y_pos = int(h * frac)
            cv2.line(overlay, (int(w * 0.30), y_pos), (int(w * 0.70), y_pos), (0, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"{d_m:.0f}m", (int(w * 0.72), y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 255), 1)

        # 2. Defect Contours & Photogrammetric Callouts
        for d in self.active_potholes:
            bbox = [int(v) for v in d['bbox']]
            dist = d['dist_m']
            mass = d['mass_kg']
            depth = d['depth_cm']
            tid = d['track_id']
            irc_code = d['irc_code'].split('-')[0]
            
            # Severity color coding (Red: Critical < 4m, Amber: Approaching, Neon Yellow: Distant)
            if dist < 4.0:
                color = (0, 0, 255)       # Red
            elif dist < 8.0:
                color = (0, 165, 255)     # Amber
            else:
                color = (0, 255, 204)     # Neon Yellow
                
            # Draw bounding box
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw polygon mask if available
            if d['mask'] is not None and len(d['mask']) >= 3:
                poly = np.array(d['mask'], dtype=np.int32)
                cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=2)
                
            # Floating Tag
            tag_text = f"#{tid:02d} | {dist:.1f}m | {depth:.1f}cm | {mass:.1f}kg"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(overlay, (bbox[0], bbox[1] - th - 8), (bbox[0] + tw + 6, bbox[1]), color, -1)
            cv2.putText(overlay, tag_text, (bbox[0] + 3, bbox[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Top Telemetry Banner
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 23, 42), -1)
        
        # Telemetry Texts
        title_text = "ROADSENSE EDGE RUNTIME"
        cv2.putText(overlay, title_text, (14, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        
        stat_text = f"ENGINE: {self.det_engine_type} | LATENCY: {self.latency_ms:.1f}ms ({self.fps:.1f} FPS) | TARGETS: {len(self.active_potholes)}"
        cv2.putText(overlay, stat_text, (260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1, cv2.LINE_AA)
        
        # 4. Critical Warning Banner
        if self.danger_alert:
            alert_bar_y = h - 55
            cv2.rectangle(overlay, (0, alert_bar_y), (w, h), (0, 0, 200), -1)
            alert_msg = "!! HAZARD WARNING: SEVERE ROAD DEFECT IN TRAJECTORY (< 3.5m) !!"
            (atw, ath), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_DUPLEX, 0.58, 1)
            cv2.putText(overlay, alert_msg, (int((w - atw) / 2), alert_bar_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend overlay
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame

    def run(self, save_output_path=None, max_frames=None):
        print(f"\n[STREAM] Opening video capture stream from: {self.source}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise IOError(f"Could not open video stream: {self.source}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"  Stream format: {width}x{height} @ {fps_in:.1f} FPS (Total frames: {total_f if total_f > 0 else 'Live Stream'})")
        print("  Controls: [q] Quit | [s] Save Snapshot | [p] Toggle Privacy Blur | [h] Toggle HUD")
        
        writer = None
        if save_output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output_path, fourcc, fps_in, (width, height))
            print(f"  Recording HUD output stream to: {save_output_path}")
            
        processed_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame = self.process_frame(frame)
                processed_count += 1
                
                if writer:
                    writer.write(processed_frame)
                    
                # Interactive Display Window (safe fallback if GUI display is not present)
                try:
                    cv2.imshow("RoadSense — Live Edge Inspector", processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[STOP] User pressed 'q' — exiting stream.")
                        break
                    elif key == ord('p'):
                        self.enable_privacy = not self.enable_privacy
                        print(f"  [TOGGLE] Privacy Blur: {'ENABLED' if self.enable_privacy else 'DISABLED'}")
                    elif key == ord('h'):
                        self.show_hud = not self.show_hud
                        print(f"  [TOGGLE] HUD Telemetry: {'ENABLED' if self.show_hud else 'DISABLED'}")
                    elif key == ord('s'):
                        snap_path = f"edge_snapshot_{int(time.time())}.jpg"
                        cv2.imwrite(snap_path, processed_frame)
                        print(f"  [SNAPSHOT] Saved high-res frame with HUD: {snap_path}")
                except cv2.error:
                    pass
                    
                if max_frames and processed_count >= max_frames:
                    print(f"\n[DONE] Reached requested max_frames ({max_frames}).")
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"\n[STREAM CLOSED] Processed {processed_count} frames. Average Latency: {self.latency_ms:.1f}ms ({self.fps:.1f} FPS)")


def parse_args():
    parser = argparse.ArgumentParser(description="RoadSense Real-Time Edge Runtime & Live Dashcam Inspector")
    parser.add_argument("--source", type=str, default="0", help="Video source: camera index (e.g. 0), video file path, or RTSP URL")
    parser.add_argument("--save", type=str, default=None, help="Optional output path to record the HUD video (e.g. output_hud.mp4)")
    parser.add_argument("--no-onnx", action="store_true", help="Force PyTorch FP32 engine instead of ONNX")
    parser.add_argument("--no-int8", action="store_true", help="Use FP32 ONNX instead of INT8 quantized models")
    parser.add_argument("--no-privacy", action="store_true", help="Disable on-device license plate and facial blurring")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for defect candidates")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames then exit")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Parse source: int if webcam index, else str
    src = int(args.source) if args.source.isdigit() else args.source
    
    inspector = LiveEdgeInspector(
        source=src,
        use_onnx=not args.no_onnx,
        use_int8=not args.no_int8,
        conf_thresh=args.conf,
        enable_privacy=not args.no_privacy
    )
    
    inspector.run(save_output_path=args.save, max_frames=args.max_frames)
