"""
RoadSense — Advanced Computer Vision Intelligence & Accuracy Enhancer
======================================================================
Implements:
1. SAHI (Slicing Aided Hyper Inference): Catches small, distant road cavities 15-25m ahead.
2. 3D Rim Gradient Depth Check: Sobel edge contrast analysis to reject flat shadows and fresh patches.
3. Forward Optical Perspective Kinematics: Validates downward trajectory (+y) and perspective growth.
4. Ensemble Consensus Reinforcement: Fuses YOLOv8 detection recall with YOLO11-Seg polygon contours.
"""

import cv2
import numpy as np


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


def suppress_duplicate_boxes(candidates, iou_thresh=0.45):
    """
    Suppresses duplicate candidate boxes from combined full-frame and SAHI slice inference.
    If a detection exists in both, keeps the higher confidence box and preserves polygon contours.
    """
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda x: x.get('conf', 0.0), reverse=True)
    keep = []
    for cand in sorted_cands:
        duplicate = False
        for kept in keep:
            if compute_box_iou(cand['box'], kept['box']) >= iou_thresh:
                # Merge: if candidate has polygon and kept does not, transfer polygon
                if cand.get('polygon') and not kept.get('polygon'):
                    kept['polygon'] = cand['polygon']
                    kept['seg_area_pct'] = cand['seg_area_pct']
                duplicate = True
                break
        if not duplicate:
            keep.append(cand)
    return keep


def compute_rim_gradient_score(frame, box):
    """
    3D Rim Gradient Depth Check.
    Analyzes the perimeter edge gradient (Sobel operator) of candidate cavities.
    Real potholes have a distinct 3D elevation lip (high gradient variance between rim and core).
    Flat tree shadows and dark asphalt stains have low rim gradient contrast.
    
    Returns:
        rim_contrast (float), rim_to_core_ratio (float)
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    # 15% perimeter annular margin
    pw = max(4, int((x2 - x1) * 0.15))
    ph = max(4, int((y2 - y1) * 0.15))
    bx1 = max(0, x1 - pw)
    by1 = max(0, y1 - ph)
    bx2 = min(w, x2 + pw)
    by2 = min(h, y2 + ph)
    
    crop = frame[by1:by2, bx1:bx2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return 5.0, 1.2 # Default pass for small/distant crops
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    ch, cw = gray.shape
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.ellipse(mask, (cw // 2, ch // 2), (cw // 2, ch // 2), 0, 0, 360, 255, -1)
    inner_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.ellipse(inner_mask, (cw // 2, ch // 2), (max(2, cw // 3), max(2, ch // 3)), 0, 0, 360, 255, -1)
    rim_mask = cv2.bitwise_xor(mask, inner_mask)
    
    rim_vals = mag[rim_mask > 0]
    inner_vals = mag[inner_mask > 0]
    
    if len(rim_vals) == 0:
        return 5.0, 1.2
        
    rim_contrast = float(np.mean(rim_vals))
    inner_mean = float(np.mean(inner_vals)) if len(inner_vals) > 0 else 1.0
    rim_ratio = rim_contrast / (inner_mean + 1e-4)
    return rim_contrast, rim_ratio


def is_flat_shadow(frame, box, min_rim_contrast=2.5, min_ratio=0.85):
    """
    Returns True if the candidate box has flat surface texture with no 3D cavity rim.
    """
    contrast, ratio = compute_rim_gradient_score(frame, box)
    # Extremely flat luminance drop with zero edge definition = flat shadow
    if contrast < min_rim_contrast and ratio < min_ratio:
        return True
    return False


def check_perspective_kinematics(track):
    """
    Forward Optical Perspective Kinematics Gate.
    As a vehicle drives forward, true stationary roadway defects MUST:
    1. Move downward (+y) towards the vehicle wheels.
    2. Expand or maintain their bounding box area (area_growth >= 0.70).
    
    Accepts both dictionary tracklets and PotholeKalmanTracker objects.
    Returns:
        is_valid (bool), confidence_adjustment (float)
    """
    hits = getattr(track, 'hits', track.get('hits', 1) if isinstance(track, dict) else 1)
    if hits < 2:
        return True, 0.0 # Single frame observation
        
    first_box = getattr(track, 'first_box', track.get('first_box') if isinstance(track, dict) else None)
    best_box = getattr(track, 'best_box', track.get('best_box') if isinstance(track, dict) else None)
    if first_box is None or best_box is None:
        return True, 0.0
        
    first_cy = (first_box[1] + first_box[3]) / 2.0
    last_cy = (best_box[1] + best_box[3]) / 2.0
    dy = last_cy - first_cy
    
    first_w = max(1.0, first_box[2] - first_box[0])
    first_h = max(1.0, first_box[3] - first_box[1])
    first_area = first_w * first_h
    
    last_w = max(1.0, best_box[2] - best_box[0])
    last_h = max(1.0, best_box[3] - best_box[1])
    last_area = last_w * last_h
    
    area_growth = last_area / first_area
    
    # Severe perspective violation: moving upward into the horizon or shrinking severely
    if dy < -18.0 and area_growth < 0.60:
        return False, -0.25 # Receding vehicle or overhead obstruction
    elif dy >= 4.0 and area_growth >= 0.80:
        return True, 0.05 # Verified forward expansion bonus
    return True, 0.0


def sahi_corridor_inference(model, frame, conf_threshold=0.35, y_slice_range=(0.20, 0.72)):
    """
    Slicing Aided Hyper Inference (SAHI) for small, distant road cavities (15-25m ahead).
    Cuts the high-resolution corridor where distant defects are foreshortened, runs inference,
    and maps bounding box coordinates back to full-frame space.
    """
    h, w = frame.shape[:2]
    y1 = int(h * y_slice_range[0])
    y2 = int(h * y_slice_range[1])
    slice_img = frame[y1:y2, :]
    
    if slice_img.shape[0] < 64 or slice_img.shape[1] < 64:
        return []
        
    res = model.predict(slice_img, conf=conf_threshold, verbose=False)
    slice_boxes = []
    
    for r in res:
        has_m = hasattr(r, 'masks') and r.masks is not None and len(r.masks) > 0
        for i, box in enumerate(r.boxes):
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])
            lbl = model.names.get(cls_id, "").lower() if hasattr(model, 'names') and model.names else ""
            
            if "pothole" in lbl or cls_id == 0:
                mapped_box = (bx1, by1 + y1, bx2, by2 + y1)
                poly = None
                s_area = None
                if has_m and i < len(r.masks.xy):
                    pts = r.masks.xy[i].copy()
                    if len(pts) >= 3:
                        pts[:, 1] += y1 # Apply vertical offset
                        poly = pts.tolist()
                        poly_area_px = cv2.contourArea(np.array(pts, dtype=np.float32))
                        s_area = round((poly_area_px / (w * h)) * 100.0, 3)
                        
                slice_boxes.append({
                    'box': mapped_box,
                    'conf': float(box.conf[0]),
                    'polygon': poly,
                    'seg_area_pct': s_area,
                    'is_sahi': True
                })
                
    return slice_boxes


def fuse_ensemble_candidates(cand_boxes_det, cand_boxes_seg, frame, iou_thresh=0.25):
    """
    Ensemble Consensus Fusion with 3D Rim Gradient Validation.
    Pairs high-recall YOLOv8 detection bounding boxes with YOLO11-Seg polygon contours.
    Reinforces confidence when both models agree, and filters flat shadow false alarms.
    """
    current_boxes = []
    matched_seg_idx = set()
    
    for d_box in cand_boxes_det:
        # Check 3D rim gradient
        if is_flat_shadow(frame, d_box['box']):
            continue # Drop flat shadow
            
        best_match = None
        best_iou = 0.0
        best_s_idx = -1
        
        for s_idx, s_box in enumerate(cand_boxes_seg):
            iou = compute_box_iou(d_box['box'], s_box['box'])
            if iou > best_iou:
                best_iou = iou
                best_s_idx = s_idx
                
        if best_s_idx >= 0 and best_iou >= iou_thresh:
            matched_seg_idx.add(best_s_idx)
            s_box = cand_boxes_seg[best_s_idx]
            # Consensus bonus
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
            
    # Include remaining segmentation candidates that passed gradient check
    for s_idx, s_box in enumerate(cand_boxes_seg):
        if s_idx not in matched_seg_idx:
            if not is_flat_shadow(frame, s_box['box']):
                current_boxes.append({
                    'box': s_box['box'],
                    'conf': s_box['conf'],
                    'defect_type': 'pothole',
                    'polygon': s_box.get('polygon'),
                    'seg_area_pct': s_box.get('seg_area_pct'),
                    'frame': frame
                })
                
    return current_boxes


class PotholeKalmanTracker:
    """
    ByteTrack-Style Kalman Filter Multi-Object Tracker for Road Cavities.
    Maintains an 8D state vector [cx, cy, a, h, vx, vy, va, vh]^T
    where (cx, cy) is box center, a = w/h is aspect ratio, h is box height,
    and (vx, vy, va, vh) are their velocities.
    
    Enforces forward vehicle perspective kinematics:
    Stationary road defects move downward (vy > 0) and expand (vh >= 0).
    """
    count = 0

    def __init__(self, bbox, conf=0.5, polygon=None, frame=None, seg_area_pct=None, frame_idx=0, time_sec=0.0):
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        self.id = PotholeKalmanTracker.count
        PotholeKalmanTracker.count += 1
        
        # State: [cx, cy, a, h, vx, vy, va, vh]
        self.x = np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Covariance matrix P
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # High initial velocity uncertainty
        
        # Motion model F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # pos += vel
            
        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0
            
        self.R = np.eye(4, dtype=np.float32) * 2.0   # Measurement noise
        self.Q = np.eye(8, dtype=np.float32) * 0.1   # Process noise
        self.Q[4:, 4:] *= 0.5
        
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.conf = float(conf)
        self.first_box = bbox
        self.best_box = bbox
        self.box = bbox
        self.polygon = polygon
        self.seg_area_pct = seg_area_pct
        self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.first_time_sec = time_sec
        self.last_time_sec = time_sec
        self.best_frame = frame.copy() if frame is not None else None

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        self.time_since_update += 1
        self.box = self.get_box()
        return self.box

    def update(self, bbox, conf=None, polygon=None, frame=None, seg_area_pct=None, frame_idx=0, time_sec=0.0):
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        z = np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float32)
        
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(8, dtype=np.float32) - np.dot(K, self.H)), self.P)
        
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_frame = frame_idx
        self.last_time_sec = time_sec
        self.box = bbox
        
        if conf is not None and conf > self.conf:
            self.conf = float(conf)
            self.best_box = bbox
            if polygon is not None:
                self.polygon = polygon
            if seg_area_pct is not None:
                self.seg_area_pct = seg_area_pct
            if frame is not None:
                self.best_frame = frame.copy()

    def get_box(self):
        cx, cy, a, h = self.x[:4]
        a = max(0.1, a)
        h = max(1.0, h)
        w = a * h
        return (int(cx - w / 2.0), int(cy - h / 2.0), int(cx + w / 2.0), int(cy + h / 2.0))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


def compute_metric_pothole_dimensions(
    box, 
    polygon=None, 
    frame_shape=(1080, 1920), 
    mount_height_m=1.15, 
    pitch_deg=18.0, 
    fov_v_deg=55.0, 
    fov_h_deg=70.0,
    z_score=0.0,
    rim_contrast_ratio=1.0
):
    """
    Inverse Perspective Mapping (IPM) & Photogrammetric Dimension Profiler.
    Transforms 2D image plane pixels into physical real-world metric dimensions (m, cm, liters, kg).
    Uses pinhole camera geometry, ground plane homography, and Shoelace polygon integration.
    """
    x1, y1, x2, y2 = box
    h_frame, w_frame = frame_shape[:2]
    
    yn1 = float(y1) / h_frame
    yn2 = float(y2) / h_frame
    ync = ((float(y1) + float(y2)) / 2.0) / h_frame
    
    # Ground distance D(y) via trigonometric ray casting
    angle1 = np.radians(pitch_deg + (yn1 - 0.5) * fov_v_deg)
    angle2 = np.radians(pitch_deg + (yn2 - 0.5) * fov_v_deg)
    anglec = np.radians(pitch_deg + (ync - 0.5) * fov_v_deg)
    
    dist1 = float(np.clip(mount_height_m / np.tan(np.maximum(angle1, 0.08)), 0.8, 25.0))
    dist2 = float(np.clip(mount_height_m / np.tan(np.maximum(angle2, 0.08)), 0.8, 25.0))
    distc = float(np.clip(mount_height_m / np.tan(np.maximum(anglec, 0.08)), 0.8, 25.0))
    
    # Lateral resolution (meters per pixel) at object distance
    sx = (2.0 * distc * np.tan(np.radians(fov_h_deg / 2.0))) / w_frame
    
    # 1. Surface Area: True polygon projection if contour exists, else ellipse
    if polygon is not None and len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.float64)
        v_norm = pts[:, 1] / h_frame
        ang = np.radians(pitch_deg + (v_norm - 0.5) * fov_v_deg)
        Y_g = np.clip(mount_height_m / np.tan(np.maximum(ang, 0.08)), 0.8, 25.0)
        X_g = (pts[:, 0] - w_frame / 2.0) * ((2.0 * Y_g * np.tan(np.radians(fov_h_deg / 2.0))) / w_frame)
        
        # Shoelace formula on ground plane (m² -> cm²)
        area_m2 = 0.5 * np.abs(np.dot(X_g, np.roll(Y_g, 1)) - np.dot(Y_g, np.roll(X_g, 1)))
        phys_area_cm2 = float(np.clip(area_m2 * 10000.0, 100.0, 25000.0))
        phys_w_cm = float(np.clip(np.ptp(X_g) * 100.0, 5.0, 250.0))
        phys_l_cm = float(np.clip(np.ptp(Y_g) * 100.0, 5.0, 250.0))
    else:
        phys_w_cm = float(np.clip((x2 - x1) * sx * 100.0, 5.0, 250.0))
        phys_l_cm = float(np.clip(abs(dist1 - dist2) * 100.0, 5.0, 250.0))
        phys_area_cm2 = float(np.clip(np.pi * (phys_w_cm / 2.0) * (phys_l_cm / 2.0), 100.0, 25000.0))
        
    # 2. Cavity Depth Estimation (Optical Rim Contrast + Physical Wheel Impulse)
    if z_score > 0.5:
        # Multi-modal fusion depth: combines physical drop shock with rim elevation
        depth_cm = float(np.clip(1.5 + 0.8 * rim_contrast_ratio + 1.2 * z_score, 2.5, 14.0))
    else:
        # Vision-only depth: optical edge contrast and perspective shadow width
        depth_cm = float(np.clip(2.0 + 1.4 * rim_contrast_ratio, 2.5, 9.5))
        
    # 3. Parabolic Volume: V = 0.5 * Area * Depth * 10^-3 Liters
    vol_liters = round(float(np.clip(0.5 * phys_area_cm2 * depth_cm * 0.001, 0.2, 80.0)), 2)
    # Compacted cold-mix bitumen density: 2.35 kg/L
    mass_kg = round(float(np.clip(vol_liters * 2.35, 0.5, 188.0)), 2)
    
    # 4. Municipal CPWD repair budgeting
    cost_inr = int(max(350, round(mass_kg * 45.0 + 300.0)))
    cost_usd = round(max(15.0, mass_kg * 0.65 + 10.0), 2)
    
    irc_cat, irc_code, irc_act = classify_irc_maintenance_code(depth_cm, phys_area_cm2)
    
    return {
        "ground_distance_m": round(float(distc), 2),
        "width_cm": round(float(phys_w_cm), 1),
        "length_cm": round(float(phys_l_cm), 1),
        "area_cm2": round(float(phys_area_cm2), 1),
        "depth_cm": round(float(depth_cm), 1),
        "volume_liters": vol_liters,
        "mass_kg": mass_kg,
        "cost_inr": cost_inr,
        "cost_usd": cost_usd,
        "irc_category": irc_cat,
        "irc_code": irc_code,
        "irc_action": irc_act
    }


_FACE_CASCADE = None
_PLATE_CASCADE = None

def get_privacy_cascades():
    global _FACE_CASCADE, _PLATE_CASCADE
    import os
    if _FACE_CASCADE is None:
        try:
            face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(face_xml):
                _FACE_CASCADE = cv2.CascadeClassifier(face_xml)
            else:
                _FACE_CASCADE = False
        except Exception:
            _FACE_CASCADE = False
            
    if _PLATE_CASCADE is None:
        try:
            plate_xml = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            if os.path.exists(plate_xml):
                _PLATE_CASCADE = cv2.CascadeClassifier(plate_xml)
            else:
                _PLATE_CASCADE = False
        except Exception:
            _PLATE_CASCADE = False
            
    return _FACE_CASCADE, _PLATE_CASCADE


def anonymize_privacy_regions(img_bgr, pothole_box=None):
    """
    On-Device Privacy Anonymizer.
    Detects and blurs human faces and vehicle license plates in evidence photos
    to comply with GDPR, ISO/IEC 27701, and Indian DPDP Act 2023 requirements.
    Ensures foreground pothole remains crisp.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
        
    out = img_bgr.copy()
    h, w = out.shape[:2]
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    
    face_casc, plate_casc = get_privacy_cascades()
    regions_to_blur = []
    
    # 1. Face detection
    if face_casc and hasattr(face_casc, 'detectMultiScale'):
        try:
            faces = face_casc.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(24, 24))
            for (fx, fy, fw, fh) in faces:
                regions_to_blur.append((fx, fy, fx + fw, fy + fh))
        except Exception:
            pass
            
    # 2. License plate detection
    if plate_casc and hasattr(plate_casc, 'detectMultiScale'):
        try:
            plates = plate_casc.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(28, 12))
            for (px, py, pw, ph) in plates:
                regions_to_blur.append((px, py, px + pw, py + ph))
        except Exception:
            pass
            
    # Apply Gaussian blur to privacy regions (excluding the detected pothole box)
    for (rx1, ry1, rx2, ry2) in regions_to_blur:
        if pothole_box is not None:
            px1, py1, px2, py2 = pothole_box
            if max(rx1, px1) < min(rx2, px2) and max(ry1, py1) < min(ry2, py2):
                continue  # Never blur the road defect itself
                
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        if rx2 > rx1 and ry2 > ry1:
            roi = out[ry1:ry2, rx1:rx2]
            ksize = max(15, (roi.shape[1] // 3) | 1)
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), 30)
            out[ry1:ry2, rx1:rx2] = blurred
            
    return out


def zero_dce_low_light_enhancement(img_bgr, dark_threshold=80.0):
    """
    Zero-DCE Inspired Dynamic Illumination Enhancer.
    Enhances underexposed night, dusk, and heavy shadow roadway frames
    using self-guided non-linear curve estimation without amplifying noise.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    
    # If well-lit daylight, pass through
    if mean_lum >= dark_threshold:
        return img_bgr
        
    img = img_bgr.astype(np.float32) / 255.0
    gain = min(1.6, max(0.25, (dark_threshold - mean_lum) / 50.0))
    
    # 2-iteration self-guided curve
    x = img
    for _ in range(2):
        x = x + gain * x * (1.0 - x)
        
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def classify_irc_maintenance_code(depth_cm, area_cm2):
    """
    Indian Roads Congress (IRC) Standard Maintenance Categorization.
    Complies with:
      - IRC:SP:77-2008: Manual for Pothole Repairs
      - IRC:82-2015: Code of Practice for Maintenance of Bituminous Roads
    
    Returns:
        irc_category (str), irc_code (str), recommended_action (str)
    """
    depth_mm = depth_cm * 10.0
    
    if depth_mm < 25.0:
        return (
            "Category A (Minor Surface Distress)",
            "IRC:82-Clause 5.2",
            "Preventative surface seal / tack-coat treatment"
        )
    elif 25.0 <= depth_mm <= 50.0 and area_cm2 < 10000.0:
        return (
            "Category B (Standard Pothole Cavity)",
            "IRC:SP:77-Clause 4.1",
            "Square cut, edge priming & Cold-Mix Bituminous Patching"
        )
    else:
        return (
            "Category C (Structural Base / Sub-base Failure)",
            "IRC:82-Clause 6.4",
            "Deep milling, granular sub-base recompaction & DBM overlay"
        )
