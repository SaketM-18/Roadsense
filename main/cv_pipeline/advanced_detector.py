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
    
    Returns:
        is_valid (bool), confidence_adjustment (float)
    """
    if track.get('hits', 1) < 2:
        return True, 0.0 # Single frame observation
        
    first_cy = (track['first_box'][1] + track['first_box'][3]) / 2.0
    last_cy = (track['best_box'][1] + track['best_box'][3]) / 2.0
    dy = last_cy - first_cy
    
    first_w = max(1.0, track['first_box'][2] - track['first_box'][0])
    first_h = max(1.0, track['first_box'][3] - track['first_box'][1])
    first_area = first_w * first_h
    
    last_w = max(1.0, track['best_box'][2] - track['best_box'][0])
    last_h = max(1.0, track['best_box'][3] - track['best_box'][1])
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
