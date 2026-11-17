import os
import sys
import glob
import shutil
import random
import subprocess
import json

print("=" * 70)
print("ROADSENSE — Real-World YOLO11 Mega-Dataset Instance Segmentation")
print("Data Sources: Farzad + Raunak + Arcioni + Muskan + Roger Singh")
print("Classes: [0: Pothole, 1: Crack, 2: Manhole]")
print("=" * 70)

# 1. Install ultralytics
try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except ImportError:
    print("Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics

from ultralytics import YOLO
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available, running on CPU")

random.seed(42)

# 2. Setup Working Dataset Structure
work_dir = "/kaggle/working/mega_seg_dataset"
train_img_dir = os.path.join(work_dir, "train", "images")
train_lbl_dir = os.path.join(work_dir, "train", "labels")
val_img_dir = os.path.join(work_dir, "val", "images")
val_lbl_dir = os.path.join(work_dir, "val", "labels")

for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

print(f"\nCreated unified segmentation dataset target at: {work_dir}")

def safe_link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def normalize_seg_label(line):
    """
    Standardizes YOLO annotations for YOLO11 Instance Segmentation:
    1. Converts 5-token bounding boxes [cls, cx, cy, w, h] to 4-point polygon rectangles (8 coords).
    2. Keeps existing polygon annotations (coords >= 6, even count).
    3. Safely maps class IDs into [0, 1, 2] (0: pothole, 1: crack, 2: manhole).
    4. Clamps all coordinates to [0.0, 1.0].
    """
    tokens = line.strip().split()
    if not tokens:
        return None
    try:
        raw_cls = int(float(tokens[0]))
        # Map class safely: 0 -> pothole, 1 -> crack, 2+ -> manhole / defect
        cls_id = 0 if raw_cls == 0 else (1 if raw_cls in [1, 2] else 2)
        coords = [float(x) for x in tokens[1:]]

        # Bounding box case (cx, cy, w, h) -> convert to 4-point polygon
        if len(coords) == 4:
            cx, cy, w, h = coords
            x1 = max(0.0, min(1.0, cx - w / 2))
            y1 = max(0.0, min(1.0, cy - h / 2))
            x2 = max(0.0, min(1.0, cx + w / 2))
            y2 = max(0.0, min(1.0, cy - h / 2))
            x3 = max(0.0, min(1.0, cx + w / 2))
            y3 = max(0.0, min(1.0, cy + h / 2))
            x4 = max(0.0, min(1.0, cx - w / 2))
            y4 = max(0.0, min(1.0, cy + h / 2))
            return f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
        
        # Polygon segmentation case (at least 3 points = 6 coordinates)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            clamped = [max(0.0, min(1.0, c)) for c in coords]
            coord_str = " ".join(f"{c:.6f}" for c in clamped)
            return f"{cls_id} {coord_str}"
    except Exception:
        return None
    return None

def process_and_write_label(src_lbl_path, dst_lbl_path):
    if not os.path.exists(src_lbl_path):
        return False
    try:
        with open(src_lbl_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        normalized_lines = []
        for line in lines:
            norm = normalize_seg_label(line)
            if norm:
                normalized_lines.append(norm)
        if not normalized_lines:
            return False
        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(normalized_lines) + "\n")
        return True
    except Exception:
        return False

# 3. Discover and Ingest All Mounted Kaggle Datasets
input_root = "/kaggle/input"
print("\nScanning all mounted Kaggle datasets in /kaggle/input:")
if os.path.exists(input_root):
    for entry in os.listdir(input_root):
        print(f" • Found: {entry}")

# Find all image-label directory pairs
pair_counts = {}
for root, dirs, files in os.walk(input_root):
    if "images" in dirs:
        img_dir = os.path.join(root, "images")
        lbl_dir = os.path.join(root, "labels")
        if not os.path.exists(lbl_dir):
            parent = os.path.dirname(root)
            alt_lbl = os.path.join(parent, "labels")
            if os.path.exists(alt_lbl):
                lbl_dir = alt_lbl
        
        if os.path.exists(lbl_dir):
            norm_path = root.replace("\\", "/").lower()
            dataset_name = norm_path.split("/kaggle/input/")[-1].split("/")[0] if "/kaggle/input/" in norm_path else "dataset"
            
            # Determine split
            is_val = any(k in norm_path for k in ["/val", "/valid", "/test", "_val", "_valid"])
            
            valid_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # If not explicitly split, split 85/15
            if not is_val and not any(k in norm_path for k in ["/train", "_train"]):
                random.shuffle(valid_images)
                split_idx = int(0.85 * len(valid_images))
                splits = [(valid_images[:split_idx], False), (valid_images[split_idx:], True)]
            else:
                splits = [(valid_images, is_val)]
                
            for img_list, target_is_val in splits:
                target_i = val_img_dir if target_is_val else train_img_dir
                target_l = val_lbl_dir if target_is_val else train_lbl_dir
                
                for img_name in img_list:
                    base = os.path.splitext(img_name)[0]
                    src_img = os.path.join(img_dir, img_name)
                    src_lbl = os.path.join(lbl_dir, f"{base}.txt")
                    
                    dst_img_name = f"{dataset_name[:6]}_{img_name}"
                    dst_lbl_name = f"{dataset_name[:6]}_{base}.txt"
                    dst_img = os.path.join(target_i, dst_img_name)
                    dst_lbl = os.path.join(target_l, dst_lbl_name)
                    
                    if process_and_write_label(src_lbl, dst_lbl):
                        safe_link_or_copy(src_img, dst_img)
                        pair_counts[dataset_name] = pair_counts.get(dataset_name, 0) + 1

print("\nIngestion Summary by Dataset:")
for name, cnt in pair_counts.items():
    print(f" • {name}: {cnt} validated polygon-segmented scenes")

total_train = len(os.listdir(train_img_dir))
total_val = len(os.listdir(val_img_dir))
print(f"\nFinal Scaled Dataset Overview:")
print(f" • Training Set:   {total_train} images")
print(f" • Validation Set: {total_val} images")
print(f" • Combined Total: {total_train + total_val} segmented road scenes")

if total_train == 0 or total_val == 0:
    print("ERROR: Ingestion failed. Dumping input structure:")
    for r, d, f in os.walk(input_root):
        print(" ", r, d[:3], f[:3])
    sys.exit(1)

# 4. Generate data_seg.yaml
yaml_path = "/kaggle/working/data_seg.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {train_img_dir}\n")
    f.write(f"val: {val_img_dir}\n")
    f.write("nc: 3\n")
    f.write("names: ['pothole', 'crack', 'manhole']\n")

print(f"\nGenerated unified YAML at {yaml_path}:")
with open(yaml_path, "r") as f:
    print(f.read())

# 5. Fine-Tune YOLO11m-Seg with Copy-Paste & Cosine Annealing
print("\n" + "=" * 70)
print("Starting YOLO11m-Seg Mega-Dataset Fine-Tuning (55 Epochs)...")
print("Advanced Augmentations: Copy-Paste (0.3), Mosaic (1.0), MixUp (0.2), Cosine LR")
print("=" * 70)

model = YOLO("yolo11m-seg.pt")

results = model.train(
    data=yaml_path,
    epochs=55,
    imgsz=640,
    batch=16,
    patience=14,
    workers=4,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,       # Cosine learning rate schedule for smooth global convergence
    copy_paste=0.3,    # Real-world cut-and-paste augmentation for polygon masks
    mosaic=1.0,        # Multi-defect contextual exposure
    mixup=0.2,         # Asphalt texture blending
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    project="/kaggle/working/RoadSense_YOLO11_Seg",
    name="mega_model",
    verbose=True
)

# 6. Evaluation & Metric Summary
print("\n" + "=" * 70)
print("VALIDATING MEGA-DATASET MODEL PERFORMANCE...")
print("=" * 70)

val_metrics = model.val()

box_map50 = float(val_metrics.box.map50) if hasattr(val_metrics, 'box') else 0.0
box_map = float(val_metrics.box.map) if hasattr(val_metrics, 'box') else 0.0
seg_map50 = float(val_metrics.seg.map50) if hasattr(val_metrics, 'seg') else 0.0
seg_map = float(val_metrics.seg.map) if hasattr(val_metrics, 'seg') else 0.0

metrics_dict = {
    "mask_mAP50": round(seg_map50, 4),
    "mask_mAP50_95": round(seg_map, 4),
    "box_mAP50": round(box_map50, 4),
    "box_mAP50_95": round(box_map, 4),
    "mean_precision": round(float(val_metrics.box.mp), 4) if hasattr(val_metrics, 'box') else 0.0,
    "mean_recall": round(float(val_metrics.box.mr), 4) if hasattr(val_metrics, 'box') else 0.0,
    "total_training_images": total_train,
    "total_validation_images": total_val
}

print("\n" + "=" * 70)
print("FINAL MEGA-MODEL VALIDATION ACCURACY:")
print(f" • Polygon Mask mAP@50:    {metrics_dict['mask_mAP50']:.4f}")
print(f" • Polygon Mask mAP@50-95: {metrics_dict['mask_mAP50_95']:.4f}")
print(f" • Bounding Box mAP@50:    {metrics_dict['box_mAP50']:.4f}")
print(f" • Bounding Box mAP@50-95: {metrics_dict['box_mAP50_95']:.4f}")
print(f" • Mean Precision:         {metrics_dict['mean_precision']:.4f}")
print(f" • Mean Recall:            {metrics_dict['mean_recall']:.4f}")
print("=" * 70)

# 7. Export Model Checkpoints
best_weights = "/kaggle/working/RoadSense_YOLO11_Seg/mega_model/weights/best.pt"
out_pothole_seg = "/kaggle/working/pothole_yolo11_seg.pt"
out_compat = "/kaggle/working/pothole_yolov8.pt"

if os.path.exists(best_weights):
    shutil.copy(best_weights, out_pothole_seg)
    shutil.copy(best_weights, out_compat)
    print(f"\n[SUCCESS] Deployed model saved to: {out_pothole_seg}")
    print(f"Size: {os.path.getsize(out_pothole_seg)/(1024*1024):.2f} MB")
else:
    for root, dirs, files in os.walk("/kaggle/working/RoadSense_YOLO11_Seg"):
        for f in files:
            if f.endswith(".pt"):
                src = os.path.join(root, f)
                shutil.copy(src, out_pothole_seg)
                shutil.copy(src, out_compat)
                print(f"Copied fallback weights {src} -> {out_pothole_seg}")
                break

with open("/kaggle/working/metrics_seg.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

print("\nMEGA-SCALE YOLO11 SEGMENTATION TRAINING RUN COMPLETE!")
