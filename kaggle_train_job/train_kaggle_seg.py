import os
import sys
import glob
import shutil
import random
import subprocess
import json

print("=" * 70)
print("ROADSENSE — Real-World YOLO11 Instance Segmentation Training")
print("Target: Polygon Masks for Potholes, Cracks, & Manholes")
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

# Set random seed
random.seed(42)

# 2. Setup Working Dataset Structure
work_dir = "/kaggle/working/unified_seg_dataset"
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

# 3. Discover and Ingest Mounted Segmentation Datasets
input_root = "/kaggle/input"
print("\nInspecting mounted Kaggle datasets in /kaggle/input:")
for entry in os.listdir(input_root):
    entry_path = os.path.join(input_root, entry)
    print(f" • Found: {entry}")

# A. Ingest Pothole Segmentation Dataset (farzadnekouei/pothole-image-segmentation-dataset)
pothole_seg_train_img = None
pothole_seg_val_img = None

for root, dirs, files in os.walk(input_root):
    norm = root.replace("\\", "/").lower()
    if "pothole" in norm and "segmentation" in norm and "roboflow" not in norm:
        if "train" in norm and "images" in dirs:
            pothole_seg_train_img = os.path.join(root, "images")
        elif ("val" in norm or "valid" in norm) and "images" in dirs:
            pothole_seg_val_img = os.path.join(root, "images")

# Fallback: search for any train/images in input
if not pothole_seg_train_img:
    for root, dirs, files in os.walk(input_root):
        norm = root.replace("\\", "/").lower()
        if "pothole" in norm and "train" in norm and "images" in dirs:
            pothole_seg_train_img = os.path.join(root, "images")
        elif "pothole" in norm and ("val" in norm or "valid" in norm) and "images" in dirs:
            pothole_seg_val_img = os.path.join(root, "images")

pothole_added = 0
if pothole_seg_train_img:
    pothole_train_lbl = pothole_seg_train_img.replace("images", "labels")
    for img_file in os.listdir(pothole_seg_train_img):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(pothole_seg_train_img, img_file)
            src_lbl = os.path.join(pothole_train_lbl, lbl_file)
            if os.path.exists(src_lbl):
                dst_img = os.path.join(train_img_dir, f"seg1_{img_file}")
                dst_lbl = os.path.join(train_lbl_dir, f"seg1_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)
                pothole_added += 1

if pothole_seg_val_img:
    pothole_val_lbl = pothole_seg_val_img.replace("images", "labels")
    for img_file in os.listdir(pothole_seg_val_img):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(pothole_seg_val_img, img_file)
            src_lbl = os.path.join(pothole_val_lbl, lbl_file)
            if os.path.exists(src_lbl):
                dst_img = os.path.join(val_img_dir, f"seg1_{img_file}")
                dst_lbl = os.path.join(val_lbl_dir, f"seg1_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)

print(f"[Dataset 1] Ingested {pothole_added} high-density polygon pothole segmentation samples.")

# B. Ingest Real-World Multi-Class Polygon Hazards (lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)
rdd_images_dir = None
rdd_labels_dir = None

for root, dirs, files in os.walk(input_root):
    norm = root.replace("\\", "/").lower()
    if "road-damage" in norm or "arcioni" in norm:
        # We specifically want polygon labels (in 'labels/')
        if "labels" in dirs and "labels-yolo" not in os.path.join(root, "labels").lower():
            rdd_labels_dir = os.path.join(root, "labels")
        if "images" in dirs:
            rdd_images_dir = os.path.join(root, "images")

rdd_added = 0
if rdd_images_dir and rdd_labels_dir:
    print(f"\n[Dataset 2] Found polygon road hazard dataset at:\n  Images: {rdd_images_dir}\n  Labels: {rdd_labels_dir}")
    all_imgs = [f for f in os.listdir(rdd_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_imgs)
    
    split_idx = int(0.85 * len(all_imgs))
    train_rdd = all_imgs[:split_idx]
    val_rdd = all_imgs[split_idx:]
    
    for split_imgs, target_i_dir, target_l_dir in [(train_rdd, train_img_dir, train_lbl_dir), (val_rdd, val_img_dir, val_lbl_dir)]:
        for img_file in split_imgs:
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(rdd_images_dir, img_file)
            src_lbl = os.path.join(rdd_labels_dir, lbl_file)
            
            if os.path.exists(src_lbl):
                dst_img = os.path.join(target_i_dir, f"rdd_{img_file}")
                dst_lbl = os.path.join(target_l_dir, f"rdd_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)
                rdd_added += 1

print(f"[Dataset 2] Ingested {rdd_added} polygon road hazard samples (Potholes, Cracks, Manholes).")

total_train = len(os.listdir(train_img_dir))
total_val = len(os.listdir(val_img_dir))
print(f"\nFinal Unified Dataset Statistics:")
print(f" • Training Images:   {total_train}")
print(f" • Validation Images: {total_val}")
print(f" • Total Volume:      {total_train + total_val} segmented road scenes")

if total_train == 0 or total_val == 0:
    print("ERROR: Ingestion failed to find training images. Dumping /kaggle/input structure:")
    for r, d, f in os.walk(input_root):
        print(" ", r, d[:3], f[:3])
    sys.exit(1)

# 4. Generate data_seg.yaml with Multi-Class Polygon Masks
yaml_path = "/kaggle/working/data_seg.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {train_img_dir}\n")
    f.write(f"val: {val_img_dir}\n")
    f.write("nc: 3\n")
    f.write("names: ['pothole', 'crack', 'manhole']\n")

print(f"\nGenerated unified YAML at {yaml_path}:")
with open(yaml_path, "r") as f:
    print(f.read())

# 5. Train YOLO11m-Seg (Medium Instance Segmentation Backbone)
print("\n" + "=" * 70)
print("Starting YOLO11m-Seg Training on Tesla T4 GPU...")
print("Task: Instance Segmentation (Boxes + Pixel Polygon Masks)")
print("Backbone: yolo11m-seg.pt (22.4M Parameters)")
print("Classes: [0: Pothole, 1: Crack, 2: Manhole]")
print("=" * 70)

# Load pretrained YOLO11 medium segmentation model
model = YOLO("yolo11m-seg.pt")

results = model.train(
    data=yaml_path,
    epochs=40,
    imgsz=640,
    batch=16,
    patience=12,
    workers=4,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    mosaic=1.0,      # Small defect contour recognition
    mixup=0.15,      # Road texture blending
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,       # Glare and tree shadow invariance
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    project="/kaggle/working/RoadSense_YOLO_Seg",
    name="seg_model",
    verbose=True
)

# 6. Post-Training Validation (Box & Mask Metrics)
print("\n" + "=" * 70)
print("RUNNING FINAL VALIDATION ACROSS POLYGON MASKS & BOUNDING BOXES...")
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
    "mean_recall": round(float(val_metrics.box.mr), 4) if hasattr(val_metrics, 'box') else 0.0
}

print("\n" + "=" * 70)
print("FINAL SEGMENTATION VALIDATION ACCURACY:")
print(f" • Polygon Mask mAP@50:    {metrics_dict['mask_mAP50']:.4f}")
print(f" • Polygon Mask mAP@50-95: {metrics_dict['mask_mAP50_95']:.4f}")
print(f" • Bounding Box mAP@50:    {metrics_dict['box_mAP50']:.4f}")
print(f" • Bounding Box mAP@50-95: {metrics_dict['box_mAP50_95']:.4f}")
print("=" * 70)

# 7. Export Production Weights
best_weights = "/kaggle/working/RoadSense_YOLO_Seg/seg_model/weights/best.pt"
out_pothole_seg = "/kaggle/working/pothole_yolo11_seg.pt"
out_compat = "/kaggle/working/pothole_yolov8.pt"

if os.path.exists(best_weights):
    shutil.copy(best_weights, out_pothole_seg)
    shutil.copy(best_weights, out_compat)
    print(f"\n[SUCCESS] Segmentation model saved to: {out_pothole_seg}")
    print(f"Size: {os.path.getsize(out_pothole_seg)/(1024*1024):.2f} MB")
else:
    for root, dirs, files in os.walk("/kaggle/working/RoadSense_YOLO_Seg"):
        for f in files:
            if f.endswith(".pt"):
                src = os.path.join(root, f)
                shutil.copy(src, out_pothole_seg)
                shutil.copy(src, out_compat)
                print(f"Copied fallback weights {src} -> {out_pothole_seg}")
                break

with open("/kaggle/working/metrics_seg.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

print("\nKAGGLE YOLO11 INSTANCE SEGMENTATION TRAINING COMPLETED SUCCESSFULLY!")

