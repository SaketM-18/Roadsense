import os
import sys
import glob
import shutil
import random
import subprocess
import json

print("=" * 70)
print("ROADSENSE — Real-World Multi-Class Hazard Training (YOLOv8m)")
print("Real-World Elements: Potholes, Cracks, Manholes, & Road Negatives")
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

# Set random seed for reproducibility
random.seed(42)

# 2. Setup Working Dataset Structure
work_dir = "/kaggle/working/unified_dataset"
train_img_dir = os.path.join(work_dir, "train", "images")
train_lbl_dir = os.path.join(work_dir, "train", "labels")
val_img_dir = os.path.join(work_dir, "val", "images")
val_lbl_dir = os.path.join(work_dir, "val", "labels")

for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

print(f"\nCreated unified dataset target directories under: {work_dir}")

def safe_link_or_copy(src, dst):
    """Attempt symlink first for instant zero-copy setup, fallback to copy."""
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

# 3. Discover and Ingest Mounted Datasets
input_root = "/kaggle/input"
print("\nInspecting mounted Kaggle datasets in /kaggle/input:")
for entry in os.listdir(input_root):
    entry_path = os.path.join(input_root, entry)
    print(f" • Found dataset: {entry} ({'Directory' if os.path.isdir(entry_path) else 'File'})")

# A. Ingest Potholes Dataset (anggadwisunarto/potholes-detection-yolov8 or similar)
pothole_train_img = None
pothole_val_img = None

for root, dirs, files in os.walk(input_root):
    norm = root.replace("\\", "/").lower()
    if "pothole" in norm:
        if "train" in norm and "images" in dirs:
            pothole_train_img = os.path.join(root, "images")
        elif ("val" in norm or "valid" in norm) and "images" in dirs:
            pothole_val_img = os.path.join(root, "images")

pothole_added = 0
if pothole_train_img:
    pothole_train_lbl = pothole_train_img.replace("images", "labels")
    for img_file in os.listdir(pothole_train_img):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(pothole_train_img, img_file)
            src_lbl = os.path.join(pothole_train_lbl, lbl_file)
            if os.path.exists(src_lbl):
                dst_img = os.path.join(train_img_dir, f"ds1_{img_file}")
                dst_lbl = os.path.join(train_lbl_dir, f"ds1_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)
                pothole_added += 1

if pothole_val_img:
    pothole_val_lbl = pothole_val_img.replace("images", "labels")
    for img_file in os.listdir(pothole_val_img):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(pothole_val_img, img_file)
            src_lbl = os.path.join(pothole_val_lbl, lbl_file)
            if os.path.exists(src_lbl):
                dst_img = os.path.join(val_img_dir, f"ds1_{img_file}")
                dst_lbl = os.path.join(val_lbl_dir, f"ds1_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)

print(f"[Dataset 1] Ingested {pothole_added} base pothole training samples.")

# B. Ingest Multi-Class Real-World Road Hazards (lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)
rdd_images_dir = None
rdd_labels_dir = None

for root, dirs, files in os.walk(input_root):
    norm = root.replace("\\", "/").lower()
    if "road-damage" in norm or "arcioni" in norm or "cracks" in norm:
        if "labels-yolo" in dirs:
            rdd_labels_dir = os.path.join(root, "labels-YOLO")
        elif "labels" in dirs and not rdd_labels_dir:
            rdd_labels_dir = os.path.join(root, "labels")
        if "images" in dirs:
            rdd_images_dir = os.path.join(root, "images")

rdd_added = 0
if rdd_images_dir and rdd_labels_dir:
    print(f"\n[Dataset 2] Found real-world hazards dataset at:\n  Images: {rdd_images_dir}\n  Labels: {rdd_labels_dir}")
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
                
                # Check if labels are polygons (more than 5 values per line) or standard YOLO (5 values)
                with open(src_lbl, "r") as lf:
                    lines = lf.readlines()
                
                converted_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) == 4:
                        # Already YOLO format [cls, xc, yc, w, h]
                        converted_lines.append(line.strip())
                    elif len(coords) >= 6:
                        # Polygon format [cls, x1, y1, x2, y2, ...] -> derive bounding box
                        xs = coords[0::2]
                        ys = coords[1::2]
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)
                        xc = (xmin + xmax) / 2.0
                        yc = (ymin + ymax) / 2.0
                        w = xmax - xmin
                        h = ymax - ymin
                        converted_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                
                if converted_lines:
                    safe_link_or_copy(src_img, dst_img)
                    with open(dst_lbl, "w") as out_lf:
                        out_lf.write("\n".join(converted_lines) + "\n")
                    rdd_added += 1

print(f"[Dataset 2] Ingested and harmonized {rdd_added} multi-class real-world samples.")

total_train = len(os.listdir(train_img_dir))
total_val = len(os.listdir(val_img_dir))
print(f"\nFinal Unified Dataset Statistics:")
print(f" • Training Images:   {total_train}")
print(f" • Validation Images: {total_val}")
print(f" • Total Volume:      {total_train + total_val} real-world road scenes")

if total_train == 0 or total_val == 0:
    print("ERROR: Ingestion failed to find training images. Dumping /kaggle/input structure:")
    for r, d, f in os.walk(input_root):
        print(" ", r, d[:3], f[:3])
    sys.exit(1)

# 4. Generate data.yaml with Multi-Class Real-World Elements
yaml_path = "/kaggle/working/data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {train_img_dir}\n")
    f.write(f"val: {val_img_dir}\n")
    f.write("nc: 3\n")
    f.write("names: ['pothole', 'crack', 'manhole']\n")

print(f"\nGenerated unified YAML at {yaml_path}:")
with open(yaml_path, "r") as f:
    print(f.read())

# 5. Fine-Tune YOLOv8m (Medium Backbone - 25.9M Parameters)
print("\n" + "=" * 70)
print("Starting YOLOv8m Multi-Class Training on Tesla T4 GPU...")
print("Backbone: YOLOv8m | Classes: [0: Pothole, 1: Crack, 2: Manhole]")
print("=" * 70)

model = YOLO("yolov8m.pt")

results = model.train(
    data=yaml_path,
    epochs=35,
    imgsz=640,
    batch=16,
    patience=10,
    workers=4,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    mosaic=1.0,      # Multi-defect context and small hazard recognition
    mixup=0.15,      # Blends road textures to avoid overfitting to asphalt color
    hsv_h=0.015,     # Robustness to hue variations
    hsv_s=0.7,       # Robustness to rain/wet surface saturation
    hsv_v=0.4,       # Robustness to direct sun glare and deep tree shadows
    degrees=10.0,    # Dashcam tilt/pitch tolerance
    translate=0.1,
    scale=0.5,       # Varied vehicle approach distances
    fliplr=0.5,
    project="/kaggle/working/RoadSense_YOLO",
    name="hazard_model",
    verbose=True
)

# 6. Comprehensive Post-Training Validation
print("\n" + "=" * 70)
print("RUNNING FINAL VALIDATION ACROSS REAL-WORLD ELEMENTS...")
print("=" * 70)

val_metrics = model.val()

class_names = ['pothole', 'crack', 'manhole']
per_class_metrics = {}

try:
    for i, name in enumerate(class_names):
        if i < len(val_metrics.box.maps):
            per_class_metrics[name] = {
                "mAP50": round(float(val_metrics.box.maps[i]), 4)
            }
except Exception as e:
    print(f"Note on per-class extraction: {e}")

overall_metrics = {
    "overall_mAP50": round(float(val_metrics.box.map50), 4),
    "overall_mAP50_95": round(float(val_metrics.box.map), 4),
    "mean_precision": round(float(val_metrics.box.mp), 4),
    "mean_recall": round(float(val_metrics.box.mr), 4),
    "per_class": per_class_metrics
}

print("\n" + "=" * 70)
print("FINAL VALIDATION ACCURACY:")
print(f" • Overall mAP@50:    {overall_metrics['overall_mAP50']:.4f}")
print(f" • Overall mAP@50-95: {overall_metrics['overall_mAP50_95']:.4f}")
print(f" • Precision:         {overall_metrics['mean_precision']:.4f}")
print(f" • Recall:            {overall_metrics['mean_recall']:.4f}")
for cname, cdata in per_class_metrics.items():
    print(f"   - {cname.capitalize():8s} mAP: {cdata.get('mAP50', 0):.4f}")
print("=" * 70)

# 7. Export Best Weights for Production Deployment
best_weights = "/kaggle/working/RoadSense_YOLO/hazard_model/weights/best.pt"
out_pothole = "/kaggle/working/pothole_yolov8.pt"
out_hazard = "/kaggle/working/roadsense_hazard_v2.pt"

if os.path.exists(best_weights):
    shutil.copy(best_weights, out_pothole)
    shutil.copy(best_weights, out_hazard)
    print(f"\n[SUCCESS] Production model saved to: {out_pothole}")
    print(f"Size: {os.path.getsize(out_pothole)/(1024*1024):.2f} MB")
else:
    for root, dirs, files in os.walk("/kaggle/working/RoadSense_YOLO"):
        for f in files:
            if f.endswith(".pt"):
                src = os.path.join(root, f)
                shutil.copy(src, out_pothole)
                shutil.copy(src, out_hazard)
                print(f"Copied fallback weights {src} -> {out_pothole}")
                break

with open("/kaggle/working/metrics.json", "w") as f:
    json.dump(overall_metrics, f, indent=2)

print("\nKAGGLE REAL-WORLD HAZARDS TRAINING COMPLETED SUCCESSFULLY!")
