import os
import sys
import glob
import shutil
import random
import subprocess
import json

print("=" * 70)
print("ROADSENSE — Real-World YOLO11 Mega-Dataset Instance Segmentation")
print("Data Sources: Farzad + Raunak + Arcioni (3,000+ Annotated Scenes)")
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

# 3. Discover and Ingest All Mounted Kaggle Datasets
input_root = "/kaggle/input"
print("\nScanning all mounted Kaggle datasets in /kaggle/input:")
for entry in os.listdir(input_root):
    entry_path = os.path.join(input_root, entry)
    print(f" • Found: {entry}")

ds1_count = 0
ds2_count = 0
ds3_count = 0

# A. Ingest Farzad Pothole Segmentation Dataset (farzadnekouei)
farzad_train_img = None
farzad_val_img = None

for root, dirs, files in os.walk(input_root):
    norm = root.replace("\\", "/").lower()
    if "farzad" in norm or "pothole_segmentation_yolov8" in norm:
        if "train" in norm and "images" in dirs:
            farzad_train_img = os.path.join(root, "images")
        elif ("val" in norm or "valid" in norm) and "images" in dirs:
            farzad_val_img = os.path.join(root, "images")

if farzad_train_img:
    farzad_train_lbl = farzad_train_img.replace("images", "labels")
    for img_file in os.listdir(farzad_train_img):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(farzad_train_img, img_file)
            src_lbl = os.path.join(farzad_train_lbl, lbl_file)
            if os.path.exists(src_lbl):
                dst_img = os.path.join(train_img_dir, f"fzd_{img_file}")
                dst_lbl = os.path.join(train_lbl_dir, f"fzd_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)
                ds1_count += 1

if farzad_val_img:
    farzad_val_lbl = farzad_val_img.replace("images", "labels")
    for img_file in os.listdir(farzad_val_img):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(farzad_val_img, img_file)
            src_lbl = os.path.join(farzad_val_lbl, lbl_file)
            if os.path.exists(src_lbl):
                dst_img = os.path.join(val_img_dir, f"fzd_{img_file}")
                dst_lbl = os.path.join(val_lbl_dir, f"fzd_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)

print(f"[Dataset 1 - Farzad] Ingested {ds1_count} high-density polygon pothole samples.")

# B. Ingest Raunak Kesharwani Segmentation Dataset
rk_train_img = None
rk_val_img = None

for root, dirs, files in os.walk(input_root):
    norm = root.replace("\\", "/").lower()
    if "raunak" in norm or "kesharwani" in norm:
        if "train" in norm and "images" in dirs:
            rk_train_img = os.path.join(root, "images")
        elif ("val" in norm or "valid" in norm) and "images" in dirs:
            rk_val_img = os.path.join(root, "images")

if rk_train_img:
    rk_train_lbl = rk_train_img.replace("images", "labels")
    for img_file in os.listdir(rk_train_img):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(rk_train_img, img_file)
            src_lbl = os.path.join(rk_train_lbl, lbl_file)
            if os.path.exists(src_lbl):
                dst_img = os.path.join(train_img_dir, f"rk_{img_file}")
                dst_lbl = os.path.join(train_lbl_dir, f"rk_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)
                ds2_count += 1

if rk_val_img:
    rk_val_lbl = rk_val_img.replace("images", "labels")
    for img_file in os.listdir(rk_val_img):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            src_img = os.path.join(rk_val_img, img_file)
            src_lbl = os.path.join(rk_val_lbl, lbl_file)
            if os.path.exists(src_lbl):
                dst_img = os.path.join(val_img_dir, f"rk_{img_file}")
                dst_lbl = os.path.join(val_lbl_dir, f"rk_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)

print(f"[Dataset 2 - Raunak] Ingested {ds2_count} diverse road pothole segmentation samples.")

# C. Ingest Lorenzo Arcioni Multi-Class Road Damage (Potholes, Cracks, Manholes)
rdd_images_dir = None
rdd_labels_dir = None

for root, dirs, files in os.walk(input_root):
    norm = root.replace("\\", "/").lower()
    if "road-damage" in norm or "arcioni" in norm:
        if "labels" in dirs and "labels-yolo" not in os.path.join(root, "labels").lower():
            rdd_labels_dir = os.path.join(root, "labels")
        if "images" in dirs:
            rdd_images_dir = os.path.join(root, "images")

if rdd_images_dir and rdd_labels_dir:
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
                dst_img = os.path.join(target_i_dir, f"arc_{img_file}")
                dst_lbl = os.path.join(target_l_dir, f"arc_{lbl_file}")
                safe_link_or_copy(src_img, dst_img)
                safe_link_or_copy(src_lbl, dst_lbl)
                ds3_count += 1

print(f"[Dataset 3 - Arcioni] Ingested {ds3_count} multi-class samples (Potholes, Cracks, Manholes).")

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
