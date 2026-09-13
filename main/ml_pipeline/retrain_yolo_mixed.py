import os
import shutil
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, 'cv_pipeline', 'pothole_yolov8.pt')
DATA_YAML = os.path.join(BASE_DIR, 'ml_pipeline', 'mixed_dataset', 'data.yaml')
PROJECT_DIR = os.path.join(BASE_DIR, 'ml_pipeline', 'yolo_training')

def retrain_mixed():
    print("=" * 60)
    print("YOLOv8 MIXED DATASET FINE-TUNING")
    print("=" * 60)
    
    if not os.path.exists(DATA_YAML):
        print("Error: data.yaml not found. Run build_mixed_dataset.py first.")
        return
        
    print(f"Loading untainted base model (yolov8n.pt)...")
    model = YOLO('yolov8n.pt')
    
    print("\nStarting Training on Mixed Dataset (Potholes + Shadows)...")
    results = model.train(
        data=DATA_YAML,
        epochs=15,
        imgsz=640,
        project=PROJECT_DIR,
        name='mixed_tuning',
        exist_ok=True,
        patience=5,
        save=True
    )
    
    best_weights = os.path.join(PROJECT_DIR, 'mixed_tuning', 'weights', 'best.pt')
    if os.path.exists(best_weights):
        print(f"\nTraining Complete! Overwriting corrupted model at {MODEL_FILE}")
        shutil.copy(best_weights, MODEL_FILE)
        print("[SUCCESS] YOLOv8 is now trained to recognize potholes AND ignore shadows.")
    else:
        print("\nError: Training completed but best.pt was not found.")

if __name__ == '__main__':
    retrain_mixed()
