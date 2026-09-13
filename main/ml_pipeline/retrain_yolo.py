import os
import shutil
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, 'cv_pipeline', 'pothole_yolov8.pt')
DATA_YAML = os.path.join(BASE_DIR, 'ml_pipeline', 'shadow_dataset', 'data.yaml')
PROJECT_DIR = os.path.join(BASE_DIR, 'ml_pipeline', 'yolo_training')

def retrain_shadows():
    print("=" * 60)
    print("YOLOv8 SHADOW FINE-TUNING (HARD NEGATIVE MINING)")
    print("=" * 60)
    
    if not os.path.exists(DATA_YAML):
        print("Error: data.yaml not found. Run extract_shadows.py first.")
        return
        
    print(f"Loading base model: {MODEL_FILE}...")
    model = YOLO(MODEL_FILE)
    
    print("\nStarting Training on Background Images...")
    results = model.train(
        data=DATA_YAML,
        epochs=15,
        imgsz=640,
        project=PROJECT_DIR,
        name='shadow_tuning',
        exist_ok=True,
        patience=5,
        save=True
    )
    
    best_weights = os.path.join(PROJECT_DIR, 'shadow_tuning', 'weights', 'best.pt')
    if os.path.exists(best_weights):
        print(f"\nTraining Complete! Saving new weights to {MODEL_FILE}")
        shutil.copy(best_weights, MODEL_FILE)
        print("[SUCCESS] YOLOv8 is now immune to the extracted shadows.")
    else:
        print("\nError: Training completed but best.pt was not found.")

if __name__ == '__main__':
    retrain_shadows()
