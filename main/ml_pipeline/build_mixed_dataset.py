import os
import shutil
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(BASE_DIR, 'Pothole Detection.v3i.yolov8.zip')
DATASET_DIR = os.path.join(BASE_DIR, 'ml_pipeline', 'mixed_dataset')
SHADOW_DIR = os.path.join(BASE_DIR, 'ml_pipeline', 'shadow_dataset')

def build_dataset():
    print("=" * 60)
    print("BUILDING ULTIMATE MIXED DATASET")
    print("=" * 60)
    
    if not os.path.exists(ZIP_PATH):
        print(f"Error: Could not find {ZIP_PATH}")
        return
        
    if os.path.exists(DATASET_DIR):
        print("Cleaning up old mixed dataset...")
        shutil.rmtree(DATASET_DIR)
        
    print(f"1. Extracting Roboflow Dataset ({os.path.basename(ZIP_PATH)})...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
        
    print("2. Fixing data.yaml paths...")
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            lines = f.readlines()
            
        with open(yaml_path, 'w') as f:
            for line in lines:
                # Roboflow sometimes hardcodes the path, we need to make it relative or absolute to our dir
                if line.startswith('train:'):
                    f.write(f"train: {os.path.join(DATASET_DIR, 'train', 'images')}\n")
                elif line.startswith('val:'):
                    f.write(f"val: {os.path.join(DATASET_DIR, 'valid', 'images')}\n")
                elif line.startswith('test:'):
                    # Some datasets have test, some don't. Safe to just update path if it exists
                    f.write(f"test: {os.path.join(DATASET_DIR, 'test', 'images')}\n")
                else:
                    f.write(line)
                    
    print("3. Injecting Shadow (Hard Negative) Images...")
    shadow_img_dir = os.path.join(SHADOW_DIR, 'images', 'train')
    shadow_lbl_dir = os.path.join(SHADOW_DIR, 'labels', 'train')
    
    target_img_dir = os.path.join(DATASET_DIR, 'train', 'images')
    target_lbl_dir = os.path.join(DATASET_DIR, 'train', 'labels')
    
    if not os.path.exists(target_img_dir):
        print("Error: train/images folder not found in extracted Roboflow dataset.")
        return
        
    count = 0
    if os.path.exists(shadow_img_dir):
        for file in os.listdir(shadow_img_dir):
            if file.endswith('.jpg'):
                shutil.copy(os.path.join(shadow_img_dir, file), os.path.join(target_img_dir, file))
                shutil.copy(os.path.join(shadow_lbl_dir, file.replace('.jpg', '.txt')), os.path.join(target_lbl_dir, file.replace('.jpg', '.txt')))
                count += 1
                
    print(f"Successfully injected {count} shadow images into the training set.")
    print("\n[SUCCESS] Mixed dataset is fully compiled and ready for training!")

if __name__ == '__main__':
    build_dataset()
