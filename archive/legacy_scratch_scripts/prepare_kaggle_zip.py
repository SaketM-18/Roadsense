import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'ml_pipeline', 'mixed_dataset')
ZIP_OUT_PATH = os.path.join(BASE_DIR, 'RoadHealth_Kaggle_Dataset')

def prepare_kaggle():
    print("=" * 60)
    print("PACKAGING KAGGLE DATASET")
    print("=" * 60)
    
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    if os.path.exists(yaml_path):
        print("1. Rewriting data.yaml for Kaggle absolute paths...")
        yaml_content = """path: /kaggle/input/roadhealth-mixed-dataset
train: train/images
val: valid/images

names:
  0: pothole
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
    else:
        print("Error: data.yaml not found in mixed_dataset.")
        return
        
    print(f"2. Compressing dataset to {os.path.basename(ZIP_OUT_PATH)}.zip...")
    # shutil.make_archive automatically appends .zip
    shutil.make_archive(ZIP_OUT_PATH, 'zip', DATASET_DIR)
    
    print("\n[SUCCESS] Dataset is packaged and ready for Kaggle!")

if __name__ == '__main__':
    prepare_kaggle()
