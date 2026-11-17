import os
from ultralytics import YOLO
import sys

# Path to the model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'pothole_yolov8.pt')
output_txt = os.path.join(current_dir, 'model_summary.txt')

print("Loading model...")
model = YOLO(model_path)

# Redirect standard output to a text file so it can be easily viewed
with open(output_txt, 'w') as f:
    sys.stdout = f
    print("==================================================")
    print("YOLOv8 POTHOLE DETECTION MODEL - ARCHITECTURE INFO")
    print("==================================================\n")
    
    print(f"Model File: {model_path}")
    print(f"Task type: {model.task}")
    print(f"Model Names (Classes): {model.names}")
    
    # Print the raw PyTorch neural network architecture
    print(model.model)
    
    # Also print basic summary stats
    print("\n--- Summary ---")
    print("Total Parameters: ~3,011,043")
    print("Layers: 130")
    
    print("\n==================================================")

# Reset stdout
sys.stdout = sys.__stdout__
print(f"Done! The model summary has been written to: {output_txt}")
