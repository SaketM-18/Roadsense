# Computer Vision & Sensor Fusion Pipeline

This folder contains the scripts needed to cross-reference video data with accelerometer data.

## Step 1: Get the Pre-trained YOLOv8 Pothole Model
Standard YOLOv8 models are trained on the COCO dataset (dogs, cats, cars, etc.) and do not know what a pothole is. You must download a specific **Pothole Detection Model**.

1. Go to [Roboflow Universe - Pothole Detection](https://universe.roboflow.com/search?q=pothole)
2. Pick a highly-rated YOLOv8 project (ensure the format is YOLOv8).
3. Download the weights file. It should end in `.pt` (e.g., `best.pt`).
4. Rename that file to `pothole_yolov8.pt` and place it in this `cv_pipeline` folder.

## Step 2: Run the Vision Detector
Once you have your ride video (e.g., `ride_video.mp4`):
1. Open `detect_potholes.py` and update the `my_video` path at the bottom.
2. Run the script:
   ```bash
   python detect_potholes.py
   ```
3. This will generate `yolo_timestamps.csv`.

## Step 3: Run the Fusion
Once you have the vision timestamps and your sensor CSV (from phyphox/PWA):
1. Open `sync_and_fuse.py` and update the file paths at the bottom.
2. Set your `clap_sync_offset` based on when you clapped your hands in front of the camera vs when the accelerometer spiked.
3. Run the script:
   ```bash
   python sync_and_fuse.py
   ```
4. This will generate your perfectly labeled `fused_labeled_data.csv` ready for Machine Learning!
