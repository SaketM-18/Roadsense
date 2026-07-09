import cv2
import os
import csv
from ultralytics import YOLO

def analyze_video(video_path, model_path, output_csv):
    print(f"Loading YOLOv8 model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have downloaded a pothole-specific YOLOv8 model (.pt file).")
        return

    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    detections = []
    frame_count = 0

    print("Analyzing video frame-by-frame... (This may take a while)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO inference (conf=0.5 means only confident detections)
        # We set verbose=False to keep the terminal clean
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Calculate the exact timestamp of this frame
        current_time_sec = frame_count / fps
        
        # Check if YOLO found anything
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # We assume class 0 is 'pothole'. Depending on the pre-trained model, you might need to adjust this.
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                # If the model detects a pothole, log the timestamp
                if "pothole" in class_name.lower():
                    detections.append({
                        "video_time_sec": round(current_time_sec, 3),
                        "confidence": round(conf, 2)
                    })
                    print(f"--> Pothole detected at {round(current_time_sec, 2)}s (Confidence: {conf:.2f})")
                    
        frame_count += 1

    cap.release()

    # Save to CSV
    print(f"\nSaving {len(detections)} detections to {output_csv}...")
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["video_time_sec", "confidence"])
        writer.writeheader()
        writer.writerows(detections)
        
    print("Done!")

if __name__ == "__main__":
    # Update these paths to match your actual files after your ride
    base_dir = r"d:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health"
    
    # Example video file
    my_video = os.path.join(base_dir, r"test\sample_ride.mp4")
    
    # Local pre-trained pothole model (downloaded from HuggingFace)
    my_model = os.path.join(base_dir, r"cv_pipeline\pothole_yolov8.pt")
    
    # Output CSV
    out_csv = os.path.join(base_dir, r"cv_pipeline\yolo_timestamps.csv")
    
    # Uncomment the line below to run the analysis once you have the files
    # analyze_video(my_video, my_model, out_csv)
    print("Script ready! Provide the .mp4 video path in 'my_video', then uncomment the analyze_video() line.")
