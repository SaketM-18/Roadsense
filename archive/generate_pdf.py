import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ultralytics import YOLO

base = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health'
video_path = os.path.join(base, r'Videos\Video1\stabilized_ride1.mp4')
model = YOLO(os.path.join(base, r'cv_pipeline\pothole_yolov8.pt'))

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mask_y = int(h * 0.75)

print("Scanning video for true potholes (masking scooter)...")
detections = []
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if frame_idx % 15 == 0:
        masked_frame = frame.copy()
        cv2.rectangle(masked_frame, (0, mask_y), (w, h), (0, 0, 0), -1)
        results = model.predict(masked_frame, conf=0.5, verbose=False)
        has_pothole = False
        conf = 0
        for box in results[0].boxes:
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
            if y_center < mask_y:
                has_pothole = True
                conf = float(box.conf[0])
                break
        if has_pothole:
            detections.append((frame_idx/fps, conf))
    frame_idx += 1

cap.release()

unique_dets = []
for t, c in sorted(detections, key=lambda x: x[1], reverse=True):
    if not any(abs(t - ut) < 2.0 for ut, uc in unique_dets):
        unique_dets.append((t, c))

pothole_times = [t for t, c in unique_dets[:6]]
# Pick some normal times that are definitely far from any detection
normal_times = []
for t in [10.0, 25.0, 45.0, 60.0, 80.0, 105.0, 120.0, 5.0, 15.0]:
    if not any(abs(t - dt) < 3.0 for dt, _ in detections):
        normal_times.append(t)
        if len(normal_times) == 6:
            break

def get_frame_with_boxes(t, is_pothole):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    cap.release()
    if ret:
        if is_pothole:
            masked_frame = frame.copy()
            cv2.rectangle(masked_frame, (0, mask_y), (w, h), (0, 0, 0), -1)
            results = model.predict(masked_frame, conf=0.4, verbose=False)
            for box in results[0].boxes:
                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                if y_center < mask_y:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Pothole {float(box.conf[0]):.2f}", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

pdf_path = os.path.join(base, 'Road_Conditions_Report.pdf')

with PdfPages(pdf_path) as pdf:
    # Potholes
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
    fig.suptitle('Model Identified Potholes (Scooter Masked)', fontsize=16)
    for ax, t in zip(axes.flatten(), pothole_times):
        frame = get_frame_with_boxes(t, True)
        if frame is not None:
            ax.imshow(frame)
        ax.set_title(f'Pothole detected at {t:.1f}s')
        ax.axis('off')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # Normal Roads
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
    fig.suptitle('Model Identified Normal / Good Roads', fontsize=16)
    for ax, t in zip(axes.flatten(), normal_times):
        frame = get_frame_with_boxes(t, False)
        if frame is not None:
            ax.imshow(frame)
        ax.set_title(f'Normal road at {t:.1f}s')
        ax.axis('off')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f'Saved PDF report to {pdf_path}')
