import cv2, os, numpy as np
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
base = os.path.dirname(os.path.dirname(current_dir))
model_path = os.path.join(current_dir, 'pothole_yolov8.pt')
model = YOLO(model_path)

timestamps = [8.0, 30.0, 55.0, 72.0, 89.0, 116.0]
out_dir = os.path.join(base, 'Videos', 'stabilized_frames')
os.makedirs(out_dir, exist_ok=True)

for vid_num in [1, 2]:
    vid_dir = os.path.join(base, f'Videos/Video{vid_num}')
    orig_file = [f for f in os.listdir(vid_dir) if 'WhatsApp' in f][0]
    orig_path = os.path.join(vid_dir, orig_file)
    stab_path = os.path.join(vid_dir, f'stabilized_ride{vid_num}.mp4')
    
    print(f'\n===== VIDEO {vid_num} =====')
    
    # Blur comparison
    cap_orig = cv2.VideoCapture(orig_path)
    cap_stab = cv2.VideoCapture(stab_path)
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    
    print('Blur Score Comparison (higher = sharper):')
    print(f'  Time   Original   Stabilized   Change')
    for t in timestamps:
        frame_num = int(t * fps)
        if frame_num >= cap_orig.get(cv2.CAP_PROP_FRAME_COUNT):
            continue
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        cap_stab.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, f_orig = cap_orig.read()
        _, f_stab = cap_stab.read()
        if f_orig is None or f_stab is None:
            continue
        blur_orig = cv2.Laplacian(cv2.cvtColor(f_orig, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        blur_stab = cv2.Laplacian(cv2.cvtColor(f_stab, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        pct = ((blur_stab - blur_orig) / blur_orig * 100) if blur_orig > 0 else 0
        print(f'  {t:.0f}s     {blur_orig:.1f}      {blur_stab:.1f}       {pct:+.0f}%')
        
        # Save stabilized frame for visual comparison
        if vid_num == 1:
            cv2.imwrite(os.path.join(out_dir, f'stab_{t:.0f}s.jpg'), f_stab)
    
    cap_orig.release()
    cap_stab.release()
    
    # Re-run YOLO on stabilized video
    print(f'\nRunning YOLO on stabilized video...')
    cap = cv2.VideoCapture(stab_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    detections = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 15 == 0:
            results = model.predict(frame, conf=0.5, verbose=False)
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    time_sec = frame_idx / fps
                    detections.append((time_sec, conf))
        frame_idx += 1
    cap.release()
    
    print(f'Stabilized YOLO detections (conf>0.5): {len(detections)}')
    high_conf = [d for d in detections if d[1] > 0.7]
    print(f'High confidence (conf>0.7): {len(high_conf)}')
    for t, c in high_conf:
        print(f'  {t:.1f}s  conf={c:.2f}')

print('\n===== COMPARISON DONE =====')
