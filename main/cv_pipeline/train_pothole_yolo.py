"""
RoadSense — Standalone GPU YOLO Retraining Script
=================================================
Usage:
    python train_pothole_yolo.py [--data <data.yaml>] [--epochs 50] [--model yolov8s.pt]

Requirements:
    pip install ultralytics roboflow torch
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on road damage/pothole dataset")
    parser.add_argument('--data', type=str, default='pothole_dataset/data.yaml', help='Path to dataset data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Backbone model (yolov8n.pt or yolov8s.pt)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    args = parser.parse_args()

    print("=" * 60)
    print("ROADSENSE — YOLOv8 Pothole Vision Retraining")
    print(f"Backbone: {args.model} | Epochs: {args.epochs} | ImgSz: {args.imgsz}")
    print("=" * 60)

    if not os.path.exists(args.data):
        print(f"Dataset descriptor '{args.data}' not found.")
        print("Please download an annotated dataset (e.g. from Roboflow or RDD2022) into './pothole_dataset'.")
        sys.exit(1)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=15,
        workers=4,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project='RoadSense_YOLO',
        name='pothole_detector_v2'
    )

    metrics = model.val()
    print("\n" + "=" * 60)
    print("TRAINING & VALIDATION COMPLETE")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("=" * 60)
    print("Best weights saved to: RoadSense_YOLO/pothole_detector_v2/weights/best.pt")
    print("To deploy in RoadSense, copy best.pt to: main/cv_pipeline/pothole_yolov8.pt")

if __name__ == '__main__':
    main()

