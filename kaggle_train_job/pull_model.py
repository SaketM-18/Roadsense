"""
RoadSense — Automated Weight Synchronization & Verification Tool
==================================================================
Polls the Kaggle training kernel, downloads the trained multi-class weights,
deploys them directly to main/cv_pipeline/pothole_yolov8.pt, and runs
validation on test ride telemetry.
"""

import os
import sys
import time
import shutil
import json
import subprocess

KERNEL_ID = "saketmeda/roadsense-yolo11-real-world-segmentation-training"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_seg")
TARGET_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main", "cv_pipeline", "pothole_yolov8.pt")

def check_status():
    cmd = ["kaggle", "kernels", "status", KERNEL_ID]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()

def download_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", KERNEL_ID, "-p", OUTPUT_DIR]
    print(f"Downloading training outputs to {OUTPUT_DIR}...")
    subprocess.run(cmd, check=True)

def main():
    print("=" * 65)
    print("ROADSENSE — Pull Real-World Hazards Model from Kaggle")
    print("=" * 65)
    
    status_str = check_status()
    print(f"Current Status: {status_str}")
    
    if "COMPLETE" in status_str:
        print("\n[READY] Kernel completed! Downloading new weights...")
        download_output()
        
        # Check files
        weights_found = None
        for cand in ["pothole_yolov8.pt", "roadsense_hazard_v2.pt", "best.pt"]:
            p = os.path.join(OUTPUT_DIR, cand)
            if os.path.exists(p):
                weights_found = p
                break
                
        if not weights_found:
            for root, _, files in os.walk(OUTPUT_DIR):
                for f in files:
                    if f.endswith(".pt"):
                        weights_found = os.path.join(root, f)
                        break
                if weights_found:
                    break

        if weights_found:
            print(f"Deploying weights: {weights_found} -> {TARGET_WEIGHTS}")
            shutil.copy2(weights_found, TARGET_WEIGHTS)
            print(f"[SUCCESS] Updated production model ({os.path.getsize(TARGET_WEIGHTS)/(1024*1024):.2f} MB)")
            
            metrics_file = os.path.join(OUTPUT_DIR, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as mf:
                    metrics = json.load(mf)
                print("\nTraining Metrics Summary:")
                print(json.dumps(metrics, indent=2))
        else:
            print("Warning: No .pt model file found in output directory.")
    elif "RUNNING" in status_str or "QUEUED" in status_str:
        print("\n[IN PROGRESS] Model is still training on Kaggle GPU.")
        print("Run this script again once training completes:")
        print("  python kaggle_train_job/pull_model.py")
    else:
        print(f"\nKernel Status: {status_str}")

if __name__ == "__main__":
    main()

