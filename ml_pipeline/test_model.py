import os
import sys

# Ensure Python can find the ml_pipeline modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from inference import run_inference

def main():
    print("=== Road Health Model Tester ===")
    
    # 1. Define the path to your new test file
    # I found the file in your 'test' folder!
    base_dir = r"d:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health"
    test_file = os.path.join(base_dir, r"test\Acceleration with g 2026-06-22 19-05-32\Raw Data.csv")
    
    # 2. Define the path to the Brain (the saved .pkl model)
    model_path = os.path.join(base_dir, r"ml_pipeline\road_health_rf_model.pkl")

    # 3. Create a fresh DataProcessor (UPGRADED: Bandpass + Absolute Acceleration)
    processor = DataProcessor(
        data_dir="",          # Not needed for inference
        low_cutoff=2.0,       # Hz - removes gravity & braking/acceleration sway
        high_cutoff=15.0,     # Hz - removes engine vibration noise
        filter_order=4,
        window_size_sec=1.0,  # 1-second chunks
        overlap_pct=0.8       # 80% overlap so we don't chop bumps
    )

    # 4. Run the test!
    if os.path.exists(test_file):
        print(f"Found test file! Running the AI...")
        run_inference(file_path=test_file, model_path=model_path, data_processor=processor)
    else:
        print(f"Error: Could not find the file at {test_file}")

if __name__ == "__main__":
    main()
