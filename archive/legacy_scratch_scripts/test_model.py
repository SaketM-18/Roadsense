import os
import sys

# Ensure Python can find the ml_pipeline modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from inference import run_inference

def main():
    print("=== Road Health Model Tester ===")
    
    # 1. Define the path to test file dynamically
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(ml_dir)
    test_dir = os.path.join(main_dir, "test")
    
    test_file = None
    for folder_name in ["Four", "one", "two", "Five", "Three"]:
        candidate_dir = os.path.join(test_dir, folder_name)
        if os.path.exists(candidate_dir):
            csvs = [os.path.join(candidate_dir, f) for f in os.listdir(candidate_dir) if f.startswith("roadsense_raw_") and f.endswith(".csv")]
            if csvs:
                test_file = csvs[0]
                break
    
    # 2. Define the path to the model (prefer v2 ensemble)
    model_path = os.path.join(ml_dir, "road_health_rf_model_v2.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(ml_dir, "road_health_rf_model.pkl")

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
