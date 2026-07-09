import os
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from inference import run_inference

def main():
    # Define paths
    base_dir = r"d:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health"
    data_dir = os.path.join(base_dir, "accelaration with g data")
    model_save_path = os.path.join(os.path.dirname(__file__), "road_health_rf_model.pkl")

    print("=== Road Health Detection ML Pipeline ===")

    # 1. Initialize DataProcessor (UPGRADED: Bandpass Filter + Absolute Acceleration)
    processor = DataProcessor(
        data_dir=data_dir,
        low_cutoff=2.0,       # Hz - removes gravity & braking/acceleration sway
        high_cutoff=15.0,     # Hz - removes engine vibration noise
        filter_order=4,
        window_size_sec=1.0,  
        overlap_pct=0.8       # 80% overlap to avoid chopping bumps in half
    )

    # 2. Extract Features from Training Data
    print("\n--- Phase 1: Data Processing & Feature Extraction ---")
    df_features = processor.load_and_process_training_data()

    if df_features.empty:
        print("Error: No training data could be processed. Exiting.")
        return

    # 3. Train the Model
    print("\n--- Phase 2: Model Training ---")
    trainer = ModelTrainer(random_state=42)
    trainer.train(df_features, save_path=model_save_path)

    # 4. Inference on a sample continuous file
    print("\n--- Phase 3: Inference Test ---")
    
    # Try to find a continuous data file for testing
    continuous_dir = os.path.join(data_dir, "Raw continuous data")
    test_file = None
    if os.path.exists(continuous_dir):
        for root, dirs, files in os.walk(continuous_dir):
            for file in files:
                if file.endswith(".csv") and "Raw Data" in file:
                    test_file = os.path.join(root, file)
                    break
            if test_file:
                break
                
    if test_file:
        run_inference(file_path=test_file, model_path=model_save_path, data_processor=processor)
    else:
        print("No continuous data file found for inference testing.")

if __name__ == "__main__":
    main()
