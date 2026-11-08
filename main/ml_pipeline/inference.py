import os
import pandas as pd
import joblib
from data_processor import DataProcessor

def run_inference(file_path, model_path, data_processor=None):
    """
    Runs the full inference pipeline on a single continuous ride file.
    
    Args:
        file_path (str): Path to the CSV file to analyze.
        model_path (str): Path to the saved model (.pkl).
        data_processor (DataProcessor, optional): Pre-configured DataProcessor instance.
    """
    print(f"\n--- Running Inference on {os.path.basename(file_path)} ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
        
    feature_names_path = model_path.replace('.pkl', '_features.pkl')
    if not os.path.exists(feature_names_path):
        print(f"Error: Feature names file {feature_names_path} not found.")
        return

    # Load Model and Feature names
    model = joblib.load(model_path)
    expected_features = joblib.load(feature_names_path)
    
    if data_processor is None:
        # Initialize with default parameters if not provided
        data_processor = DataProcessor(data_dir="")
        
    print("Processing file (filtering and windowing)...")
    extracted_data = data_processor.process_file(file_path)
    
    if not extracted_data:
        print("No valid data extracted from the file.")
        return
        
    df_features = pd.DataFrame(extracted_data)
    
    # Ensure all expected features are present (in the exact order)
    X = df_features[expected_features]
    
    print(f"Extracted {len(X)} windows. Running predictions...")
    
    # UPGRADE: Use predict_proba instead of predict to add a CONFIDENCE THRESHOLD.
    # The old code picked whichever class had the highest probability, even if it was 34% vs 33%.
    # Now we require the model to be at least 70% confident before flagging an anomaly.
    # If it's not confident enough, we default to "Normal" (safe assumption: most road is normal).
    CONFIDENCE_THRESHOLD = 0.70
    
    probabilities = model.predict_proba(X)
    class_labels = model.classes_
    
    predictions = []
    for proba in probabilities:
        max_idx = proba.argmax()
        max_prob = proba[max_idx]
        predicted_class = class_labels[max_idx]
        
        # If the model isn't confident enough, or it's already Normal (0 or 'Normal'), keep it Normal
        is_normal = (predicted_class == 'Normal' or predicted_class == 0 or str(predicted_class).lower() == 'normal')
        if is_normal or max_prob < CONFIDENCE_THRESHOLD:
            predictions.append('Normal')
        else:
            predictions.append('Pothole' if predicted_class == 1 else str(predicted_class))
    
    # Add predictions back to the dataframe
    df_features['prediction'] = predictions
    
    # Filter only anomalies
    anomalies = df_features[df_features['prediction'] != 'Normal']
    
    if anomalies.empty:
        print("No potholes or speedbreakers detected in this file.")
    else:
        # Merge overlapping consecutive windows into single physical events
        merged_events = []
        current_event = None
        
        anomalies = anomalies.sort_values(by='start_time')
        
        for idx, row in anomalies.iterrows():
            if current_event is None:
                current_event = {'pred': row['prediction'], 'start': row['start_time'], 'end': row['end_time']}
            else:
                # If it's the same prediction and overlaps or is adjacent (within 0.1 seconds of the last end_time)
                if row['prediction'] == current_event['pred'] and row['start_time'] <= current_event['end'] + 0.1:
                    current_event['end'] = max(current_event['end'], row['end_time'])
                else:
                    merged_events.append(current_event)
                    current_event = {'pred': row['prediction'], 'start': row['start_time'], 'end': row['end_time']}
                    
        if current_event is not None:
            merged_events.append(current_event)

        # Pass 2: Detect "Severe Rough Patches"
        # If anomalies happen within 1.5 seconds of each other, group them into a chaotic patch.
        final_events = []
        if merged_events:
            current_cluster = merged_events[0].copy()
            events_in_cluster = 1
            
            for i in range(1, len(merged_events)):
                ev = merged_events[i]
                # Gap between anomalies <= 1.5 seconds
                if ev['start'] - current_cluster['end'] <= 1.5:
                    current_cluster['end'] = max(current_cluster['end'], ev['end'])
                    events_in_cluster += 1
                    current_cluster['pred'] = 'Severe Rough Patch'
                else:
                    final_events.append(current_cluster)
                    current_cluster = ev.copy()
                    events_in_cluster = 1
                    
            final_events.append(current_cluster)

        print(f"Detected {len(final_events)} physical anomalies (merged from {len(anomalies)} overlapping windows):")
        for ev in final_events:
            print(f"[{ev['start']:.2f}s - {ev['end']:.2f}s] : {ev['pred']}")
            
    return df_features
