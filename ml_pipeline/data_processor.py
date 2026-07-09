import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew

class DataProcessor:
    """
    DataProcessor handles the loading, signal processing, and feature extraction 
    of smartphone accelerometer data for road health detection.
    
    UPGRADE (Based on Research Paper):
    - Uses Absolute Acceleration (orientation-independent) instead of individual axes.
    - Uses a Bandpass Butterworth Filter (2-15 Hz) instead of a Low-Pass filter.
      This eliminates BOTH engine noise (>15 Hz) AND braking/acceleration sway (<2 Hz).
    """
    def __init__(self, data_dir, low_cutoff=2.0, high_cutoff=15.0, filter_order=4, window_size_sec=1.0, overlap_pct=0.8):
        """
        Initialize the DataProcessor.

        Args:
            data_dir (str): Base directory containing 'Potholes', 'Speedbreakers', and 'Raw continuous data'.
            low_cutoff (float): Low cutoff frequency in Hz for the bandpass filter.
                                Removes gravity (0 Hz) and slow braking/acceleration sway (<2 Hz).
            high_cutoff (float): High cutoff frequency in Hz for the bandpass filter.
                                 Removes engine vibrations (>15 Hz).
            filter_order (int): Order of the Butterworth filter.
            window_size_sec (float): Size of the sliding window in seconds for feature extraction.
            overlap_pct (float): Overlap percentage (0 to 1) between consecutive windows.
        """
        self.data_dir = data_dir
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.filter_order = filter_order
        self.window_size_sec = window_size_sec
        self.overlap_pct = overlap_pct

    def _calculate_sampling_rate(self, time_series):
        """
        Dynamically calculate the sampling rate (Hz) from the time series.
        Since the phyphox sensor rate can vary slightly, we take the average time difference.

        Args:
            time_series (pd.Series): The 'Time (s)' column.

        Returns:
            float: Estimated sampling frequency in Hz.
        """
        dt = np.mean(np.diff(time_series))
        if dt == 0:
            return 100.0 # fallback default to 100Hz
        return 1.0 / dt

    def _butter_bandpass_filter(self, data, fs):
        """
        Applies a Bandpass Butterworth filter to isolate road impact frequencies.
        
        Math Context (Research Paper Upgrade):
        A bandpass filter only allows frequencies BETWEEN the low and high cutoffs to pass.
        
        - Frequencies BELOW low_cutoff (2 Hz) are BLOCKED:
          This removes gravity (0 Hz) and the slow body sway caused by braking/accelerating.
          This is the KEY fix for the false positives we saw on the test ride!
          
        - Frequencies ABOVE high_cutoff (15 Hz) are BLOCKED:
          This removes the high-frequency engine RPM vibrations from the motorcycle.
          
        - Frequencies BETWEEN 2-15 Hz PASS THROUGH:
          This is the "sweet spot" where physical road impacts (potholes, speedbreakers) live.
          A pothole impact typically has energy in the 3-10 Hz range.
        
        We use `filtfilt` for zero-phase filtering (applies the filter forwards and backwards),
        ensuring that the phase (and thus the timing of the peaks) is not distorted.

        Args:
            data (np.array): Signal data to be filtered.
            fs (float): Sampling frequency of the signal.

        Returns:
            np.array: Filtered signal containing only road impact frequencies.
        """
        nyquist = 0.5 * fs
        low = self.low_cutoff / nyquist
        high = self.high_cutoff / nyquist
        
        # Ensure cutoffs are valid (strictly between 0 and 1)
        low = max(low, 0.01)
        high = min(high, 0.99)
        
        if low >= high:
            # Fallback: if cutoffs are invalid, just return raw data
            return data
            
        b, a = butter(self.filter_order, [low, high], btype='band', analog=False)
        y = filtfilt(b, a, data)
        return y

    def extract_features(self, window_data, fs):
        """
        Extracts time-domain and frequency-domain features from a sliding window.
        
        UPGRADE (Research Paper): 
        We now extract features from ONLY the bandpass-filtered Absolute Acceleration signal.
        This is orientation-independent: sqrt(x^2 + y^2 + z^2).
        No matter how the phone is mounted (tilted, sideways, upside down), 
        a pothole impact will always produce the same feature values.

        Features extracted:
        - Mean: Central tendency of the filtered signal (should be near 0 after bandpass).
        - Variance & Standard Deviation: Spread/energy of the signal. Higher = rougher road.
        - Peak-to-Peak (Range): Max minus Min. Indicates severity of impact.
        - RMS (Root Mean Square): Total energy of the signal in this window.
        - Kurtosis: "Spikiness" of the distribution. Very high kurtosis = sharp pothole impact.
        - Skewness: Asymmetry of the signal. Helps distinguish dips from rises.
        - Dominant Frequency: The frequency with the most power (Welch's method).
        - Spectral Energy: Total power in the frequency spectrum.

        Args:
            window_data (pd.DataFrame): Dataframe containing the window's data.
            fs (float): Sampling rate.

        Returns:
            dict: Dictionary of extracted features.
        """
        features = {}
        signal = window_data['abs_accel_filtered'].values
        
        # Time-Domain Features
        features['abs_mean'] = np.mean(signal)
        features['abs_var'] = np.var(signal)
        features['abs_std'] = np.std(signal)
        features['abs_ptp'] = np.ptp(signal)  # Peak-to-peak (max - min)
        features['abs_rms'] = np.sqrt(np.mean(signal**2))
        features['abs_kurtosis'] = kurtosis(signal, fisher=True, bias=False)
        features['abs_skew'] = skew(signal, bias=False)
        features['abs_max'] = np.max(np.abs(signal))
        
        # Frequency-Domain Features (Welch's method)
        # Welch's method estimates the power spectral density (PSD).
        try:
            nperseg = min(256, len(signal))
            if nperseg < 4:
                nperseg = len(signal)
            f, Pxx = welch(signal, fs, nperseg=nperseg)
            features['abs_dom_freq'] = f[np.argmax(Pxx)]
            features['abs_spectral_energy'] = np.sum(Pxx)
        except (ValueError, IndexError):
            features['abs_dom_freq'] = 0.0
            features['abs_spectral_energy'] = 0.0

        return features

    def process_file(self, filepath, label=None):
        """
        Processes a single CSV file:
        1. Reads data.
        2. Calculates sampling rate.
        3. Computes Absolute Acceleration (orientation-independent).
        4. Applies Bandpass Butterworth filter to isolate road impacts.
        5. Segments data into sliding windows.
        6. Extracts features for each window.

        Args:
            filepath (str): Path to the CSV file.
            label (str, optional): The class label (e.g., 'Pothole', 'Speedbreaker', 'Normal'). 
                                   If None, used for inference.

        Returns:
            list: A list of feature dictionaries (with labels if provided).
        """
        try:
            df = pd.read_csv(filepath)
            
            # Basic validation
            required_cols = ['Time (s)', 'Absolute acceleration (m/s^2)']
            for col in required_cols:
                if col not in df.columns:
                    # Fallback: compute absolute acceleration from individual axes if the column is missing
                    if col == 'Absolute acceleration (m/s^2)':
                        axis_cols = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
                        if all(c in df.columns for c in axis_cols):
                            df['Absolute acceleration (m/s^2)'] = np.sqrt(
                                df['Acceleration x (m/s^2)']**2 + 
                                df['Acceleration y (m/s^2)']**2 + 
                                df['Acceleration z (m/s^2)']**2
                            )
                        else:
                            print(f"Skipping {filepath}: Missing columns for absolute acceleration")
                            return []
                    else:
                        print(f"Skipping {filepath}: Missing column {col}")
                        return []
                    
            df = df.dropna(subset=['Time (s)', 'Absolute acceleration (m/s^2)'])
            if len(df) < 10:
                return []
                
            fs = self._calculate_sampling_rate(df['Time (s)'])
            
            # Apply Bandpass Filter to Absolute Acceleration
            # This single step eliminates BOTH engine noise AND braking/acceleration sway!
            df['abs_accel_filtered'] = self._butter_bandpass_filter(
                df['Absolute acceleration (m/s^2)'].values, fs
            )

            window_samples = int(self.window_size_sec * fs)
            step_samples = int(window_samples * (1.0 - self.overlap_pct))
            
            if window_samples <= 0 or step_samples <= 0:
                return []

            extracted_data = []
            
            for start in range(0, len(df) - window_samples + 1, step_samples):
                end = start + window_samples
                window_df = df.iloc[start:end]
                
                # Extract features
                features = self.extract_features(window_df, fs)
                
                # Add metadata
                features['start_time'] = window_df['Time (s)'].iloc[0]
                features['end_time'] = window_df['Time (s)'].iloc[-1]
                
                if label is not None:
                    features['label'] = label
                    
                extracted_data.append(features)
                
            return extracted_data
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return []

    def load_and_process_training_data(self):
        """
        Loads data from 'Potholes' and 'Speedbreakers' directories, as well as segments from 
        'Raw continuous data' that don't have anomalies (Normal).
        
        Returns:
            pd.DataFrame: A dataframe containing all extracted features and their labels.
        """
        all_features = []
        
        # Paths
        potholes_dir = os.path.join(self.data_dir, 'Potholes')
        speedbreakers_dir = os.path.join(self.data_dir, 'Speedbreakers')
        
        # Process Potholes
        if os.path.exists(potholes_dir):
            print(f"Processing Potholes...")
            for root, dirs, files in os.walk(potholes_dir):
                for file in files:
                    if file.endswith('.csv') and 'Raw Data' in file:
                        filepath = os.path.join(root, file)
                        file_features = self.process_file(filepath, label='Pothole')
                        if file_features:
                            # Only train on the top 5 windows that contain the actual physical bump.
                            # Sort by Peak-to-Peak (ptp) of the bandpass-filtered absolute acceleration.
                            file_features.sort(key=lambda x: x.get('abs_ptp', 0), reverse=True)
                            all_features.extend(file_features[:5])
                        
        # Process Speedbreakers
        if os.path.exists(speedbreakers_dir):
            print(f"Processing Speedbreakers...")
            for root, dirs, files in os.walk(speedbreakers_dir):
                for file in files:
                    if file.endswith('.csv') and 'Raw Data' in file:
                        filepath = os.path.join(root, file)
                        file_features = self.process_file(filepath, label='Speedbreaker')
                        if file_features:
                            # Only train on the top 5 windows that contain the actual physical bump.
                            file_features.sort(key=lambda x: x.get('abs_ptp', 0), reverse=True)
                            all_features.extend(file_features[:5])
                        
        # Extract Normal road data from continuous rides.
        # Only keep windows where the bandpass-filtered signal is very quiet (flat road).
        continuous_dir = os.path.join(self.data_dir, 'Raw continuous data')
        if os.path.exists(continuous_dir):
            print(f"Processing Normal (from continuous data)...")
            normal_count = 0
            for root, dirs, files in os.walk(continuous_dir):
                for file in files:
                    if file.endswith('.csv') and 'Raw Data' in file:
                        filepath = os.path.join(root, file)
                        continuous_features = self.process_file(filepath, label='Normal')
                        # Only extract "Normal" windows where the bandpass-filtered signal is relatively flat.
                        # UPGRADED: Raised threshold from 1.0 to 2.0 to capture more diverse Normal examples.
                        # This teaches the AI that slight vibrations during cruising are still "Normal".
                        # Braking/acceleration sway has already been removed by the 2 Hz high-pass!
                        if continuous_features:
                            smooth_windows = [f for f in continuous_features if f.get('abs_ptp', 100) < 2.0]
                            # UPGRADED: Sample every 2nd window instead of every 5th for more training volume.
                            sample_normal = smooth_windows[::2] 
                            all_features.extend(sample_normal)
                            normal_count += len(sample_normal)
                        
                        # Just process one continuous file for training Normal class to save time
                        if normal_count > 500:
                            break
                if normal_count > 500:
                    break

        df_features = pd.DataFrame(all_features)
        print(f"Total feature windows extracted: {len(df_features)}")
        if not df_features.empty:
            print(f"Class distribution:\n{df_features['label'].value_counts()}")
        
        return df_features
