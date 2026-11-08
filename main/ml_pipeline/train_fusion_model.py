"""
RoadSense — High-Precision ML Sensor Fusion Model Trainer (v2)
==============================================================
Trains an advanced ensemble classifier on camera-verified sensor fusion data.
Features:
- Segment-aware time-series signal processing (prevents NaN propagation across rides).
- Orientation-invariant dynamic acceleration: a_dyn = |sqrt(ax^2+ay^2+az^2) - 9.81|.
- Butterworth bandpass filtering (2 - 16 Hz) isolating road impact transients.
- Physics-based feature extraction:
  * Jerk (impulse rate of change da/dt)
  * Welch spectral band energy ratios (6-14Hz unsprung mass resonance vs 1-5Hz body sway)
  * Waveform crest factor, kurtosis, skewness, peak-to-peak amplitude
  * Gyroscopic pitch rates
- High-efficiency Gradient Boosting + Random Forest classification with Stratified K-Fold CV.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

# ============ CONFIG ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'cv_pipeline', 'master_training_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'ml_pipeline', 'road_health_rf_model_v2.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'ml_pipeline', 'road_health_rf_model_v2_features.pkl')

WINDOW_SEC = 0.40       # 400ms window captures the 100-250ms pothole shock transient
OVERLAP_PCT = 0.50      # 50% overlap for continuous temporal coverage
LOW_CUTOFF = 2.0        # Hz - removes gravity and braking sway
HIGH_CUTOFF = 16.0      # Hz - removes engine RPM vibration
FILTER_ORDER = 3


def bandpass_filter(data, fs, low=LOW_CUTOFF, high=HIGH_CUTOFF, order=FILTER_ORDER):
    """Zero-phase Butterworth bandpass filter isolating physical road impacts."""
    nyquist = 0.5 * fs
    low_n = max(low / nyquist, 0.01)
    high_n = min(high / nyquist, 0.99)
    if low_n >= high_n:
        return data
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, data)


def extract_window_features(sig_accel, sig_gyro, fs):
    """Extract physics-based time-domain and frequency-domain features from a window."""
    rms_a = np.sqrt(np.mean(sig_accel**2))
    p2p_a = np.ptp(sig_accel)
    std_a = np.std(sig_accel)
    max_a = np.max(np.abs(sig_accel))
    kurt_a = float(kurtosis(sig_accel, bias=False))
    skew_a = float(skew(sig_accel, bias=False))
    crest_a = max_a / (rms_a + 1e-6)
    
    # Jerk: numerical derivative of acceleration (detects sudden rim impacts)
    if len(sig_accel) > 1:
        jerk_a = float(np.ptp(np.diff(sig_accel))) * fs
    else:
        jerk_a = 0.0

    # Gyroscope dynamics
    rms_g = np.sqrt(np.mean(sig_gyro**2))
    p2p_g = np.ptp(sig_gyro)

    # Frequency Domain (Welch Power Spectral Density)
    try:
        nperseg = min(len(sig_accel), 32)
        if nperseg < 4:
            nperseg = len(sig_accel)
        f, pxx = welch(sig_accel, fs, nperseg=nperseg)
        low_band = float(np.sum(pxx[(f >= 1.0) & (f < 5.0)]))
        mid_band = float(np.sum(pxx[(f >= 5.0) & (f <= 15.0)]))
        high_band = float(np.sum(pxx[(f > 15.0)]))
        band_ratio = mid_band / (low_band + 1e-6)
        dom_freq = float(f[np.argmax(pxx)])
    except Exception:
        low_band = 0.0
        mid_band = 0.0
        high_band = 0.0
        band_ratio = 1.0
        dom_freq = 0.0

    return {
        'accel_rms': rms_a,
        'accel_p2p': p2p_a,
        'accel_std': std_a,
        'accel_max': max_a,
        'accel_kurt': kurt_a,
        'accel_skew': skew_a,
        'accel_crest': crest_a,
        'accel_jerk': jerk_a,
        'gyro_rms': rms_g,
        'gyro_p2p': p2p_g,
        'spectral_low_e': low_band,
        'spectral_mid_e': mid_band,
        'spectral_high_e': high_band,
        'spectral_band_ratio': band_ratio,
        'spectral_dom_freq': dom_freq
    }


def train_pipeline():
    print("=" * 65)
    print("ROADSENSE ML TRAINER v2 — High-Precision Camera-Verified Fusion")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Master training data not found at: {DATA_PATH}")
        return

    # 1. Load Data
    print("\n[1/5] Loading master training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Total records: {len(df)}")

    # Keep rows with valid vertical acceleration
    valid = df[df['accel_z'].notnull()].copy()
    print(f"  Valid sensor rows: {len(valid)}")

    # 2. Segment Data by Continuous Recording Runs
    # Negative time jumps or gaps > 2 seconds indicate a separate ride session
    time_diff = np.diff(valid['time_s'].values)
    split_indices = np.where((time_diff < -0.8) | (time_diff > 2.0))[0] + 1
    split_indices = np.concatenate([[0], split_indices, [len(valid)]])
    print(f"  Identified {len(split_indices)-1} distinct ride recording sessions")

    # 3. Extract Features Across Sessions
    print("\n[2/5] Running segment-aware signal processing & feature extraction...")
    all_windows = []

    for idx in range(len(split_indices) - 1):
        seg = valid.iloc[split_indices[idx]:split_indices[idx+1]].copy()
        if len(seg) < 25:
            continue

        # Dynamic sampling rate for this session
        t_vals = seg['time_s'].values
        dt = np.median(np.diff(t_vals))
        fs = 1.0 / dt if (dt and 0.003 < dt < 0.25) else 50.0

        # Orientation-Invariant Dynamic Acceleration
        has_3axis = seg['accel_x'].notnull().sum() > (len(seg) * 0.7)
        if has_3axis:
            ax = seg['accel_x'].fillna(0).values
            ay = seg['accel_y'].fillna(0).values
            az = seg['accel_z'].fillna(0).values
            mag = np.sqrt(ax**2 + ay**2 + az**2)
            dyn_acc = np.abs(mag - 9.81)
        else:
            az = seg['accel_z'].values
            dyn_acc = np.abs(az - np.median(az))

        # Gyroscope magnitude
        has_3gyro = seg['gyro_alpha'].notnull().sum() > (len(seg) * 0.7)
        if has_3gyro:
            gx = seg['gyro_alpha'].fillna(0).values
            gy = seg['gyro_beta'].fillna(0).values
            gz = seg['gyro_gamma'].fillna(0).values
            gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        elif 'gyro_y' in seg.columns and seg['gyro_y'].notnull().sum() > (len(seg) * 0.4):
            gyro_mag = np.abs(seg['gyro_y'].fillna(0).values)
        else:
            gyro_mag = np.zeros_like(dyn_acc)

        # Adaptive Rolling Baseline & Dynamic Z-Score
        roll_s = pd.Series(dyn_acc)
        w_size = max(5, int(10.0 * fs))
        roll_m = roll_s.rolling(w_size, min_periods=max(3, int(fs)), center=True).mean().bfill().ffill()
        roll_std = roll_s.rolling(w_size, min_periods=max(3, int(fs)), center=True).std().bfill().ffill().replace(0, 1.0)
        z_score = ((roll_s - roll_m) / roll_std).values

        # Strict physical peak gating: Camera-verified pothole (Label == 1 AND z >= 1.5) OR severe impact (z >= 2.8)
        raw_label = seg['Label'].values
        true_hazard = (((raw_label == 1) & (z_score >= 1.5)) | (z_score >= 2.8)).astype(int)
        seg['true_hazard'] = true_hazard

        # Zero-phase Bandpass Filter
        try:
            filt_a = bandpass_filter(dyn_acc, fs)
            filt_g = bandpass_filter(gyro_mag, fs)
        except Exception:
            filt_a = dyn_acc
            filt_g = gyro_mag

        seg['filt_a'] = filt_a
        seg['filt_g'] = filt_g

        # Sliding Window
        win_samples = max(8, int(WINDOW_SEC * fs))
        step_samples = max(4, int(win_samples * (1.0 - OVERLAP_PCT)))

        for w_start in range(0, len(seg) - win_samples, step_samples):
            w_df = seg.iloc[w_start:w_start + win_samples]
            wa = w_df['filt_a'].values
            wg = w_df['filt_g'].values
            
            # Ground truth: window contains a verified physical pothole impact peak
            lbl = 1 if w_df['true_hazard'].max() == 1 else 0

            feats = extract_window_features(wa, wg, fs)
            feats['label'] = lbl
            all_windows.append(feats)

    feature_df = pd.DataFrame(all_windows).replace([np.inf, -np.inf], 0).fillna(0)
    print(f"  Extracted windows: {len(feature_df)}")
    pothole_cnt = (feature_df['label'] == 1).sum()
    normal_cnt = (feature_df['label'] == 0).sum()
    print(f"  Distribution: {normal_cnt} Normal Road ({normal_cnt/len(feature_df)*100:.1f}%), {pothole_cnt} Potholes ({pothole_cnt/len(feature_df)*100:.1f}%)")

    # 4. Train Model
    print("\n[3/5] Evaluating & Training Ensemble Classifier...")
    X = feature_df.drop('label', axis=1)
    y = feature_df['label']
    feature_names = list(X.columns)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hgb = HistGradientBoostingClassifier(
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        class_weight='balanced',
        random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[('hgb', hgb), ('rf', rf)],
        voting='soft'
    )

    preds = cross_val_predict(ensemble, X, y, cv=skf)
    proba = cross_val_predict(ensemble, X, y, cv=skf, method='predict_proba')[:, 1]

    acc = accuracy_score(y, preds)
    roc = roc_auc_score(y, proba)
    f1_w = f1_score(y, preds, average='weighted')
    f1_m = f1_score(y, preds, average='macro')

    print("\n" + "-" * 55)
    print(f"  5-FOLD CROSS VALIDATION METRICS:")
    print(f"  • Overall Accuracy: {acc * 100:.2f}%")
    print(f"  • ROC-AUC Score:    {roc:.4f}")
    print(f"  • Weighted F1:      {f1_w:.4f}")
    print(f"  • Macro F1:         {f1_m:.4f}")
    print("-" * 55)

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y, preds)
    print(f"    [[TN: {cm[0,0]:4d},  FP: {cm[0,1]:4d}],")
    print(f"     [FN: {cm[1,0]:4d},  TP: {cm[1,1]:4d}]]")

    print("\n  Detailed Classification Report:")
    print(classification_report(y, preds, target_names=['Normal Road', 'Pothole (Anomaly)']))

    # 5. Fit Final Ensemble on Full Dataset & Save
    print("[4/5] Fitting final production ensemble on full dataset...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[('hgb', hgb), ('rf', rf)],
        voting='soft'
    )
    ensemble.fit(X, y)

    # Feature Importance inspection from Random Forest component
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\n  Top Feature Importances:")
    for feat, imp in importances.head(8).items():
        print(f"    {feat:22s} : {imp:.4f}")

    # Save
    print("\n[5/5] Saving model & feature metadata...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(ensemble, MODEL_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print(f"  Model successfully saved to: {MODEL_PATH}")
    print(f"  Feature list saved to:       {FEATURES_PATH}")

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE — MODEL READY FOR REAL-WORLD DEPLOYMENT")
    print("=" * 65)


if __name__ == '__main__':
    train_pipeline()
