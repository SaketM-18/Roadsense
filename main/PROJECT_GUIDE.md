# 🛣️ Road Health Detection System — Complete Project Guide

> **Project Goal:** Build a smartphone-based system that detects and classifies road surface conditions (potholes, speed breakers, normal road) in real-time using onboard sensors and machine learning.

---

## Table of Contents

1. [Approach Evaluation — Is Accelerometer-Only the Best Way?](#1-approach-evaluation)
2. [Recommended Multi-Sensor Approach](#2-recommended-multi-sensor-approach)
3. [System Architecture](#3-system-architecture)
4. [Phase 1 — Data Collection with phyphox](#4-phase-1--data-collection-with-phyphox)
5. [Phase 2 — Data Labeling with Web Tool](#5-phase-2--data-labeling-with-web-tool)
6. [Phase 3 — Data Preprocessing & Feature Engineering](#6-phase-3--data-preprocessing--feature-engineering)
7. [Phase 4 — Model Training](#7-phase-4--model-training)
8. [Phase 5 — Deployment & Real-Time Classification](#8-phase-5--deployment--real-time-classification)
9. [Phase 6 — Visualization Dashboard](#9-phase-6--visualization-dashboard)
10. [Tech Stack & Tools](#10-tech-stack--tools)
11. [Hardware & Software Requirements](#11-hardware--software-requirements)
12. [Dataset Structure](#12-dataset-structure)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Challenges & Mitigations](#14-challenges--mitigations)
15. [Timeline & Milestones](#15-timeline--milestones)
16. [References & Further Reading](#16-references--further-reading)

---

## 1. Approach Evaluation

### Your Idea: Accelerometer-Only

Using the accelerometer alone **will work**, but it has significant limitations:

| Aspect | Accelerometer Only | Multi-Sensor (Recommended) |
|---|---|---|
| Pothole detection | ✅ Good (Z-axis spikes) | ✅ Excellent |
| Speed breaker detection | ⚠️ Moderate (similar Z-pattern to potholes) | ✅ Excellent (gyroscope distinguishes roll/pitch) |
| Normal road classification | ✅ Good (low variance) | ✅ Excellent |
| False positives from turns/braking | ❌ High | ✅ Low (gyroscope filters these out) |
| Location mapping | ❌ Not possible | ✅ GPS provides coordinates |
| Speed-aware thresholding | ❌ Not possible | ✅ GPS speed normalizes readings |
| Vehicle type compensation | ❌ Not possible | ⚠️ Partially (with calibration) |

### Verdict

> **Accelerometer alone is a good starting point, but NOT the best approach.**
> The best approach uses **Accelerometer + Gyroscope + GPS** together. This is still smartphone-only — no external hardware needed.

### Alternative & Complementary Approaches

| Method | Pros | Cons | Feasibility |
|---|---|---|---|
| **Multi-sensor fusion (Accel + Gyro + GPS)** | Best accuracy, filters noise, provides location | Slightly more complex | ⭐ **Recommended** |
| **Camera-based (Computer Vision)** | Can detect visual road damage | High battery drain, compute-heavy, privacy concerns | ⚠️ Supplementary only |
| **Audio-based (Microphone)** | Different sound signatures for potholes | Very noisy, hard to isolate | ❌ Not reliable |
| **Crowdsourced data aggregation** | Reduces per-user false positives | Needs many users | ⭐ Good for Phase 2 |
| **Federated Learning** | Privacy-preserving model improvement | Complex infrastructure | ⭐ Advanced / future work |

---

## 2. Recommended Multi-Sensor Approach

### Sensors to Use (All built into smartphones)

```
┌──────────────────────────────────────────────────────────┐
│                    SMARTPHONE SENSORS                     │
├──────────────┬───────────────┬───────────────────────────┤
│ Accelerometer│   Gyroscope   │           GPS             │
│ (Primary)    │ (Secondary)   │       (Location)          │
├──────────────┼───────────────┼───────────────────────────┤
│ Measures:    │ Measures:     │ Measures:                 │
│ • X,Y,Z      │ • Roll        │ • Latitude               │
│   linear     │ • Pitch       │ • Longitude               │
│   acceleration│ • Yaw        │ • Speed                   │
│              │               │ • Altitude                │
│ Detects:     │ Detects:      │ Provides:                 │
│ • Vibrations │ • Phone tilt  │ • Location tagging        │
│ • Bumps      │ • Vehicle     │ • Speed normalization     │
│ • Jolts      │   rotation    │ • Route mapping           │
└──────────────┴───────────────┴───────────────────────────┘
```

### Why Each Sensor Matters

1. **Accelerometer (X, Y, Z axes)**
   - **Z-axis** captures vertical bumps (potholes, speed breakers)
   - **X-axis** captures lateral forces (turns, lane changes)
   - **Y-axis** captures forward/braking forces
   - This is the **core sensor** for anomaly detection

2. **Gyroscope (Roll, Pitch, Yaw)**
   - Differentiates a pothole hit from a sharp turn
   - Speed breakers produce a **gradual pitch change** vs. potholes which cause a **sudden jolt**
   - Filters out false positives caused by driver behavior

3. **GPS (Lat, Lng, Speed, Altitude)**
   - Tags each detection with precise coordinates
   - Speed data normalizes vibrations (same bump feels different at 20 km/h vs. 60 km/h)
   - Enables heatmap visualization of road quality

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM OVERVIEW                          │
└─────────────────────────────────────────────────────────────────┘

  ┌───────────┐     ┌──────────────┐     ┌──────────────────────┐
  │  PHYPHOX  │────▶│ WEB LABELING │────▶│   DATA PIPELINE      │
  │  (Phone)  │     │   TOOL       │     │                      │
  │           │     │  (Browser)   │     │• Clean & filter      │
  │• Record   │     │              │     │• Window & extract    │
  │  accel +  │     │• Upload CSV  │     │  features            │
  │  gyro +   │     │• View graphs │     │                      │
  │  GPS data │     │• Click to    │     │                      │
  │• Export   │     │  label spikes│     │                      │
  │  CSV      │     │• Export      │     │                      │
  │           │     │  labeled CSV │     │                      │
  └───────────┘     └──────────────┘     └──────────┬───────────┘
                                                     │
                                                     ▼
  ┌───────────┐     ┌──────────────┐     ┌──────────────────────┐
  │ DASHBOARD │◀────│   BACKEND    │◀────│   ML MODEL TRAINING  │
  │           │     │   (API)      │     │                      │
  │• Heatmap  │     │• Store       │     │• Train classifiers   │
  │• Stats    │     │  results     │     │• Evaluate & compare  │
  │• Reports  │     │• Serve API   │     │• Export best model   │
  │           │     │• Aggregate   │     │  (TFLite for phone)  │
  └───────────┘     └──────────────┘     └──────────────────────┘
```

---

## 4. Phase 1 — Data Collection with phyphox

### Why NOT Build an Android App for Data Collection?

| | Build Custom Android App | Use phyphox (Free App) |
|---|---|---|
| **Development time** | ❌ Weeks (Kotlin, Android Studio) | ✅ Zero — just install |
| **Sensor access** | ✅ Full control | ✅ Full control (accel, gyro, GPS) |
| **Data export** | ✅ Custom CSV | ✅ CSV / Excel export built-in |
| **Labeling while driving** | ⚠️ Dangerous, inaccurate (reaction delay) | ✅ Not needed — label AFTER with web tool |
| **Reliability** | ⚠️ Your app may have bugs | ✅ Battle-tested, used in universities |
| **Focus** | ❌ Waste time on app dev | ✅ Focus on ML, the actual project |

> **Verdict:** Use **phyphox** for data collection. Build a custom Android app only later in Phase 5 when you need **real-time inference** with the trained model.

### What is phyphox?

**phyphox** (Physical Phone Experiments) is a **free, open-source app** developed by **RWTH Aachen University** (Germany). It gives you direct access to all phone sensors with CSV export.

- **Download:** [Google Play Store](https://play.google.com/store/apps/details?id=de.rwth_aachen.phyphox) or [Apple App Store](https://apps.apple.com/app/phyphox/id1127319693)
- **Website:** [phyphox.org](https://phyphox.org)
- **Cost:** Completely free, no ads
- **Works on:** Android 4.0+ and iOS 9.0+

### Step-by-Step: Setting Up phyphox for Data Collection

#### Step 1: Install & Open phyphox

```
1. Install "phyphox" from Play Store / App Store
2. Open the app → you'll see a list of sensor experiments
3. For our project, we need TWO separate recordings:
   • "Accelerometer" (under Raw Sensors)
   • "Gyroscope" (under Raw Sensors)
   OR use the "Multi" experiment to record both simultaneously
```

#### Step 2: Configure the Experiment

```
1. Tap on "Acceleration with g" (includes gravity component)
   OR "Acceleration without g" (gravity removed — recommended)
2. Tap the ⚙️ (gear) icon → Settings:
   • Sensor rate: Set to "Fastest" or "Game" (~50-100 Hz)
   • This gives you 50-100 readings per second
3. For Gyroscope: Same settings
```

#### Step 3: Record GPS Alongside (Use a Second Method)

phyphox records accelerometer and gyroscope, but GPS needs a separate approach:

| GPS Method | How | Convenience |
|---|---|---|
| **Google Maps Timeline** | Just keep Google Maps running — it auto-logs your route with timestamps | ⭐ Easiest |
| **GPS Logger App** | Install "GPS Logger" (free on Play Store) — records lat/lng/speed to CSV | ⭐ Best for this project |
| **Strava / Map My Ride** | Any fitness tracking app that exports GPX files | ✅ Good alternative |
| **phyphox Location** | phyphox has a "Location" experiment under Raw Sensors | ✅ Built-in but limited rate |

> **Recommended combo:** phyphox (accel + gyro) + GPS Logger app (location + speed), both running simultaneously.

#### Step 4: Dashcam Video (Optional but Very Helpful)

```
Why record video?
├── When you label data AFTER the drive, you may forget what happened
├── Video lets you VISUALLY CONFIRM: "Was that spike a pothole or a speed breaker?"
├── Sync video timestamp with sensor timestamp
└── Use a second phone on the dashboard recording video

Free dashcam apps:
├── Android: "AutoBoy Dash Cam" (free on Play Store)
└── iOS: Built-in Camera app in video mode
```

#### Step 5: Export Data from phyphox

```
1. After your drive, open phyphox
2. Tap the ⋮ (three dots) menu → "Export"
3. Choose format: CSV (Comma Separated)
4. Save to phone storage or share via email/Google Drive
5. Transfer CSV files to your laptop
```

**Exported CSV will look like this:**

```csv
"Time (s)","Acceleration x (m/s^2)","Acceleration y (m/s^2)","Acceleration z (m/s^2)"
0.000,0.12,-0.34,9.81
0.020,0.15,-0.31,9.78
0.040,0.11,-0.35,9.83
...
```

### Alternative Apps (if phyphox doesn't work on your phone)

| App | Platform | Sensors | Export | Free? |
|---|---|---|---|---|
| **phyphox** ⭐ | Android/iOS | Accel, Gyro, Mag, Light, GPS | CSV, Excel | ✅ Yes |
| **Sensor Logger** | Android/iOS | All sensors | CSV, JSON | ✅ Yes |
| **Physics Toolbox Suite** | Android/iOS | All sensors + real-time graphs | CSV | ✅ Yes |
| **Sensors Multitool** | Android | All sensors | CSV | ✅ Yes |

### Data Collection Protocol

```
📋 DATA COLLECTION CHECKLIST — BEFORE THE DRIVE

□ Mount phone securely on dashboard (use a phone holder / sticky mount)
□ Keep phone orientation CONSISTENT across ALL sessions
   (e.g., always portrait mode, screen facing up)
□ Start phyphox recording (Accelerometer + Gyroscope)
□ Start GPS Logger recording
□ (Optional) Start dashcam video on a second phone
□ Note the START TIME so you can sync data later

📋 DURING THE DRIVE

□ Drive normally — don't avoid potholes (you need that data!)
□ Drive at varied speeds: 20, 30, 40, 60 km/h
□ DON'T press anything — just drive
□ (Optional) Make voice notes: "That was a pothole" / "Speed breaker coming"
□ Cover these conditions:
   □ Smooth highway (5-10 min)
   □ City roads with potholes (10-15 min)
   □ Roads with speed breakers (5-10 min)
   □ Rough/unpaved roads (5-10 min)

📋 AFTER THE DRIVE

□ Stop all recordings
□ Export CSVs from phyphox + GPS Logger
□ Transfer files to laptop
□ Rename files: session_001_accel.csv, session_001_gyro.csv, session_001_gps.csv
□ Open the Web Labeling Tool → label the data (see Phase 2)

📋 DATA TARGETS

□ Minimum 30 minutes per session
□ At least 5-10 sessions total
□ Aim for 500+ pothole events, 500+ speed breaker events
□ Collect from 2-3 different vehicles if possible
□ Collect from 5+ different roads/routes
```

### Merging Sensor + GPS Data

After collection, merge accelerometer, gyroscope, and GPS data using timestamps:

```python
import pandas as pd

# Load separate CSVs
accel = pd.read_csv('session_001_accel.csv')
gyro  = pd.read_csv('session_001_gyro.csv')
gps   = pd.read_csv('session_001_gps.csv')

# Rename columns for clarity
accel.columns = ['timestamp', 'accel_x', 'accel_y', 'accel_z']
gyro.columns  = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z']
gps.columns   = ['timestamp', 'latitude', 'longitude', 'speed', 'altitude']

# Merge on nearest timestamp (accel is highest frequency, so it's the base)
merged = pd.merge_asof(
    accel.sort_values('timestamp'),
    gyro.sort_values('timestamp'),
    on='timestamp',
    direction='nearest'
)
merged = pd.merge_asof(
    merged.sort_values('timestamp'),
    gps.sort_values('timestamp'),
    on='timestamp',
    direction='nearest'
)

merged.to_csv('session_001_merged.csv', index=False)
print(f"Merged dataset: {len(merged)} rows")
```

---

## 5. Phase 2 — Data Labeling with Web Tool

### Why Label AFTER Collection (Not During)?

| | Label During Drive | Label After Drive (Web Tool) |
|---|---|---|
| **Safety** | ❌ Dangerous — pressing buttons while driving | ✅ Safe — done at home on laptop |
| **Accuracy** | ⚠️ Low — human reaction delay ~0.5-1 sec | ✅ High — zoom into exact timestamps |
| **Corrections** | ❌ Can't undo a wrong label | ✅ Easy to relabel, delete, redo |
| **Speed** | ⚠️ Can miss events | ✅ Scan entire recording carefully |
| **Needs Android app?** | ❌ Yes — must build one | ✅ No — just a web page |

### How Can You Tell What's a Pothole vs Speed Breaker?

You do NOT need AI to label — **the sensor graph tells you visually:**

```
Accelerometer Z-axis (vertical acceleration)

NORMAL ROAD:
    ──────────────────────────────────    Flat, small noise
    ~9.8 m/s² (just gravity)              Easy to identify

POTHOLE (sudden hole in road):
                 ╱╲
    ────────────╱  ╲──────────────────    Sharp, NARROW spike
                    ╲╱                    Duration: < 0.3 sec
                                          Very sudden & violent

SPEED BREAKER (gradual hump):
              ╱‾‾‾‾‾‾╲
    ─────────╱        ╲──────────────    WIDE, smooth wave
                                          Duration: 1-2 sec
                                          Gradual rise & fall
```

### 3 Strategies to Confirm Your Labels

| Strategy | How | When to Use |
|---|---|---|
| **1. Spike shape** | Sharp & narrow = pothole, Wide & smooth = speed breaker | Always — this is the primary method |
| **2. Dashcam video** | Watch the video at the same timestamp to see what's on the road | When you're unsure about a spike |
| **3. GPS + Google Maps** | Check the location on Google Maps — speed breakers are at fixed spots (near schools, intersections) | For verifying speed breaker labels |

### The Web Labeling Tool

Build a simple **HTML/CSS/JavaScript web tool** (runs locally in your browser, no server needed) that provides:

#### Core Features

```
┌─────────────────────────────────────────────────────────────────────┐
│                     WEB LABELING TOOL FEATURES                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📂 UPLOAD   → Drag & drop your phyphox CSV file                   │
│                                                                     │
│  📊 GRAPH    → Interactive zoomable graph showing:                  │
│               • Accelerometer Z-axis (main line — shows bumps)     │
│               • Accelerometer X, Y (secondary — for context)        │
│               • Gyroscope pitch (helps distinguish pothole vs bump) │
│               • Acceleration magnitude (combined signal)            │
│                                                                     │
│  🏷️ LABEL    → Click on a spike → select label:                    │
│               • 🔴 Pothole                                          │
│               • 🟡 Speed Breaker                                    │
│               • 🟢 Normal (default — anything unlabeled)            │
│                                                                     │
│  ✂️ SELECT   → Click & drag to select a TIME RANGE for the label   │
│               (a pothole event spans ~0.3 sec, select that region)  │
│                                                                     │
│  🔍 ZOOM     → Scroll to zoom in/out on the time axis              │
│               → Zoom in to see individual samples                   │
│                                                                     │
│  🗑️ DELETE   → Right-click a label to remove it                    │
│                                                                     │
│  📥 EXPORT   → Download labeled CSV with a "label" column added    │
│                                                                     │
│  📊 STATS    → Show count: "42 potholes, 38 speed breakers labeled"│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Tech Stack for the Web Tool

| Component | Technology | Why |
|---|---|---|
| **Structure** | HTML5 | Simple, no framework needed |
| **Styling** | CSS3 (Vanilla) | Clean UI, dark theme for long labeling sessions |
| **Logic** | JavaScript (Vanilla) | No dependencies to install |
| **Charting** | [Chart.js](https://www.chartjs.org/) with `chartjs-plugin-zoom` | Interactive, zoomable graphs. Free. Load via CDN. |
| **CSV Parsing** | [Papa Parse](https://www.papaparse.com/) | Best CSV parser for JS. Free. Load via CDN. |
| **File Handling** | HTML5 File API (Drag & Drop) | Built into browsers, no server needed |

> **Key Point:** The entire tool is a single `index.html` file + `style.css` + `app.js`. You open it in Chrome/Edge — no server, no npm, no setup. Just double-click and use.

#### How to Get Chart.js and Papa Parse (No Install Needed)

Just add these CDN links in your HTML `<head>`:

```html
<!-- Papa Parse — CSV parser -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>

<!-- Chart.js — Interactive graphs -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

<!-- Chart.js Zoom Plugin — Scroll to zoom, pan the graph -->
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>

<!-- Hammer.js — Required by zoom plugin for touch/drag gestures -->
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
```

That's it — no `npm install`, no `node_modules`, no build step. Everything loads from CDN.

#### Web Tool Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    LABELING WORKFLOW                              │
└──────────────────────────────────────────────────────────────────┘

 Step 1:  Open labeling_tool/index.html in Chrome
               │
 Step 2:  Drag & drop your phyphox CSV file
               │
 Step 3:  Graph appears showing Z-axis acceleration over time
               │
               ▼
        ┌──────────────────────────────────────────┐
        │  📊 You see the full drive timeline       │
        │                                          │
        │  Normal road = flat line                 │
        │  Spikes = anomalies (potholes/bumps)     │
        │                                          │
        │  Zoom in on each spike to examine it     │
        └──────────────────────────────────────────┘
               │
 Step 4:  Click on a spike → pop-up asks: "Pothole" or "Speed Breaker"?
               │
               ├── Sharp narrow spike → select "Pothole"
               ├── Wide smooth wave  → select "Speed Breaker"
               └── Unsure? Check dashcam video at that timestamp
               │
 Step 5:  Repeat for all visible spikes (normal road auto-labeled)
               │
 Step 6:  Click "Export Labeled CSV"
               │
               ▼
        ┌──────────────────────────────────────────┐
        │  📥 Downloaded: session_001_labeled.csv   │
        │                                          │
        │  Now has a "label" column:               │
        │  normal / pothole / speed_breaker        │
        │                                          │
        │  Ready for ML pipeline! → Phase 3        │
        └──────────────────────────────────────────┘
```

#### How Convenient Should It Be?

The tool should feel as smooth as a professional annotation tool. Key UX requirements:

```
✅ MUST HAVE:
├── Drag & drop file upload (no file dialogs)
├── Instant graph rendering (< 2 sec for 30 min of data)
├── Smooth zoom with mouse scroll (zoom into spikes)
├── Pan by clicking and dragging the graph
├── Click on graph to place a label at that timestamp
├── Keyboard shortcuts: P = Pothole, S = Speed Breaker, D = Delete label
├── Undo/Redo (Ctrl+Z / Ctrl+Y)
├── Label counter showing progress (e.g., "42 potholes, 38 speed breakers")
├── One-click CSV export with labels
├── Dark theme (comfortable for long sessions)
└── Works offline (after first CDN load, or use local files)

⭐ NICE TO HAVE:
├── Auto-detect potential spikes and highlight them (threshold-based)
├── Side-by-side video player synced with the graph timeline
├── Batch label: select a region → label all spikes in it
├── Show GPS location on a mini-map for current graph position
└── Save/load labeling session (in case you want to continue later)
```

---

## 6. Phase 3 — Data Preprocessing & Feature Engineering

### Step 1: Load & Clean Raw Data

```python
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# Load CSV
df = pd.read_csv('sensor_data.csv')

# Columns expected:
# timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
# latitude, longitude, speed, label

# Remove gravity component from accelerometer (high-pass filter)
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=0.3, fs=50, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

df['accel_z_filtered'] = highpass_filter(df['accel_z'].values)
```

### Step 2: Windowing (Sliding Window)

Split continuous data into fixed-size windows for classification:

```python
WINDOW_SIZE = 128     # Number of samples per window (~2.5 sec at 50Hz)
OVERLAP = 64          # 50% overlap

def create_windows(data, window_size, overlap):
    windows = []
    step = window_size - overlap
    for i in range(0, len(data) - window_size, step):
        window = data.iloc[i:i + window_size]
        windows.append(window)
    return windows
```

### Step 3: Feature Extraction

For **each window**, extract these features from accelerometer and gyroscope axes:

```python
def extract_features(window):
    features = {}
    
    for axis in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        data = window[axis].values
        
        # Time-domain features
        features[f'{axis}_mean']     = np.mean(data)
        features[f'{axis}_std']      = np.std(data)
        features[f'{axis}_max']      = np.max(data)
        features[f'{axis}_min']      = np.min(data)
        features[f'{axis}_range']    = np.max(data) - np.min(data)
        features[f'{axis}_rms']      = np.sqrt(np.mean(data**2))
        features[f'{axis}_skew']     = pd.Series(data).skew()
        features[f'{axis}_kurtosis'] = pd.Series(data).kurtosis()
        features[f'{axis}_zcr']      = np.sum(np.diff(np.sign(data)) != 0)  # Zero-crossing rate
        features[f'{axis}_energy']   = np.sum(data**2)
        
        # Frequency-domain features (FFT)
        fft_vals = np.abs(np.fft.rfft(data))
        features[f'{axis}_fft_mean']   = np.mean(fft_vals)
        features[f'{axis}_fft_max']    = np.max(fft_vals)
        features[f'{axis}_fft_energy'] = np.sum(fft_vals**2)
    
    # Combined features
    accel_mag = np.sqrt(window['accel_x']**2 + window['accel_y']**2 + window['accel_z']**2)
    features['accel_magnitude_mean'] = np.mean(accel_mag)
    features['accel_magnitude_std']  = np.std(accel_mag)
    features['accel_magnitude_max']  = np.max(accel_mag)
    
    # Speed feature (if available)
    if 'speed' in window.columns:
        features['speed_mean'] = np.mean(window['speed'])
    
    return features
```

### Feature Summary

| Category | Features | Count |
|---|---|---|
| **Time-domain** (per axis × 6 axes) | mean, std, max, min, range, RMS, skewness, kurtosis, ZCR, energy | 60 |
| **Frequency-domain** (per axis × 6 axes) | FFT mean, FFT max, FFT energy | 18 |
| **Combined** | Acceleration magnitude (mean, std, max), speed | 4 |
| **Total** | | **~82 features** |

---

## 7. Phase 4 — Model Training

### Classification Labels

```
┌─────────────────────────────────────────┐
│           CLASS LABELS                  │
├─────────────────────────────────────────┤
│  0  │  Normal Road    (smooth driving)  │
│  1  │  Pothole        (sudden jolt)     │
│  2  │  Speed Breaker  (gradual bump)    │
└─────────────────────────────────────────┘
```

### Model Options (from Simple to Complex)

#### Model 1: Random Forest (Start Here ⭐)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Prepare data
X = feature_dataframe.drop('label', axis=1)
y = feature_dataframe['label']

# Handle class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

#### Model 2: XGBoost (Better accuracy)

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)
```

#### Model 3: 1D-CNN (Deep Learning — Best for raw signals)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Reshape windowed data: (samples, timesteps, channels)
# channels = 6 (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
X_train_cnn = X_train_raw.reshape(-1, WINDOW_SIZE, 6)

model = models.Sequential([
    layers.Conv1D(64, 5, activation='relu', input_shape=(WINDOW_SIZE, 6)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(128, 5, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(128, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

#### Model 4: LSTM (Best for temporal patterns)

```python
model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 6)),
    layers.Dropout(0.3),
    layers.LSTM(32),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])
```

### Model Comparison Expected Results

| Model | Expected Accuracy | Training Time | On-Device Inference |
|---|---|---|---|
| Random Forest | 85–90% | Fast (minutes) | ❌ Hard to deploy |
| XGBoost | 88–93% | Fast (minutes) | ⚠️ Moderate |
| 1D-CNN | 92–96% | Moderate (hours) | ✅ TFLite ready |
| LSTM | 93–97% | Slow (hours) | ✅ TFLite ready |
| CNN + LSTM Hybrid | 95–99% | Slow (hours) | ✅ TFLite ready |

> **Recommendation:** Start with Random Forest for quick validation, then move to 1D-CNN or LSTM for deployment.

---

## 8. Phase 5 — Deployment & Real-Time Classification

### Convert Model to TensorFlow Lite

```python
# Convert the trained model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantize for smaller size
tflite_model = converter.convert()

# Save
with open('road_health_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Model size should be < 5 MB for smooth on-device inference
print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
```

### Android Integration (TFLite)

```kotlin
// In Android app — Load and run TFLite model
val interpreter = Interpreter(loadModelFile("road_health_model.tflite"))

// Create input buffer from sensor window
val inputBuffer = ByteBuffer.allocateDirect(WINDOW_SIZE * 6 * 4) // float32
inputBuffer.order(ByteOrder.nativeOrder())

// Fill buffer with latest sensor window data
for (sample in sensorWindow) {
    inputBuffer.putFloat(sample.accelX)
    inputBuffer.putFloat(sample.accelY)
    inputBuffer.putFloat(sample.accelZ)
    inputBuffer.putFloat(sample.gyroX)
    inputBuffer.putFloat(sample.gyroY)
    inputBuffer.putFloat(sample.gyroZ)
}

// Run inference
val outputBuffer = Array(1) { FloatArray(3) } // 3 classes
interpreter.run(inputBuffer, outputBuffer)

// Result: outputBuffer[0] = [normalProb, potholeProb, speedBreakerProb]
val predictedClass = outputBuffer[0].indices.maxByOrNull { outputBuffer[0][it] }
```

### Real-Time Pipeline on Phone

```
Sensor Data (50Hz)
       │
       ▼
┌─────────────┐
│ Ring Buffer  │  ← Stores last 128 samples (~2.5 sec)
│ (128 samples)│
└──────┬──────┘
       │ Every 64 samples (1.3 sec)
       ▼
┌─────────────┐
│ Preprocess   │  ← Normalize, filter
│ Window       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ TFLite       │  ← Run inference (< 50ms)
│ Inference    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Result +     │  ← Tag with GPS coordinates
│ GPS Location │
└──────┬──────┘
       │
       ├──▶ Alert User (vibration/sound)
       └──▶ Send to Backend (API)
```

---

## 9. Phase 6 — Visualization Dashboard

### Web Dashboard Features

Build a web dashboard to visualize detected road anomalies:

- **Interactive Map** — Show potholes and speed breakers on a map (use Leaflet.js or Google Maps API)
- **Heatmap Layer** — Color-coded road quality (green = good, yellow = moderate, red = poor)
- **Statistics** — Total detections, per-road breakdown, trends over time
- **Admin Panel** — Verify/reject crowd-sourced detections

### Backend API (Flask/FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Detection(BaseModel):
    latitude: float
    longitude: float
    classification: str   # "pothole", "speed_breaker", "normal"
    confidence: float
    speed: float
    timestamp: str

@app.post("/api/report")
async def report_detection(detection: Detection):
    # Save to database
    save_to_db(detection)
    return {"status": "received"}

@app.get("/api/detections")
async def get_detections(lat: float, lng: float, radius: float = 1.0):
    # Return detections within radius (km)
    results = query_nearby(lat, lng, radius)
    return results
```

---

## 10. Tech Stack & Tools

### Development Stack

| Component | Technology | Purpose |
|---|---|---|
| **Data Collection** | phyphox + GPS Logger (free phone apps) | Record sensor data during drives |
| **Data Labeling** | Custom Web Tool (HTML/CSS/JS + Chart.js) | Visually label sensor spikes after collection |
| **Data Processing** | Python, Pandas, NumPy, SciPy | Clean & preprocess data |
| **Visualization** | Matplotlib, Seaborn, Plotly | Explore data patterns |
| **ML — Classical** | Scikit-learn, XGBoost | Random Forest, SVM, XGBoost |
| **ML — Deep Learning** | TensorFlow / Keras, or PyTorch | CNN, LSTM models |
| **Model Deployment** | TensorFlow Lite | On-device inference |
| **Backend API** | FastAPI or Flask | Receive & serve detections |
| **Database** | PostgreSQL + PostGIS, or Firebase | Store geo-tagged detections |
| **Web Dashboard** | HTML/CSS/JS + Leaflet.js | Visualize map & stats |
| **Version Control** | Git + GitHub | Code management |
| **Notebooks** | Jupyter Notebook / Google Colab | Experimentation & training |

---

## 11. Hardware & Software Requirements

### Hardware

```
MINIMUM REQUIREMENTS:
├── Android/iOS Smartphone with:
│   ├── Accelerometer    ✅ (available on ALL smartphones)
│   ├── Gyroscope        ✅ (most phones from 2016+)
│   └── GPS              ✅ (available on ALL smartphones)
│
├── Car/Bike + Phone Mount (dashboard/handlebar)
│   └── ₹200-500 phone holder from Amazon/local shop
│
├── (Optional) Second phone for dashcam video
│
├── PC/Laptop for Labeling + ML Training:
│   ├── RAM: 8 GB minimum (16 GB recommended)
│   ├── Storage: 10 GB free space
│   ├── GPU: Optional (NVIDIA recommended for deep learning)
│   ├── CPU: Any modern processor
│   └── Browser: Chrome or Edge (for web labeling tool)
│
└── (Optional) Google Colab — Free GPU for training
```

### Software

```
PHONE APPS (Free, no development needed):
├── phyphox (Play Store / App Store) — Sensor recording
├── GPS Logger (Play Store) — Location + speed recording
└── (Optional) AutoBoy Dash Cam — Video recording

LAPTOP SOFTWARE:
├── Python 3.9+ (Anaconda recommended)
├── Any modern browser (Chrome / Edge / Firefox)
│
├── Python Libraries:
│   pip install numpy pandas scipy matplotlib seaborn
│   pip install scikit-learn xgboost imbalanced-learn
│   pip install tensorflow                       # For deep learning
│   pip install fastapi uvicorn                  # For backend
│   pip install folium                           # For map visualization
│
├── Web Labeling Tool (no install — just open in browser):
│   └── Uses Chart.js + Papa Parse via CDN links
│
├── (Later, for deployment only) Android Studio:
│   └── Only needed in Phase 5 when integrating TFLite model
│   └── Android Dependencies:
│       implementation 'org.tensorflow:tensorflow-lite:2.14.0'
│       implementation 'com.google.android.gms:play-services-location:21.0.1'
│
└── Optional:
    pip install jupyter notebook
    pip install plotly dash                       # Interactive dashboards
```

---

## 12. Dataset Structure

### Raw Data Format (CSV)

```csv
timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,latitude,longitude,speed,label
1686812400000,0.12,-0.34,9.81,0.01,-0.02,0.00,12.9716,77.5946,35.2,normal
1686812400020,0.15,-0.31,9.78,0.01,-0.01,0.01,12.9716,77.5946,35.1,normal
1686812400040,2.45,-1.22,14.56,0.85,-0.92,0.12,12.9717,77.5947,34.8,pothole
...
```

### Folder Structure

```
Road Health/
│
├── PROJECT_GUIDE.md              ← This file
│
├── data/
│   ├── raw/                      ← Raw CSV files from phyphox + GPS Logger
│   │   ├── session_001_accel.csv
│   │   ├── session_001_gyro.csv
│   │   ├── session_001_gps.csv
│   │   ├── session_001_merged.csv   ← After Python merge script
│   │   └── ...
│   ├── labeled/                  ← Output from Web Labeling Tool
│   │   ├── session_001_labeled.csv
│   │   ├── session_002_labeled.csv
│   │   └── ...
│   ├── processed/                ← Cleaned & windowed data
│   │   ├── features.csv
│   │   └── windows.npy
│   └── labels/                   ← Label mapping
│       └── class_mapping.json
│
├── labeling_tool/                ← Web Labeling Tool (open in browser)
│   ├── index.html                ← Main page — drag & drop CSV here
│   ├── style.css                 ← Dark theme UI styling
│   └── app.js                    ← Chart.js graphs + labeling logic
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_merge_sensor_data.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── merge.py              ← Merge accel + gyro + GPS CSVs
│   │   ├── clean.py
│   │   ├── filter.py
│   │   └── window.py
│   ├── features/
│   │   └── extract.py
│   ├── models/
│   │   ├── train_rf.py
│   │   ├── train_xgb.py
│   │   ├── train_cnn.py
│   │   └── train_lstm.py
│   └── deployment/
│       ├── convert_tflite.py
│       └── api_server.py
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost_model.json
│   ├── cnn_model.h5
│   └── road_health_model.tflite  ← Final deployed model
│
├── dashboard/                    ← Road quality visualization dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── android_app/                  ← Android Studio project (Phase 5 only)
│   └── ...                       ← Built AFTER model is trained
│
├── videos/                       ← Dashcam footage (optional, for reference)
│   ├── session_001.mp4
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## 13. Evaluation Metrics

### What to Measure

| Metric | What it tells you | Target |
|---|---|---|
| **Accuracy** | Overall correctness | > 90% |
| **Precision** (per class) | Of all predicted potholes, how many were real? | > 85% |
| **Recall** (per class) | Of all real potholes, how many did we catch? | > 90% |
| **F1-Score** (per class) | Balance of precision & recall | > 87% |
| **Confusion Matrix** | Where the model gets confused | Few off-diagonals |
| **Inference Latency** | Time per prediction on phone | < 100ms |
| **Model Size** | TFLite file size | < 5 MB |

### Confusion Matrix Example (Goal)

```
                    Predicted
                Normal  Pothole  SpeedBreaker
Actual Normal    [  95%     3%        2%  ]
       Pothole   [   5%    92%        3%  ]
       SpeedBrkr [   3%     4%       93%  ]
```

---

## 14. Challenges & Mitigations

| Challenge | Impact | Mitigation |
|---|---|---|
| **Phone placement varies** | Different vibration patterns | Calibration step at app start; train with varied placements |
| **Vehicle type affects vibrations** | Car vs. bike vs. auto-rickshaw | Collect data from multiple vehicle types; add vehicle type as feature |
| **Speed affects vibration intensity** | Same bump feels different at different speeds | Normalize features by speed (from GPS) |
| **Class imbalance** (most road is normal) | Model biased toward "normal" | Use SMOTE oversampling; class weights in loss function |
| **Sensor noise** | False detections | Apply low-pass/high-pass filters; use moving average |
| **GPS accuracy in urban areas** | Wrong location tagging | Use fused location provider; apply Kalman filter smoothing |
| **Battery drain** | Users won't keep app running | Optimize sensor sampling; use batched sensor readings |
| **Different phone models** | Sensor quality varies | Normalize raw data; test across devices |

---

## 15. Timeline & Milestones

### Recommended Schedule (10-12 weeks for college project)

```
WEEK 1-2:  📚 Research & Planning
           ├── Literature review
           ├── Finalize approach
           ├── Set up Python environment
           ├── Install phyphox + GPS Logger on phone
           └── Do a 5-minute test drive to verify sensor recording works

WEEK 3:    🛠️ Build Web Labeling Tool
           ├── Create labeling_tool/ (HTML + CSS + JS)
           ├── Integrate Chart.js for interactive graphs
           ├── Add click-to-label functionality
           ├── Add CSV export with labels
           └── Test with sample data

WEEK 4-5:  🚗 Data Collection (Field Work)
           ├── Collect data from 5-10 routes using phyphox
           ├── Record GPS alongside with GPS Logger
           ├── (Optional) Record dashcam video
           ├── Merge sensor + GPS CSVs using Python script
           └── Collect from 2+ vehicle types

WEEK 5-6:  🏷️ Data Labeling
           ├── Open Web Labeling Tool
           ├── Upload merged CSVs one by one
           ├── Label all potholes and speed breakers
           ├── Aim for 500+ events per class
           └── Export labeled CSVs

WEEK 7-8:  🔧 Preprocessing & Feature Engineering
           ├── Clean raw data
           ├── Apply filters
           ├── Create sliding windows
           ├── Extract features
           └── Explore and visualize data

WEEK 9-10: 🤖 Model Training & Evaluation
           ├── Train Random Forest baseline
           ├── Train XGBoost
           ├── Train CNN / LSTM
           ├── Compare results
           ├── Hyperparameter tuning
           └── Select best model

WEEK 11:   🚀 Deployment
           ├── Convert to TFLite
           ├── Build Android app with TFLite integration
           ├── Test real-time inference on phone
           └── Build basic web dashboard (heatmap)

WEEK 12:   📝 Documentation & Presentation
           ├── Write project report
           ├── Create presentation
           ├── Demo preparation
           └── Final testing
```

---

## 16. References & Further Reading

### Research Papers
1. "Smartphone-based road surface monitoring using accelerometer data" — IEEE, 2024
2. "Deep Learning for Pothole Detection from Vibration Signals" — MDPI Sensors, 2024
3. "LSTM with 1D-LBP for Road Surface Classification" — London Met University, 2025
4. "Federated Learning for Privacy-Preserving Road Monitoring" — SciTePress, 2025

### Existing Projects & Tools
- **SmartRoadSense** (smartroadsense.it) — Open-source road quality monitoring
- **Roadroid** — Commercial road roughness measurement app
- **phyphox** (phyphox.org) — Physics experiments app with sensor export

### Datasets
- Search for "road surface classification accelerometer dataset" on:
  - Kaggle
  - UCI Machine Learning Repository
  - IEEE DataPort
  - Zenodo

### YouTube / Tutorials
- Search: "pothole detection using smartphone accelerometer machine learning"
- Search: "Android sensor data collection tutorial Kotlin"
- Search: "TensorFlow Lite Android integration guide"

---

## Quick Start Checklist

```
□ Step 1:  Install Python + required libraries (see Section 11)
□ Step 2:  Install phyphox + GPS Logger on your phone
□ Step 3:  Do a test drive — collect 10 minutes of data
□ Step 4:  Transfer CSVs to laptop, merge with Python script
□ Step 5:  Build the Web Labeling Tool (or use the one from Section 5)
□ Step 6:  Upload merged CSV → can you see pothole spikes? If yes, you're on track!
□ Step 7:  Label all spikes → export labeled CSV
□ Step 8:  Follow the preprocessing pipeline (Section 6)
□ Step 9:  Extract features and train Random Forest (Section 7)
□ Step 10: Evaluate — if accuracy > 85%, proceed to deep learning
□ Step 11: Train CNN/LSTM, convert to TFLite (Section 8)
□ Step 12: Build Android app with real-time inference (Section 8)
□ Step 13: Build web dashboard (Section 9)
□ Step 14: Demo and present! 🎉
```

---

> **💡 Pro Tips:**
> - Start simple — get a working pipeline with Random Forest first before trying deep learning.
> - Data quality matters more than model complexity. Spend time on good data collection.
> - The gyroscope is your secret weapon — it's what separates pothole from speed breaker.
> - Test on roads you haven't collected training data from (real-world generalization).
> - If this is a college project, the heatmap dashboard is your "wow factor" for the presentation.

---

*Last updated: June 15, 2026*
*Project: Road Health Detection System*
