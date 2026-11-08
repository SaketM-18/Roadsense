# 🛣️ RoadSense: AI Infrastructure Intelligence

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-blue.svg?logo=ultralytics)](https://ultralytics.com)
[![Leaflet](https://img.shields.io/badge/Mapping-Leaflet-199900.svg?logo=leaflet)](https://leafletjs.com)
[![Standard](https://img.shields.io/badge/Standards-ASTM_D6433_|_IRI-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

> **RoadSense** is an automated IoT and machine learning infrastructure monitoring platform that transforms commodity smartphones into enterprise-grade road surveying tools. By fusing high-frequency physical telemetry with computer vision, RoadSense detects, verifies, classifies, and maps road surface hazards (potholes, speed breakers, degradation) in real-time.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ROADSENSE PLATFORM                             │
└─────────────────────────────────────────────────────────────────────────────┘

 [SMARTPHONE / VEHICLE SENSORS]
  • 40 Hz 3D Accelerometer (ax, ay, az, dynamic magnitude)
  • Gyroscope pitch & roll (gx, gy, gz)
  • High-accuracy GPS Telemetry (Lat, Lon, Velocity)
  • Dashcam Optical Video Feed (30 FPS)
                       │
                       │ REST / WebSockets
                       ▼
 [CENTRAL API BACKEND (FastAPI: main/server.py)]
  • Asynchronous Background Worker Queue
  • Static PWA Dashboard Hosting (main/survey_app/)
  • Dynamic PDF & GeoJSON Dispatch
                       │
                       ├──────────────────────────────────────────────────────┐
                       │                                                      │
                       ▼                                                      ▼
 [AI & PHYSICS ENGINE (process_ride.py)]                 [SPATIAL CLUSTERING (spatial_cluster.py)]
  • Streaming Video Stabilization (O(1) RAM)              • Haversine DBSCAN (ε = 8.0 meters)
  • YOLOv8 Nano Pothole Detection + CLAHE                 • Multi-Ride Consensus & Confidence
  • Dynamic Velocity Delay: t_impact = d / v              • Centroid Synthesis
  • Strict Physical Verification Gate (Z-Score)           • Citywide Master GeoJSON
  • Pothole Dimensions (cm² area & cm depth)
  • ASTM D6433 Pavement Condition Index (PCI)
  • International Roughness Index (IRI)
  • Visual Photographic Evidence Extraction
                       │
                       ▼
 [OUTPUTS & MUNICIPAL DISPATCH]
  • RFC 7946 Canonical GeoJSON (GIS / Mapbox / QGIS)
  • ASTM D6433 Municipal PDF Inspection Report & Work-Orders
  • Interactive Web Dashboard (Leaflet + Dark Mode)
```

---

## ✨ Key Real-World Features

### 1. True Multi-Modal Sensor Fusion (Strict Verification Gate)
* **The AND Gate**: When the camera spots a pothole, the physical telemetry engine verifies it against the calibrated vehicle vibration baseline ($Z\text{-score} > 1.5$). Visual false positives (shadows, road paint, oil spills) are thrown out.
* **The Failsafe Override**: Severe mechanical shocks ($Z\text{-score} \ge 3.0$) are recorded even if missed by the camera (e.g. at night or behind a vehicle).
* **Speedbreaker Disambiguation**: Uses pitch deflection rate ($\text{gyro}_y$) to distinguish smooth speed bumps from sharp pothole impacts.

### 2. Dynamic Speed-Dependent Lookahead ($t_{\text{impact}}$)
Unlike static time delays, RoadSense dynamically calculates the physical tire impact window based on vehicle velocity:
$$t_{\text{impact}} = t_{\text{vision}} + \text{clip}\left(\frac{d_{\text{lookahead}}}{v_{\text{speed}}},\, 0.15\text{s},\, 1.5\text{s}\right)$$

### 3. Orientation-Invariant 3D IMU & 40 Hz Telemetry
* Eliminates dependence on the phone being flat by calculating the true dynamic shock magnitude:
  $$a_{\text{dyn}} = |\sqrt{a_x^2 + a_y^2 + a_z^2} - 9.81|$$
* Records high-frequency telemetry at **40 Hz**, preventing transient 50 ms pothole shocks from slipping between samples.

### 4. Spatial Clustering & Multi-Ride Consensus (DBSCAN)
* Consolidates duplicate GPS detections across different runs using **Haversine DBSCAN ($\epsilon = 8.0\text{ m}$)**.
* Computes **Multi-Pass Ground Truth Confidence**:
  * **High Confidence (Gold Halo)**: Verified across $\ge 2$ independent vehicle passes.
  * **Medium Confidence**: Significant single-pass impact.

### 5. Standardized Pavement Ratings (ASTM D6433 PCI & IRI)
* Computes standard municipal **Pavement Condition Index (PCI, 0–100 scale)**:
  * 85–100: **Good** | 70–84: **Satisfactory** | 55–69: **Fair** | 40–54: **Poor** | 25–39: **Very Poor** | $<25$: **Serious / Failed**
* Calculates estimated **International Roughness Index (IRI, m/km)**.

### 6. Photographic Visual Evidence & Pothole Dimensions
* Automatically crops high-resolution frames with green bounding boxes and confidence overlays for each verified hazard (`evidence/`).
* Estimates physical pothole area in $\text{cm}^2$ and impact depth in $\text{cm}$.
* Embeds clickable photos into the interactive map and official PDF work orders.

### 7. Municipal Inspection Work-Order PDF Generator
* Generates audit-ready PDF reports with ReportLab containing:
  * Executive condition scorecard (PCI score, IRI rating, defect density).
  * Priority repair dispatch table sorted by severity with exact GPS coordinates.
  * Recommended maintenance action (Emergency Cold Patching, Milling, etc.).
  * Photo gallery of verified defects.

### 8. Live Driver Heads-Up Proximity Alerts ("Waze for Road Hazards")
* **Real-Time Warning HUD:** As a rider approaches within 50–80m of any severe or moderate defect in the city database, a neon alert pill triggers with a distance countdown.
* **Audio Synthesizer & Speech Warnings:** Emits dual-tone warning chimes and voice alerts (*"Caution: Severe Pothole 45m ahead"*) using native Web Audio & Speech APIs without downloading external sound assets.
* **Speed-Adaptive Lookahead Distance:** Automatically increases the warning lookahead distance as vehicle speed increases.

### 9. Municipal Defect Lifecycle & Pothole Repair Auto-Verification
* **Trajectory-Matched Verification:** Tracks defect trajectories over time using vectorized Haversine distance. When a subsequent vehicle ride passes over a known canonical hazard:
  * **Smooth Asphalt ($Z_{\text{score}} < 1.0$):** Automatically marks the defect as `Repaired & Verified`, updating its lifecycle stage to `Repaired & Closed`.
  * **Persistent Shock ($Z_{\text{score}} \ge 2.5$):** Flags the defect as `Deteriorating Defect`, increments the deterioration counter, and escalates municipal dispatch priority to `P1 (Urgent)`.
* **Contractor Dispatch & Verification:** Municipal engineers can toggle repair statuses directly on the live map (`POST /api/hazards/{id}/status`), trigger auto-verification across all passes (`POST /api/hazards/auto-verify`), and export contractor dispatch work-orders as CSV.

### 10. Smart Edge Shock Triggering (95–99% Storage & Bandwidth Saver)
* **In-Memory FIFO Rolling Ring-Buffer:** Instead of constantly saving massive 1080p continuous video files to phone storage, RoadSense uses a circular in-memory buffer holding the most recent 3.0 seconds (6 x 500ms slices) with strict $O(1)$ RAM usage.
* **Incident Window Capture:** When dynamic acceleration spikes ($a_{\text{dyn}} \ge 2.5\text{ m/s}^2$):
  * Locks the pre-impact buffer ($1.5\text{s}$ prior).
  * Snapshots the camera frame via canvas for instant visual evidence.
  * Records for $1.5\text{s}$ post-impact, outputting a compact 3.0-second incident `.webm` snippet.
* **Massive Edge Efficiency:** Surveys that previously consumed 200–500 MB now consume only 2–5 MB, saving **95–99%** of phone disk space and cellular data. Smooth passes save 100% video storage.

### 11. Municipal Asphalt Material & Financial Repair Budgeting
* **Parabolic Cavity Volume Model:** Automatically computes asphalt cavity volume:
  $$V_{\text{asphalt}} = \frac{1}{2} \times \text{Area (cm}^2) \times \text{Depth (cm)} \times 10^{-3} \quad (\text{Liters})$$
* **Material Procurement Mass (kg):** Multiplies volume by compacted bitumen density ($\rho \approx 2.35\text{ kg/L}$) to calculate cold-mix bitumen mass needed per defect.
* **Contractor Cost Estimation:** Computes material and base compaction labor costs (in ₹ INR and \$ USD) per hazard and aggregated citywide across all active defects.

### 12. World Bank ASTM E1926 Quarter-Car Simulation & Continuous Corridor Heatmaps
* **Golden Car 2-DOF Suspension Simulation:** Solves the ASTM E1926 4-state differential equations on real accelerometer dynamics to produce certified International Roughness Index (IRI in $\text{m/km}$) matching municipal laser profiler vans.
* **25-Meter Spatial Corridors:** Slices continuous GPS survey routes into micro-segments with ASTM D6433 PCI color coding (Neon Green $\rightarrow$ Lime $\rightarrow$ Yellow $\rightarrow$ Orange $\rightarrow$ Red), rendered dynamically on Leaflet.

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/RoadSense.git
cd RoadSense
pip install -r requirements.txt
```

### 2. Launch the Web Application & Backend Server
Start the unified platform with one click:
```bash
.\run_server.bat
```
or run manually:
```bash
python main/server.py
```
Open **`http://localhost:8000`** in your browser. (On a phone connected to the same Wi-Fi, open `http://<YOUR_COMPUTER_IP>:8000`).

---

## 🧠 AI Models & Benchmarks

RoadSense couples high-frequency physical dynamics with deep computer vision to achieve near-zero false alarms:

| Model | Architecture | Dataset / Input | Key Metric | Role |
|---|---|---|---|---|
| **YOLO Vision** | YOLOv8s (21.5 MB) | 10,000+ dashcam road frames | **81.06% mAP@50** (77.2% Precision) | Spots visual defects, bounding boxes, estimated surface area ($\text{cm}^2$) |
| **Sensor Fusion** | Soft Voting Ensemble (HGB + RF) | 20,449 multi-ride IMU records | **91.34% ROC-AUC** (84.5% Accuracy) | Validates physical tire strike ($\frac{da}{dt}$ jerk, 6–14 Hz resonance, z-score) |

### Retraining Vision on Cloud GPUs (Kaggle / Colab)
A self-contained Kaggle retraining pipeline is included:
* Notebook: `main/cv_pipeline/train_pothole_yolo_kaggle.ipynb`
* Script: `main/cv_pipeline/train_pothole_yolo.py`
* Run remotely with:
  ```bash
  kaggle kernels push -p kaggle_train_job
  ```

---

## 🛠️ CLI Pipeline Usage

You can also run individual processing modules directly:

### Process a Ride Directory
```bash
# Full pipeline with video stabilization and YOLO fusion
python main/process_ride.py "main/test/Four"

# Fast mode (bypasses video re-stabilization if camera EIS is active)
python main/process_ride.py "main/test/Four" --no-stabilize
```

### Calibrate Vehicle Suspension Profile
Drive over smooth asphalt across various speeds, record telemetry, and run:
```bash
python main/calibrate.py "path/to/calibration_recording.csv"
```

### Run Spatial Deduplication & Generate Citywide GeoJSON
```bash
python main/spatial_cluster.py "main/test"
```
Outputs:
* `main/aggregated_hazards/canonical_hazards.geojson`
* `main/aggregated_hazards/canonical_hazards.csv`

### Generate Municipal Work-Order PDF & Contractor Dispatch CSV
```bash
python main/generate_inspection_report.py "main/test/Four/fused_labeled.csv"
```
Outputs:
* `RoadSense_Inspection_WorkOrder.pdf`
* `inspection_work_order.csv`

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System health, profile status, active maps |
| `POST` | `/api/rides/upload` | Upload mobile telemetry CSV + video (runs async) |
| `GET` | `/api/rides` | List all historical and uploaded surveys with PCI scores |
| `GET` | `/api/rides/{id}` | Get full telemetry rows and GPS coordinates |
| `GET` | `/api/rides/{id}/report.pdf` | Generate and download official Municipal PDF Audit |
| `GET` | `/api/rides/{id}/download` | Download labeled fused CSV |
| `GET` | `/api/rides/{id}/evidence/{img}` | Serve visual evidence bounding-box snapshot |
| `GET` | `/api/rides/{id}/corridors` | Slices ride into 25m continuous micro-segments with PCI colors for heatmaps |
| `GET` | `/api/hazards/geojson` | Fetch canonical deduplicated hazard FeatureCollection |
| `GET` | `/api/hazards/summary` | Real-time municipal lifecycle metrics (repaired, active, resolution rate) |
| `POST` | `/api/hazards/auto-verify` | Run automated multi-pass repair verification from survey rides |
| `POST` | `/api/hazards/{cluster_id}/status` | Update canonical hazard repair status and lifecycle stage |
| `GET` | `/api/hazards/export/geojson` | Download RFC 7946 GeoJSON attachment for QGIS / ArcGIS |
| `GET` | `/api/hazards/export/contractor-dispatch.csv` | Download municipal contractor dispatch work-orders as CSV |
| `POST` | `/api/calibrate/upload` | Upload calibration CSV and update vehicle profile |
| `GET` | `/` | Web Dashboard frontend (survey_app) |

Interactive Swagger API documentation is available at **`http://localhost:8000/docs`**.

---

## 📂 Project Directory Structure

```text
RoadSense/
├── main/
│   ├── server.py                        # FastAPI central backend & static web server
│   ├── process_ride.py                  # Core pipeline (Optical flow, YOLOv8, sensor fusion)
│   ├── spatial_cluster.py               # Haversine DBSCAN multi-ride deduplication engine
│   ├── generate_inspection_report.py    # ASTM D6433 Municipal PDF & CSV work-order generator
│   ├── calibrate.py                     # Vehicle suspension calibration script
│   ├── vehicle_profile.json             # Speed-binned baseline vibration profile
│   │
│   ├── survey_app/                      # Interactive Web Application
│   │   ├── index.html                   # Command center, Leaflet map, 40Hz recorder
│   │   └── canonical_hazards.json       # Pre-computed citywide clustered hazard layer
│   │
│   ├── cv_pipeline/                     # Computer vision models & master dataset
│   │   ├── pothole_yolov8.pt            # Pre-trained YOLOv8 pothole weights
│   │   └── master_training_data.csv     # Master training dataset
│   │
│   ├── ml_pipeline/                     # Machine learning models & feature extractors
│   │   ├── train_fusion_model.py        # Random Forest sensor fusion trainer
│   │   └── road_health_rf_model_v2.pkl  # Trained Random Forest classifier
│   │
│   ├── aggregated_hazards/              # Citywide canonical hazard outputs
│   │   ├── canonical_hazards.geojson    # GeoJSON for GIS / Leaflet
│   │   └── canonical_hazards.csv        # Aggregated tabular hazard summary
│   │
│   └── test/                            # Historical real-world scooter test runs
│       ├── one/                         # Ride scan 1
│       ├── two/                         # Ride scan 2
│       └── Four/                        # Ride scan 4 (with video, CSV, PDF, evidence)
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Comprehensive project guide
```

---

## 📜 License
This project is licensed under the MIT License.

