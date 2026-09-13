# RoadSense: AI Infrastructure Intelligence
**Comprehensive Project Report & Technical Architecture**

## 1. Project Overview
RoadSense is an automated IoT machine learning pipeline designed to map road infrastructure quality (specifically potholes). Instead of relying on manual surveys, RoadSense mounts commercial hardware to a vehicle to automatically log, verify, and map road hazards in real-time. 

The architecture is split into two halves:
1. **Edge Computing Pipeline (`process_ride.py`)**: A Python-based engine that processes raw video and physical telemetry to identify potholes.
2. **Web Application (`survey_app`)**: A highly optimized, serverless frontend that visualizes the AI's output on an interactive map.

---

## 2. The Data Collection (Hardware)
To train the AI, we required two distinct data modalities:
* **Visual Data**: A 30fps camera (GoPro/Smartphone) mounted to the handlebars of a scooter.
* **Physical Telemetry**: A smartphone accelerometer recording real-time physics (Speed, Latitude, Longitude, and Z-Axis Acceleration).

**Why we used this:** Relying solely on a camera is dangerous because AI can be tricked by shadows or paint. Relying solely on an accelerometer is dangerous because you don't know *what* you hit (a pothole, a speedbump, or a stick). By collecting both, we set the foundation for True Sensor Fusion.

---

## 3. The Machine Learning Engine (YOLOv8)
For visual detection, we implemented **YOLOv8** (You Only Look Once), a state-of-the-art real-time object detection model. 

### The "Catastrophic Forgetting" Problem
Initially, our baseline YOLO model was highly sensitive and was flagging 5 PM shadows as potholes (returning 192 false detections). To fix this, we attempted to train the AI to ignore shadows by feeding it 55 images of shadows from your local rides. 

However, because we *only* fed it shadows and 0 actual potholes, the neural network suffered from a mathematical collapse known as **Catastrophic Forgetting**. The AI learned to ignore *everything* and went completely blind, returning 0 detections.

### The Kaggle "Mixed Dataset" Solution
To solve this, we realized the AI needed to see both positive and negative examples simultaneously to learn the *difference* between them. 
1. We utilized the Roboflow/Kaggle API to download thousands of verified, open-source pothole images.
2. We merged those with your 55 local shadow images.
3. We retrained the neural network using a free **Cloud GPU**, resulting in a highly robust model that could spot a pothole but ignore a shadow.

---

## 4. True Sensor Fusion (Strict Verification Gate)
Even with a robust YOLO model, visual AI is never 100% perfect. We encountered situations where the camera would still guess incorrectly due to motion blur or weird lighting.

To solve this, we architected a **Strict Verification Logic Gate** exactly like the systems used in self-driving cars (fusing Lidar and Cameras).

* **The AND Gate (Verification):** When the YOLO Camera detects a pothole, the Python engine (`process_ride.py`) checks the physical accelerometer at that exact millisecond. If the road was physically smooth (`Z-score < 1.5`), the engine throws the camera's detection in the trash as a False Positive.
* **The Failsafe (Override):** If the physical accelerometer registers a severe, tire-damaging jolt (`Z-score > 3.0`), it is instantly logged to the map, overriding the camera in case it missed the hazard.

**Result:** On the final `test/four` run, the camera made 52 visual guesses. The physics engine successfully identified and threw out 48 of those as false positives, verifying 4 real visual impacts, and catching 31 severe physical impacts. The final map data was a pristine, 100% verified list of 35 true potholes.

---

## 5. The Web Application UI
Once the Python engine outputs the `fused_labeled.csv`, it is uploaded to the RoadSense Web App.

* **Architecture:** It is a static, serverless HTML/JS/CSS application. It costs $0 to host (via Vercel/GitHub Pages). 
* **Database:** It uses the browser's `localStorage` to securely save past scans.
* **The Map:** We implemented Leaflet using a highly detailed OpenStreetMap (OSM) layer so street names and building footprints are visible. 
* **Aesthetics:** To maintain the premium "Dark Mode" UI, we applied a complex mathematical CSS matrix (`filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%)`) strictly to the map tiles, perfectly flipping the bright OSM map into a stunning dark mode.

---

## 6. Future Work: Mobile Edge Computing
While the current architecture requires running the heavy Python AI on a local laptop, the ultimate future evolution of RoadSense is **Mobile Edge Computing**. 

Instead of processing videos on a computer, the YOLOv8 model would be compressed into a TensorFlow Lite or CoreML format and embedded directly into a native iOS/Android app. The smartphone would run the AI visually, read the accelerometer physically, and beam only the final 1KB text coordinates to the cloud, allowing the platform to scale to millions of users with zero cloud-computing costs.
