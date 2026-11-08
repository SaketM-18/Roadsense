# RoadSense Multi-Ride Benchmark Summary

*Generated on: 2026-11-09 00:43:37*

### Executive Benchmark Table

| Ride | Potholes | Severe | Speedbreakers | Cam Detections | PCI Score | IRI (m/km) | Bitumen (kg) | Budget (INR) | Distance (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one | 69 | 48 | 0 | 52 | 8.0 (Failed) | 0.8 (Very Good) | 497.6 | Rs.43,516 | 690 |
| two | 77 | 44 | 0 | 60 | 8.0 (Failed) | 0.8 (Very Good) | 599.6 | Rs.50,512 | 722 |
| Three | 164 | 85 | 0 | 159 | 8.0 (Failed) | 0.8 (Very Good) | 1024.3 | Rs.96,549 | 1083 |
| Four | 184 | 94 | 12 | 175 | 8.0 (Failed) | 16.0 (Serious / Failed) | 1945.8 | Rs.143,970 | 1027 |
| Five | 9 | 2 | 1 | 18 | 8.0 (Failed) | 16.0 (Serious / Failed) | 32.0 | Rs.4,321 | 2 |


### Key Engineering Highlights
- **YOLO11 Instance Segmentation Active**: Contours extracted via Green's contour theorem.
- **World Bank ASTM E1926 IRI & ASTM D6433 PCI**: Calculated across continuous corridors.
- **Zero-phase Butterworth Filter & Speedbreaker Disambiguation**: Applied to all accelerometer traces.
