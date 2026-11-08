# RoadSense Multi-Ride Benchmark Summary

*Generated on: 2026-11-08 15:14:09*

### Executive Benchmark Table

| Ride | Potholes | Severe | Speedbreakers | Cam Detections | PCI Score | IRI (m/km) | Bitumen (kg) | Budget (INR) | Distance (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one | 0 | 0 | 0 | 0 | 100.0 (Good) | 0.8 (Very Good) | 0.0 | Rs.0 | 690 |
| two | 0 | 0 | 0 | 0 | 100.0 (Good) | 0.8 (Very Good) | 0.0 | Rs.0 | 722 |
| Three | 0 | 0 | 0 | 0 | 100.0 (Good) | 0.8 (Very Good) | 0.0 | Rs.0 | 1083 |
| Four | 48 | 31 | 7 | 10 | 8.0 (Failed) | 16.0 (Serious / Failed) | 363.4 | Rs.30,890 | 1027 |
| Five | 0 | 0 | 0 | 0 | 100.0 (Good) | 16.0 (Serious / Failed) | 0.0 | Rs.0 | 2 |


### Key Engineering Highlights
- **YOLO11 Instance Segmentation Active**: Contours extracted via Green's contour theorem.
- **World Bank ASTM E1926 IRI & ASTM D6433 PCI**: Calculated across continuous corridors.
- **Zero-phase Butterworth Filter & Speedbreaker Disambiguation**: Applied to all accelerometer traces.
