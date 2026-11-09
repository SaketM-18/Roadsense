# RoadSense Multi-Ride Benchmark Summary

*Generated on: 2026-11-10 00:43:25*

### Executive Benchmark Table

| Ride | Potholes | Severe | Speedbreakers | Cam Detections | PCI Score | IRI (m/km) | Bitumen (kg) | Budget (INR) | Distance (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one | 71 | 51 | 0 | 53 | 8.0 (Failed) | 0.8 (Very Good) | 5361.1 | Rs.262,568 | 690 |
| two | 79 | 53 | 0 | 63 | 8.0 (Failed) | 0.8 (Very Good) | 6736.8 | Rs.326,878 | 722 |
| Three | 159 | 92 | 0 | 161 | 8.0 (Failed) | 0.8 (Very Good) | 10520.7 | Rs.521,164 | 1083 |
| Four | 167 | 94 | 12 | 157 | 8.0 (Failed) | 16.0 (Serious / Failed) | 9461.5 | Rs.475,929 | 1027 |
| Five | 6 | 1 | 1 | 15 | 8.0 (Failed) | 16.0 (Serious / Failed) | 157.3 | Rs.8,877 | 2 |


### Key Engineering Highlights
- **YOLO11 Instance Segmentation Active**: Contours extracted via Green's contour theorem.
- **World Bank ASTM E1926 IRI & ASTM D6433 PCI**: Calculated across continuous corridors.
- **Zero-phase Butterworth Filter & Speedbreaker Disambiguation**: Applied to all accelerometer traces.
