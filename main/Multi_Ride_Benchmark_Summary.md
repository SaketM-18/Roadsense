# RoadSense Multi-Ride Benchmark Summary

*Generated on: 2026-11-08 19:35:48*

### Executive Benchmark Table

| Ride | Potholes | Severe | Speedbreakers | Cam Detections | PCI Score | IRI (m/km) | Bitumen (kg) | Budget (INR) | Distance (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one | 27 | 18 | 0 | 27 | 8.0 (Failed) | 0.8 (Very Good) | 183.5 | Rs.16,392 | 690 |
| two | 30 | 13 | 0 | 30 | 8.0 (Failed) | 0.8 (Very Good) | 142.9 | Rs.15,609 | 722 |
| Three | 77 | 59 | 0 | 40 | 8.0 (Failed) | 0.8 (Very Good) | 547.9 | Rs.47,980 | 1083 |
| Four | 48 | 31 | 7 | 10 | 8.0 (Failed) | 16.0 (Serious / Failed) | 363.4 | Rs.30,890 | 1027 |
| Five | 3 | 1 | 0 | 3 | 8.0 (Failed) | 16.0 (Serious / Failed) | 6.2 | Rs.1,240 | 2 |


### Key Engineering Highlights
- **YOLO11 Instance Segmentation Active**: Contours extracted via Green's contour theorem.
- **World Bank ASTM E1926 IRI & ASTM D6433 PCI**: Calculated across continuous corridors.
- **Zero-phase Butterworth Filter & Speedbreaker Disambiguation**: Applied to all accelerometer traces.
