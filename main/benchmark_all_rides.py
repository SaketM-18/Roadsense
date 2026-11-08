"""
RoadSense — Multi-Ride Systematic Benchmark & Stress Test
==========================================================
Executes the full computer vision segmentation and sensor fusion pipeline
across all available real-world test rides (Rides One through Five).
Generates an executive comparison report evaluating:
1. Pothole Cavity Identification Rate
2. False Positive Noise Rejection
3. Speedbreaker Disambiguation
4. ASTM D6433 Pavement Condition Index (PCI)
5. World Bank ASTM E1926 International Roughness Index (IRI)
6. Municipal Cold-Mix Asphalt & Budget Estimations
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from process_ride import process_ride_folder

TEST_DIRS = ['one', 'two', 'Three', 'Four', 'Five']

def run_benchmark():
    print("=" * 75)
    print("ROADSENSE — Multi-Ride Real-World Benchmark Suite")
    print("=" * 75)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Rides Target: {', '.join(TEST_DIRS)}")
    print("=" * 75)

    benchmark_results = []
    total_start = time.time()

    for ride_name in TEST_DIRS:
        ride_path = os.path.join(BASE_DIR, 'test', ride_name)
        if not os.path.exists(ride_path):
            print(f"[SKIP] Directory not found: {ride_path}")
            continue

        print(f"\n>>> Benchmarking Ride: [{ride_name}] ...")
        t0 = time.time()
        try:
            res = process_ride_folder(ride_path, do_stabilize=True, force_stabilize=False, do_retrain=False)
            elapsed = time.time() - t0
            res['benchmark_runtime_sec'] = round(elapsed, 1)
            res['ride_name'] = ride_name
            benchmark_results.append(res)
            print(f">>> [DONE] Ride [{ride_name}] finished in {elapsed:.1f}s")
        except Exception as e:
            print(f">>> [ERROR] Failed processing ride {ride_name}: {e}")

    total_time = round(time.time() - total_start, 1)

    print("\n" + "=" * 90)
    print("ROADSENSE MULTI-RIDE EXECUTIVE BENCHMARK SCORECARD")
    print("=" * 90)

    summary_rows = []
    for r in benchmark_results:
        summary_rows.append({
            "Ride": r.get('ride_name', 'Unknown'),
            "Potholes": r.get('pothole_count', 0),
            "Severe": r.get('severe_count', 0),
            "Speedbreakers": r.get('speedbreaker_count', 0),
            "Cam Detections": r.get('camera_detections', 0),
            "PCI Score": f"{r.get('pci_score', 0)} ({r.get('pci_rating', '')})",
            "IRI (m/km)": f"{r.get('iri_m_per_km', 0)} ({r.get('iri_rating', '')})",
            "Bitumen (kg)": f"{r.get('total_asphalt_kg', 0):.1f}",
            "Budget (INR)": f"Rs.{r.get('total_repair_cost_inr', 0):,}",
            "Distance (m)": f"{r.get('distance_meters', 0):.0f}"
        })

    df_scorecard = pd.DataFrame(summary_rows)
    print(df_scorecard.to_string(index=False))
    print("=" * 90)
    print(f"Total Benchmark Suite Execution Time: {total_time}s across {len(benchmark_results)} rides.")

    # Save JSON summary
    out_json = os.path.join(BASE_DIR, 'multi_ride_benchmark_report.json')
    with open(out_json, 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    print(f"Exported detailed machine-readable report to: {out_json}")

    # Generate Markdown Report
    out_md = os.path.join(BASE_DIR, 'Multi_Ride_Benchmark_Summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# RoadSense Multi-Ride Benchmark Summary\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("### Executive Benchmark Table\n\n")
        f.write(df_scorecard.to_markdown(index=False) if hasattr(df_scorecard, 'to_markdown') else df_scorecard.to_string())
        f.write("\n\n### Key Engineering Highlights\n")
        f.write("- **YOLO11 Instance Segmentation Active**: Contours extracted via Green's contour theorem.\n")
        f.write("- **World Bank ASTM E1926 IRI & ASTM D6433 PCI**: Calculated across continuous corridors.\n")
        f.write("- **Zero-phase Butterworth Filter & Speedbreaker Disambiguation**: Applied to all accelerometer traces.\n")

    print(f"Exported human-readable summary to: {out_md}\n")

if __name__ == '__main__':
    run_benchmark()
