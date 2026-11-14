"""
RoadSense — Edge Runtime ONNX Model Quantizer
=============================================
Converts FP32 ONNX models (YOLOv8 Detector and YOLO11 Segmenter)
into optimized INT8 quantized models using ONNX Runtime Quantization.
Reduces model footprint by ~70% and accelerates edge CPU inference.
"""

import os
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V8_FP32 = os.path.join(BASE_DIR, 'pothole_yolov8.onnx')
V8_INT8 = os.path.join(BASE_DIR, 'pothole_yolov8_int8.onnx')
SEG_FP32 = os.path.join(BASE_DIR, 'pothole_yolo11_seg.onnx')
SEG_INT8 = os.path.join(BASE_DIR, 'pothole_yolo11_seg_int8.onnx')


def quantize_model(input_path, output_path, model_name="Model"):
    print(f"\n[{model_name}] Starting INT8 dynamic quantization...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")
        
    fp32_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"  Source FP32 size: {fp32_size:.2f} MB")
    
    t0 = time.time()
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        extra_options={'EnableSubgraph': True}
    )
    t_quant = time.time() - t0
    
    int8_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = ((fp32_size - int8_size) / fp32_size) * 100
    print(f"  Quantized INT8 size: {int8_size:.2f} MB ({reduction:.1f}% size reduction in {t_quant:.1f}s)")
    return fp32_size, int8_size


def benchmark_inference(fp32_path, int8_path, model_label="Model", iterations=20, imgsz=640):
    print(f"\n--- Latency Benchmark: {model_label} (CPU, {iterations} iterations, {imgsz}x{imgsz}) ---")
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    # 1. FP32 Session
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    sess_fp32 = ort.InferenceSession(fp32_path, sess_opts, providers=['CPUExecutionProvider'])
    input_name_fp32 = sess_fp32.get_inputs()[0].name
    
    # Warmup
    for _ in range(3):
        _ = sess_fp32.run(None, {input_name_fp32: dummy_input})
        
    t0 = time.time()
    for _ in range(iterations):
        _ = sess_fp32.run(None, {input_name_fp32: dummy_input})
    t_fp32 = ((time.time() - t0) / iterations) * 1000.0
    
    # 2. INT8 Session
    sess_int8 = ort.InferenceSession(int8_path, sess_opts, providers=['CPUExecutionProvider'])
    input_name_int8 = sess_int8.get_inputs()[0].name
    
    # Warmup
    for _ in range(3):
        _ = sess_int8.run(None, {input_name_int8: dummy_input})
        
    t0 = time.time()
    for _ in range(iterations):
        _ = sess_int8.run(None, {input_name_int8: dummy_input})
    t_int8 = ((time.time() - t0) / iterations) * 1000.0
    
    speedup = t_fp32 / max(t_int8, 0.001)
    fps_fp32 = 1000.0 / t_fp32
    fps_int8 = 1000.0 / t_int8
    
    print(f"  FP32 Latency: {t_fp32:.1f} ms/frame ({fps_fp32:.1f} FPS)")
    print(f"  INT8 Latency: {t_int8:.1f} ms/frame ({fps_int8:.1f} FPS)")
    print(f"  Speedup:      {speedup:.2f}x faster on CPU!")
    
    return {
        "fp32_ms": round(t_fp32, 1),
        "int8_ms": round(t_int8, 1),
        "speedup": round(speedup, 2),
        "fps_fp32": round(fps_fp32, 1),
        "fps_int8": round(fps_int8, 1)
    }


def run_quantization_pipeline():
    print("=" * 65)
    print("ROADSENSE — ONNX INT8 Edge Quantization Suite")
    print("=" * 65)
    
    # 1. YOLOv8 Detector
    v8_fp32_size, v8_int8_size = quantize_model(V8_FP32, V8_INT8, "YOLOv8-Detector")
    v8_bench = benchmark_inference(V8_FP32, V8_INT8, "YOLOv8-Detector", iterations=15)
    
    # 2. YOLO11 Segmenter
    seg_fp32_size, seg_int8_size = quantize_model(SEG_FP32, SEG_INT8, "YOLO11-Segmenter")
    seg_bench = benchmark_inference(SEG_FP32, SEG_INT8, "YOLO11-Segmenter", iterations=10)
    
    print("\n" + "=" * 65)
    print("QUANTIZATION & BENCHMARK SCORECARD")
    print("=" * 65)
    print(f"YOLOv8 Detector:  {v8_fp32_size:.1f}MB -> {v8_int8_size:.1f}MB | {v8_bench['fp32_ms']}ms -> {v8_bench['int8_ms']}ms ({v8_bench['fps_int8']} FPS, {v8_bench['speedup']}x)")
    print(f"YOLO11 Segmenter: {seg_fp32_size:.1f}MB -> {seg_int8_size:.1f}MB | {seg_bench['fp32_ms']}ms -> {seg_bench['int8_ms']}ms ({seg_bench['fps_int8']} FPS, {seg_bench['speedup']}x)")
    print("=" * 65)


if __name__ == '__main__':
    run_quantization_pipeline()

