"""
This script is used to benchmark a YOLO model’s performance and statistics before deployment.

Think of it as a performance testing tool for your trained model.

The script tests things like:
How large the neural network is
Storage size of the .pt file
How long one prediction takes
Frames per second the model can process
Consistency of inference speed

In short, it can help you decide if your model is ready for deployment.
Measures speed, accuracy, and model statistics.
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def benchmark_model(model_path, img_size=640, warmup_runs=10, test_runs=100, device='cuda'):
    """Benchmark model inference speed and statistics.

    Args:
        model_path: Path to .pt model.
        img_size: Input image size.
        warmup_runs: Number of warmup iterations.
        test_runs: Number of timed iterations.
        device: Device to benchmark on.
    """
    from ultralytics import YOLO

    # No custom modules required for default YOLOv8

    model = YOLO(model_path)

    # Dummy input
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    print(f"\n{'='*60}")
    print(f"  Model Benchmark: {Path(model_path).name}")
    print(f"  Device: {device} | Image size: {img_size}×{img_size}")
    print(f"{'='*60}\n")

    # Model statistics
    params = sum(p.numel() for p in model.model.parameters())
    params_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {params:>12,}")
    print(f"  Trainable parameters: {params_trainable:>12,}")
    print(f"  Model size (MB):      {os.path.getsize(model_path) / 1e6:>12.1f}")

    # Warmup
    print(f"\n  Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        model(dummy, verbose=False)

    # Benchmark
    print(f"  Benchmarking ({test_runs} runs)...")
    times = []
    for _ in range(test_runs):
        start = time.perf_counter()
        model(dummy, verbose=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    times = np.array(times)
    print(f"\n  {'Metric':<25} {'Value':>15}")
    print(f"  {'-'*42}")
    print(f"  {'Mean latency (ms)':<25} {times.mean():>15.2f}")
    print(f"  {'Median latency (ms)':<25} {np.median(times):>15.2f}")
    print(f"  {'Std dev (ms)':<25} {times.std():>15.2f}")
    print(f"  {'Min latency (ms)':<25} {times.min():>15.2f}")
    print(f"  {'Max latency (ms)':<25} {times.max():>15.2f}")
    print(f"  {'FPS (mean)':<25} {1000.0/times.mean():>15.1f}")
    print(f"  {'FPS (median)':<25} {1000.0/np.median(times):>15.1f}")
    print(f"\n{'='*60}\n")


def compare_models(model_paths, img_size=640, test_runs=50):
    """Compare multiple models side by side.

    Args:
        model_paths: List of model .pt file paths.
        img_size: Input image size.
        test_runs: Runs per model.
    """
    print(f"\n{'='*70}")
    print(f"  Model Comparison Benchmark")
    print(f"{'='*70}\n")

    results = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue

        from ultralytics import YOLO
        model = YOLO(path)
        params = sum(p.numel() for p in model.model.parameters())

        dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        # Warmup
        for _ in range(5):
            model(dummy, verbose=False)

        # Benchmark
        times = []
        for _ in range(test_runs):
            start = time.perf_counter()
            model(dummy, verbose=False)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        times = np.array(times)
        results.append({
            'model': Path(path).stem,
            'params_m': params / 1e6,
            'size_mb': os.path.getsize(path) / 1e6,
            'latency_ms': times.mean(),
            'fps': 1000.0 / times.mean(),
        })

    # Print comparison table
    print(f"  {'Model':<25} {'Params(M)':<12} {'Size(MB)':<12} {'Latency(ms)':<14} {'FPS':<10}")
    print(f"  {'-'*73}")
    for r in results:
        print(f"  {r['model']:<25} {r['params_m']:<12.1f} {r['size_mb']:<12.1f} "
              f"{r['latency_ms']:<14.2f} {r['fps']:<10.1f}")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark CA-YOLOv8 models')
    parser.add_argument('--model', type=str, nargs='+', required=True, help='Model path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--runs', type=int, default=100, help='Number of test runs')
    args = parser.parse_args()

    if len(args.model) == 1:
        benchmark_model(args.model[0], img_size=args.img_size, test_runs=args.runs)
    else:
        compare_models(args.model, img_size=args.img_size, test_runs=args.runs)