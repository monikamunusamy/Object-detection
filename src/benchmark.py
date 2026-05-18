"""
Benchmark inference speed across multiple runs.

Usage:
    python src/benchmark.py --model models/yolov8n.onnx --runs 200
"""

import cv2
import numpy as np
import argparse
import time


def benchmark(model_path: str, runs: int = 100, warmup: int = 10):
    net = cv2.dnn.readNetFromONNX(model_path)

    # Dummy frame (random noise)
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(dummy, 1/255.0, (640, 640), swapRB=True, crop=False)

    print(f"[INFO] Warming up ({warmup} runs) ...")
    for _ in range(warmup):
        net.setInput(blob)
        net.forward()

    print(f"[INFO] Benchmarking ({runs} runs) ...")
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        net.setInput(blob)
        net.forward()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"\n{'─'*40}")
    print(f"  Model    : {model_path}")
    print(f"  Runs     : {runs}")
    print(f"  Mean     : {latencies.mean():.2f} ms")
    print(f"  Median   : {np.median(latencies):.2f} ms")
    print(f"  Std      : {latencies.std():.2f} ms")
    print(f"  Min      : {latencies.min():.2f} ms")
    print(f"  Max      : {latencies.max():.2f} ms")
    print(f"  Avg FPS  : {1000/latencies.mean():.1f}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--runs",  type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    benchmark(args.model, args.runs, args.warmup)
