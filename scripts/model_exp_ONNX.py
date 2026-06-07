#!/usr/bin/env python3
"""
Export trained Context Aware (CA)-YOLOv8 to ONNX and TensorRT for deployment.
"""

"""
When you build an AI model (like YOLOv8) during development, it usually runs inside frameworks like PyTorch.
But deployment environments (CCTV systems, edge devices, servers) often need faster and lighter versions of the model.

ONNX is a standard model format used to export AI models so they can run on different frameworks and hardware.

What it does

ONNX acts like a universal file format for neural networks.

That's where ONNX and TensorRT come in.

Exports trained YOLO models to the optimized ONNX format for faster edge device inference.

"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_model(model_path, formats=None, img_size=640, half=False):
    """Export trained model to various formats.

    Args:
        model_path: Path to .pt model file.
        formats: List of export formats (default: ['onnx']).
        img_size: Input image size.
        half: Use FP16 quantization.
    """
    from ultralytics import YOLO

    # No custom modules required for default YOLOv8 export

    if formats is None:
        formats = ['onnx']

    model = YOLO(model_path)

    for fmt in formats:
        print(f"\n[Export] Exporting to {fmt.upper()}...")
        exported = model.export(format=fmt, imgsz=img_size, half=half)
        print(f"[Export] Saved: {exported}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export CA-YOLOv8 model')
    parser.add_argument('--model', type=str, required=True, help='Path to .pt model')
    parser.add_argument('--format', type=str, nargs='+', default=['onnx'], help='Export formats')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--half', action='store_true', help='FP16 quantization')
    args = parser.parse_args()

    export_model(args.model, args.format, args.img_size, args.half)