"""
Export YOLOv8 model to ONNX format + optional fine-tuning helper.

Quick start:
    python src/export_model.py --model yolov8n          # export pretrained nano model
    python src/export_model.py --model yolov8n --train  # fine-tune first, then export
"""

import argparse
import os


def export_to_onnx(model_name: str = "yolov8n", output_dir: str = "models"):
    """Download pretrained weights and export to ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Run:  pip install ultralytics")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading {model_name}.pt ...")
    model = YOLO(f"{model_name}.pt")

    export_path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    print(f"[INFO] Exported → {export_path}")

    # Move into models/ folder
    dest = os.path.join(output_dir, f"{model_name}.onnx")
    os.replace(export_path, dest)
    print(f"[INFO] Saved to {dest}")
    return dest


def fine_tune(model_name: str, data_yaml: str, epochs: int = 50, imgsz: int = 640):
    """Fine-tune a YOLOv8 model on a custom dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Run:  pip install ultralytics")

    print(f"[INFO] Fine-tuning {model_name}.pt for {epochs} epochs ...")
    model = YOLO(f"{model_name}.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name="yolov8_custom",
        patience=10,
        save=True,
    )
    print(f"[INFO] Training complete. Best weights: {results.save_dir}/weights/best.pt")
    return results


def evaluate(model_name: str, data_yaml: str):
    """Evaluate model performance on validation set."""
    from ultralytics import YOLO

    model = YOLO(f"{model_name}.pt")
    metrics = model.val(data=data_yaml)
    print(f"\n[RESULTS]")
    print(f"  mAP@50:    {metrics.box.map50:.3f}")
    print(f"  mAP@50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Export / Train Helper")
    parser.add_argument("--model",   default="yolov8n", choices=["yolov8n","yolov8s","yolov8m","yolov8l","yolov8x"])
    parser.add_argument("--output",  default="models",  help="Directory to save ONNX model")
    parser.add_argument("--train",   action="store_true", help="Fine-tune before exporting")
    parser.add_argument("--data",    default="data/dataset.yaml", help="Path to dataset YAML (for training)")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--eval",    action="store_true", help="Run validation metrics")
    args = parser.parse_args()

    if args.train:
        fine_tune(args.model, args.data, args.epochs)

    if args.eval:
        evaluate(args.model, args.data)

    export_to_onnx(args.model, args.output)
