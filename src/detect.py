"""
Real-Time Object Detection using YOLOv8
Author: Your Name
Description: Detect objects in images, videos, or webcam feed using YOLOv8 + ONNX Runtime
"""

import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Assign a unique color to each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8).tolist()

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = 640


# ─── Core Detection ───────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load YOLOv8 ONNX model using OpenCV DNN."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Export it first:\n"
            "  from ultralytics import YOLO\n"
            "  model = YOLO('yolov8n.pt')\n"
            "  model.export(format='onnx')"
        )
    net = cv2.dnn.readNetFromONNX(model_path)
    print(f"[INFO] Model loaded: {model_path}")
    return net


def preprocess(frame: np.ndarray):
    """Resize and normalise frame for YOLOv8 input."""
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 255.0,
        size=(INPUT_SIZE, INPUT_SIZE),
        swapRB=True,
        crop=False,
    )
    return blob


def postprocess(outputs, orig_w: int, orig_h: int, conf_thresh: float, nms_thresh: float):
    """
    Parse YOLOv8 raw output into (boxes, scores, class_ids).

    YOLOv8 output shape: [1, 84, 8400]
    84 = 4 (cx,cy,w,h) + 80 class scores
    """
    predictions = np.squeeze(outputs[0]).T   # (8400, 84)

    # Extract bounding boxes and class scores
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    # Best class per prediction
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter by confidence
    mask = confidences >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return [], [], []

    # Scale from model coords to original image coords
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    # Convert cx,cy,w,h → x1,y1,w,h (top-left format for NMS)
    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) * scale_x
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) * scale_y
    w  =  boxes_xywh[:, 2] * scale_x
    h  =  boxes_xywh[:, 3] * scale_y

    boxes = np.stack([x1, y1, w, h], axis=1).astype(int).tolist()

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences.tolist(), conf_thresh, nms_thresh)
    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()
    return (
        [boxes[i] for i in indices],
        [confidences[i] for i in indices],
        [class_ids[i] for i in indices],
    )


def draw_detections(frame: np.ndarray, boxes, scores, class_ids) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x, y, w, h = box
        label = COCO_CLASSES[cls_id]
        color = COLORS[cls_id]

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        text = f"{label}  {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)

        # Label text
        cv2.putText(
            frame, text,
            (x + 4, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return frame


# ─── Inference Pipeline ───────────────────────────────────────────────────────

def run_on_image(net, image_path: str, output_path: str = None, show: bool = True):
    """Run detection on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = frame.shape[:2]
    blob = preprocess(frame)
    net.setInput(blob)

    t0 = time.perf_counter()
    outputs = net.forward()
    latency_ms = (time.perf_counter() - t0) * 1000

    boxes, scores, class_ids = postprocess(outputs, w, h, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    frame = draw_detections(frame, boxes, scores, class_ids)

    # Stats overlay
    info = f"Objects: {len(boxes)}  |  Latency: {latency_ms:.1f}ms"
    cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 160), 2)

    print(f"[INFO] {len(boxes)} object(s) detected in {latency_ms:.1f}ms")
    for box, score, cls_id in zip(boxes, scores, class_ids):
        print(f"       └─ {COCO_CLASSES[cls_id]:<20} conf={score:.2f}  box={box}")

    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"[INFO] Saved → {output_path}")

    if show:
        cv2.imshow("YOLOv8 Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frame


def run_on_video(net, source, output_path: str = None, show: bool = True):
    """Run detection on a video file or webcam (source=0)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    frame_count = 0
    total_latency = 0.0

    print(f"[INFO] Starting inference on {'webcam' if source == 0 else source} ...")
    print("[INFO] Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = preprocess(frame)
        net.setInput(blob)

        t0 = time.perf_counter()
        outputs = net.forward()
        latency_ms = (time.perf_counter() - t0) * 1000

        boxes, scores, class_ids = postprocess(outputs, w, h, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        frame = draw_detections(frame, boxes, scores, class_ids)

        frame_count += 1
        total_latency += latency_ms
        avg_fps = 1000 / (total_latency / frame_count)

        overlay = f"FPS: {avg_fps:.1f}  |  Latency: {latency_ms:.0f}ms  |  Objects: {len(boxes)}"
        cv2.putText(frame, overlay, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 160), 2)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Processed {frame_count} frames — avg {1000/(total_latency/frame_count):.1f} FPS")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Object Detection")
    parser.add_argument("--model",  default="models/yolov8n.onnx", help="Path to ONNX model")
    parser.add_argument("--source", default="0",                   help="Image path, video path, or 0 for webcam")
    parser.add_argument("--output", default=None,                  help="Save result to this path")
    parser.add_argument("--conf",   type=float, default=0.5,       help="Confidence threshold (default 0.5)")
    parser.add_argument("--nms",    type=float, default=0.4,       help="NMS IoU threshold (default 0.4)")
    parser.add_argument("--no-show", action="store_true",          help="Do not display window")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    CONFIDENCE_THRESHOLD = args.conf
    NMS_THRESHOLD        = args.nms
    show                 = not args.no_show

    net = load_model(args.model)

    # Detect source type
    source = args.source
    if source.isdigit():
        run_on_video(net, int(source), args.output, show)
    elif source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        run_on_image(net, source, args.output, show)
    else:
        run_on_video(net, source, args.output, show)
