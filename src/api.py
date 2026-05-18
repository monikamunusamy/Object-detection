"""
FastAPI REST API for YOLOv8 object detection.

Start the server:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /detect        — Upload an image, get detections back as JSON
    GET  /health        — Health check
    GET  /docs          — Swagger UI (auto-generated)
"""

import cv2
import numpy as np
import time
import os
from io import BytesIO
from typing import List

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Run: pip install fastapi uvicorn python-multipart")

from detect import (
    load_model, preprocess, postprocess, COCO_CLASSES,
    CONFIDENCE_THRESHOLD, NMS_THRESHOLD,
)


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Upload an image and get back detected objects with bounding boxes and confidence scores.",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.onnx")
net = load_model(MODEL_PATH)


# ─── Response schema ──────────────────────────────────────────────────────────

class Detection(BaseModel):
    label: str
    confidence: float
    box: List[int]   # [x, y, width, height]


class DetectionResponse(BaseModel):
    image_size: List[int]
    num_detections: int
    latency_ms: float
    detections: List[Detection]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    h, w = frame.shape[:2]
    blob = preprocess(frame)
    net.setInput(blob)

    t0 = time.perf_counter()
    outputs = net.forward()
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    boxes, scores, class_ids = postprocess(outputs, w, h, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    detections = [
        Detection(
            label=COCO_CLASSES[cls_id],
            confidence=round(float(score), 4),
            box=box,
        )
        for box, score, cls_id in zip(boxes, scores, class_ids)
    ]

    return DetectionResponse(
        image_size=[w, h],
        num_detections=len(detections),
        latency_ms=latency_ms,
        detections=detections,
    )
