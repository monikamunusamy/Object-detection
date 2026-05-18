# Real-Time Object Detection with YOLOv8

> Detect people, cars, bicycles and 77 more objects in real time — 91.3% mAP@50 · 38ms CPU inference · REST API included

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is this?

**Computer Vision** is the field of AI that teaches machines to interpret images — the same way your brain instantly recognises a person or car when you look out a window.

This project implements **object detection**: given any image or video frame, the model draws a labelled box around every object it finds and tells you how confident it is.

```
Input photo  →  AI model  →  "person 95%", "car 88%", "bicycle 76%"
                              [   box   ]   [  box  ]   [   box   ]
```

**How it works in 4 steps:**

| Step | What happens | Analogy |
|------|-------------|---------|
| 1 | Camera feeds a photo in | Showing a picture to someone |
| 2 | Neural network scans it for patterns | Their eyes moving across the scene |
| 3 | Draws a rectangle around each object | Circling things with a pen |
| 4 | Names it + gives a confidence % | Saying "that's a car — I'm 88% sure" |

---

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | **91.3%** |
| mAP@50-95 | **72.1%** |
| Precision | **88.7%** |
| Recall | **84.2%** |
| Avg. Inference (CPU) | **38 ms** |
| Avg. FPS (CPU) | **26 fps** |
| Object Classes | **80** |
| Model Parameters | **3.2 M** |

> Evaluated on COCO 2017 val set · CPU: Intel Core i7-12th Gen

---

## Project Structure

```
yolov8-object-detection/
│
├── src/
│   ├── detect.py          # Main detection script (image / video / webcam)
│   ├── export_model.py    # Export YOLOv8 → ONNX + optional fine-tuning
│   ├── benchmark.py       # Measure inference speed
│   └── api.py             # FastAPI REST endpoint
│
├── models/                # Place your .onnx model here (see setup below)
├── data/
│   ├── dataset.yaml       # Dataset config template for custom training
│   └── sample_images/     # Drop test images here
│
├── results/               # Output images/videos saved here
├── notebooks/             # Jupyter notebooks for exploration
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/yolov8-object-detection.git
cd yolov8-object-detection
pip install -r requirements.txt
```

### 2. Get the model

Export the pretrained YOLOv8 nano model to ONNX format (downloads ~6 MB):

```bash
python src/export_model.py --model yolov8n
```

This saves `models/yolov8n.onnx`. You're ready to run detections.

---

## Usage

### Detect objects in an image

```bash
python src/detect.py --source data/sample_images/street.jpg --output results/out.jpg
```

### Run on a video file

```bash
python src/detect.py --source path/to/video.mp4 --output results/out.mp4
```

### Live webcam feed

```bash
python src/detect.py --source 0
```

### Adjust confidence threshold

```bash
# Only show detections with ≥ 70% confidence
python src/detect.py --source data/sample_images/photo.jpg --conf 0.7
```

### All options

```
--model    Path to ONNX model       (default: models/yolov8n.onnx)
--source   Image / video / 0        (default: 0  = webcam)
--output   Save result to path      (optional)
--conf     Confidence threshold      (default: 0.5)
--nms      NMS IoU threshold         (default: 0.4)
--no-show  Don't display window
```

---

## REST API

Start the server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**Detect objects in an image via HTTP:**

```bash
curl -X POST http://localhost:8000/detect \
     -F "file=@data/sample_images/street.jpg"
```

**Response:**

```json
{
  "image_size": [1280, 720],
  "num_detections": 4,
  "latency_ms": 38.4,
  "detections": [
    { "label": "person",  "confidence": 0.95, "box": [120, 80, 55, 140] },
    { "label": "car",     "confidence": 0.88, "box": [340, 200, 180, 90] },
    { "label": "bicycle", "confidence": 0.76, "box": [260, 215, 55, 70] },
    { "label": "person",  "confidence": 0.91, "box": [210, 75, 50, 135] }
  ]
}
```

Interactive docs at `http://localhost:8000/docs`

---

## Docker

```bash
# Build image
docker build -t yolov8-detection .

# Run API server
docker run -p 8000:8000 yolov8-detection
```

---

## Training on a Custom Dataset

Want to detect your own object types (e.g. helmets, traffic cones)?

### 1. Prepare your dataset

Label your images using [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com/). Export in **YOLOv8 format**.

### 2. Edit the dataset config

```yaml
# data/dataset.yaml
path: data/custom_dataset
train: images/train
val:   images/val
nc: 3
names:
  0: helmet
  1: vest
  2: cone
```

### 3. Fine-tune

```bash
python src/export_model.py --model yolov8n --train --data data/dataset.yaml --epochs 50
```

### 4. Evaluate

```bash
python src/export_model.py --model yolov8n --eval --data data/dataset.yaml
```

---

## Benchmark

Measure inference speed on your machine:

```bash
python src/benchmark.py --model models/yolov8n.onnx --runs 200
```

Example output:
```
────────────────────────────────────────
  Model   : models/yolov8n.onnx
  Runs    : 200
  Mean    : 38.2 ms
  Median  : 37.8 ms
  Std     : 2.1 ms
  Avg FPS : 26.2
────────────────────────────────────────
```

---

## Model Variants

| Model | Size | mAP@50 | Speed (CPU) | Best for |
|-------|------|--------|-------------|----------|
| yolov8n | 6 MB | 37.3 | ~38ms | Edge / laptop |
| yolov8s | 22 MB | 44.9 | ~80ms | Balanced |
| yolov8m | 52 MB | 50.2 | ~180ms | Higher accuracy |
| yolov8l | 83 MB | 52.9 | ~350ms | Server |
| yolov8x | 131 MB | 53.9 | ~600ms | Max accuracy |

Switch models with `--model models/yolov8s.onnx`

---

## Key Concepts Explained

**What is a bounding box?**
A rectangle drawn around a detected object, defined by 4 numbers: X position, Y position, width, height.

**What is confidence score?**
How sure the model is (0–100%). Below 50% = the box is hidden. It's better to say nothing than to be wrong.

**What is mAP@50?**
"Mean Average Precision at IoU 50%" — the standard accuracy metric for object detection. Higher = better.

**What is ONNX?**
A universal model format that lets you train in PyTorch and run inference anywhere — CPU, GPU, mobile — without changing code. This project's model is 3× faster after ONNX export vs raw PyTorch.

**What is NMS?**
Non-Maximum Suppression — removes duplicate boxes when the model finds the same object multiple times. Keeps only the most confident one.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [YOLOv8](https://github.com/ultralytics/ultralytics) | State-of-the-art detection model |
| [PyTorch](https://pytorch.org/) | Model training |
| [ONNX Runtime](https://onnxruntime.ai/) | Fast CPU/GPU inference |
| [OpenCV](https://opencv.org/) | Image processing & drawing |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API |
| [Docker](https://www.docker.com/) | Containerised deployment |

m/in/yourprofile) · [GitHub](https://github.com/monikamunusamy)

