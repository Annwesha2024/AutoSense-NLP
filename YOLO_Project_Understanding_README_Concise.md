# Analysis of Real-Time Object Detection Using YOLO

## 1. Project Overview

This project analyzes YOLO11n for real-time object detection in traffic scenes and compares its CPU inference speed with SSD300 and Faster R-CNN.

The project contains two experiments:

1. YOLO11n detection on a real traffic video.
2. YOLO11n validation and three-model speed benchmarking using COCO8.

---

## 2. Objectives

- Detect vehicles and pedestrians in traffic video.
- Measure YOLO11n FPS, inference time and confidence.
- Evaluate YOLO11n using COCO8 ground-truth annotations.
- Compare CPU inference speed of YOLO11n, SSD300 and Faster R-CNN.
- Generate graphs and CSV results.
- Analyze the suitability of YOLO for real-time object detection.

---

## 3. System Flow

```text
Traffic Video
     |
     v
Frame Extraction
     |
     v
Preprocessing
     |
     v
YOLO11n Detection
     |
     +----------------------+
     |          |           |
     v          v           v
Classes     Confidence     Boxes
     |          |           |
     +----------+-----------+
                |
                v
       FPS / Object Analysis
                |
                v
          CSV + Graphs
                |
                v
       Annotated Video
```

### Model Benchmark Flow

```text
COCO8 Validation Images
          |
          v
+---------+---------+
|         |         |
YOLO11n  SSD300  Faster R-CNN
|         |         |
+---------+---------+
          |
          v
 Inference Time / FPS
          |
          v
   Model Comparison
```

---

## 4. Models

### YOLO11n
Main model used for traffic detection and YOLO validation.

- Parameters: 2,616,248
- GFLOPs: 6.5
- Framework: Ultralytics

### SSD300
Pretrained Torchvision SSD300 VGG16 model used for CPU speed benchmarking.

### Faster R-CNN
Pretrained Torchvision Faster R-CNN ResNet50 FPN V2 model used for CPU speed benchmarking.

---

## 5. Dataset and Input

### Traffic Video

- File: `input/traffic.mp4`
- Resolution: 3840 × 2160
- Original FPS: 30
- Frames: 371
- Duration: approximately 12.4 seconds

### COCO8

The YOLO validation experiment used:

- 4 validation images
- 17 object instances

Because the validation set is very small, these accuracy results should be treated as a small experimental validation, not a general accuracy estimate.

---

## 6. Environment

| Component | Configuration |
|---|---|
| OS | Windows 11 |
| Python | 3.12.0 |
| CPU | 13th Gen Intel Core i5-13420H |
| RAM | ~15.64 GB |
| GPU | Not used |
| Framework | Ultralytics / PyTorch |
| Libraries | OpenCV, Pandas, Matplotlib, Torchvision |

All model speed measurements were performed using CPU inference.

---

## 7. Project Structure

```text
yolo/
├── input/
│   └── traffic.mp4
├── output/
│   └── yolo_detection.mp4
├── results/
│   ├── performance.csv
│   ├── model_comparison.csv
│   ├── fps_analysis.png
│   ├── object_counts.png
│   ├── confidence_analysis.png
│   ├── model_speed_comparison.png
│   ├── inference_time_comparison.png
│   └── yolo_accuracy_metrics.png
├── main.py
├── benchmark_models.py
├── generate_results.py
├── requirements.txt
└── yolo11n.pt
```

---

## 8. Running the Project

Activate the environment:

```powershell
venv\Scriptsctivate
```

Run traffic detection:

```powershell
python main.py
```

Run YOLO validation:

```powershell
yolo val model=yolo11n.pt data=coco8.yaml imgsz=640 batch=1 device=cpu
```

Run SSD/Faster R-CNN benchmark:

```powershell
python benchmark_models.py
```

Generate graphs and CSV:

```powershell
python generate_results.py
```

---

## 9. Results

### Traffic Video — YOLO11n

| Metric | Result |
|---|---:|
| Frames processed | 371 |
| Average FPS | 18.52 |
| Average inference time | 95.71 ms |
| Average confidence | 0.65 |

### YOLO11n — COCO8

| Metric | Result |
|---|---:|
| Precision | 57.4% |
| Recall | 85.0% |
| mAP@50 | 81.4% |
| mAP@50–95 | 60.8% |
| Inference | 87.3 ms/image |

### Model Speed Comparison

| Model | Inference Time | FPS |
|---|---:|---:|
| YOLO11n | 87.30 ms | 11.46 |
| SSD300 | 1219.08 ms | 0.82 |
| Faster R-CNN | 5148.35 ms | 0.19 |

These speed results are specific to the CPU and software environment used in the experiment.

---

## 10. Important Interpretation

The project demonstrates that YOLO11n can continuously detect objects in a traffic video.

The traffic-video experiment achieved **18.52 FPS**.

The controlled benchmark measured YOLO11n at **11.46 FPS**, SSD300 at **0.82 FPS**, and Faster R-CNN at **0.19 FPS**.

The two YOLO FPS values come from different experiments and should not be treated as the same measurement.

The COCO8 accuracy evaluation is based on only 4 validation images and 17 instances.

SSD300 and Faster R-CNN were experimentally compared for speed only. Their precision, recall and mAP were not calculated in this project.

---

## 11. Outputs

The project generates:

```text
output/yolo_detection.mp4
results/performance.csv
results/model_comparison.csv
results/fps_analysis.png
results/object_counts.png
results/confidence_analysis.png
results/model_speed_comparison.png
results/inference_time_comparison.png
results/yolo_accuracy_metrics.png
```

---

## 12. Limitations

- COCO8 validation set is very small.
- Only one traffic video was used.
- Experiments were CPU-only.
- SSD300 and Faster R-CNN accuracy was not independently evaluated.
- The project performs object detection only.
- No lane detection, tracking, depth estimation, path planning or vehicle control is implemented.

---

## 13. Report Structure

The final report follows:

1. Title Page
2. Abstract
3. Introduction
4. Background / Literature
5. Tools & Environment
6. System Design / Methodology
7. Implementation / Modelling
8. Testing & Results
9. Discussion
10. Conclusion & Future Scope
11. References
12. Appendix

---

## 14. Final Project Summary

The project implements and evaluates a real-time object-detection pipeline using YOLO11n.

The main experimental results are:

```text
Traffic Video:
18.52 FPS average

YOLO11n COCO8:
Precision    = 57.4%
Recall       = 85.0%
mAP@50       = 81.4%
mAP@50-95    = 60.8%

CPU Speed:
YOLO11n      = 11.46 FPS
SSD300       = 0.82 FPS
Faster R-CNN = 0.19 FPS
```

The results provide a practical software-based analysis of real-time object detection for autonomous-vehicle perception.
