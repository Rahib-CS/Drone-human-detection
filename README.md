# 🚁 Drone Human Detection & Counting System


---

## A Personal Note Before We Begin

This project was a genuinely enjoyable challenge for me — and a meaningful step outside my usual domain.

My background in deep learning has been primarily in **graph-based neural networks** and **DL for wireless/communication systems** — Using GNNs for network topology modelling, or deep learning applied to signal processing and channel estimation problems. Where "data" was formatted into adjacency matrices, node features, or complex-valued signals — not pixels.

This was my **first time using OpenCV**. Going from thinking about graphs and tensors of communication signals to actually drawing bounding boxes on drone imagery, managing BGR vs RGB colour channels, and rendering text overlays on frames — it was an interesting experience for me. 

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset — VisDrone2019](#dataset--visdrone2019)
4. [Task 01 — Dataset Understanding & Preprocessing](#task-01--dataset-understanding--preprocessing)
5. [Task 02 — Model Training](#task-02--model-training)
6. [Task 03 — Detection & Human Counting](#task-03--detection--human-counting)
7. [Task 04 — Object Tracking with ByteTrack (Bonus)](#task-04--object-tracking-with-bytetrack-bonus)
8. [Task 05 — Evaluation & Visualization](#task-05--evaluation--visualization)
9. [Results Summary](#results-summary)
10. [Strengths, Limitations & Future Work](#strengths-limitations--future-work)
11. [How to Run](#how-to-run)
12. [Environment & Dependencies](#environment--dependencies)

---

## Project Overview

This project implements a full **computer vision pipeline** for analyzing drone/aerial imagery. The system detects humans and cars, counts the total number of humans per frame, visualizes the results, and optionally tracks objects across consecutive frames using ByteTrack.

| Property | Detail |
|---|---|
| **Model** | YOLOv8n (fine-tuned) |
| **Tracker** | ByteTrack (built into Ultralytics) |
| **Dataset** | VisDrone2019 |
| **Training** | Google Colab T4 GPU, 25 epochs |
| **Inference** | CPU (no GPU required) |
| **Classes** | `human` (0), `car` (1) |

---

## Repository Structure

```
drone-human-detection/
│
├── notebooks/
│   ├── Drone_Part1_DataPrep.ipynb        # Task 01 — Dataset understanding & preprocessing
│   ├── Drone_Part2_Training.ipynb        # Task 02 — Model training & evaluation
│   └── Drone_Part3_Inference.ipynb       # Task 03, 04, 05 — Detection, tracking, visualization
│
├── outputs/
│   ├── sample_detections_grid.png        # 6-image detection grid
│   ├── detect_*.jpg                      # Individual annotated images
│   ├── tracking_0000289.mp4              # ByteTrack tracking video
│   └── evaluation_metrics.png           # Metrics bar chart
│
├── data.yaml                             # YOLOv8 dataset config
└── README.md
```

> **Note:** Model weights (`best.pt`, 5.9 MB) and the full dataset are stored on Google Drive due to file size limits. Download links are provided in the [How to Run](#how-to-run) section.

---

## Dataset — VisDrone2019

**Source:** [VisDrone2019 on Kaggle](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

VisDrone is a large-scale benchmark collected by drones across 14 cities in China, covering diverse scenes — urban roads, parks, markets, residential areas — under varying lighting and weather conditions.

### Original Dataset Statistics

| Split | Images | Annotations |
|---|---|---|
| Train | 6,471 | ~390,000 |
| Val | 548 | ~33,000 |

### Original 10 Classes

`pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`

---

## Task 01 — Dataset Understanding & Preprocessing

### Class Remapping

VisDrone's 10 classes were reduced to 2 for this task:

| Original Class | Mapped To | Reason |
|---|---|---|
| `pedestrian` | `human` (0) | Both represent individual people on foot |
| `people` | `human` (0) | Crowd/group annotations, same semantic meaning |
| `car` | `car` (1) | Direct match |
| All others | **Dropped** | Out of scope for this assessment |

This remapping is non-trivial. VisDrone distinguishes `pedestrian` (individuals) from `people` (groups/crowds) — merging them into a single `human` class simplifies the counting task without losing detection capability.

### Label Format Conversion

VisDrone annotations use the format:
```
x_min, y_min, width, height, score, category_id, truncation, occlusion
```

These were converted to **YOLO format** (normalized, centre-based):
```
class_id  cx  cy  w  h
```

### Filtering

Images with zero valid labels after remapping (i.e., containing only dropped classes) were removed:

| Split | Before | After |
|---|---|---|
| Train | 6,471 | 6,468 |
| Val | 548 | 548 |

### Data Augmentation (Applied during Training)

YOLOv8's default augmentation pipeline was used:
- **Mosaic** — combines 4 images into one, improving small object recall
- **Random horizontal flip** — orientation invariance
- **HSV jitter** — lighting/colour robustness
- **Scale & translate** — viewpoint variation

These are especially important for VisDrone because drone altitude and angle vary significantly between images.

### Key Challenges Observed

1. **Extreme small object density** — pedestrians often appear as 5–15 pixel blobs. At standard resolution this is below the feature resolution of most backbone stages.
2. **Heavy occlusion** — crowded scenes (markets, crossings) have severe overlap between people.
3. **Class imbalance** — `human` annotations vastly outnumber `car` annotations in pedestrian-heavy scenes.
4. **Varying altitude** — the same physical object can appear at wildly different scales depending on drone height.

---

## Task 02 — Model Training

### Model Choice — YOLOv8n

**YOLOv8n** (nano) was selected for the following reasons:

- **Lightweight** (5.9 MB, ~3.2M parameters) — runs comfortably on CPU for inference
- **Fast training** — critical given Colab T4 GPU time limits
- **Strong small-object baseline** — YOLOv8's anchor-free head handles variable-scale objects better than earlier YOLO versions
- **ByteTrack integration** — tracking is built in via `model.track()`, requiring no separate setup

### Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Epochs | 25 | Colab T4 GPU quota constraint |
| Image size | 416 | Balance between resolution and speed |
| Batch size | 32 | Fills T4 VRAM efficiently |
| Optimizer | AdamW (default) | Stable convergence for fine-tuning |
| Learning rate | Auto (Ultralytics default) | Cosine warmup + decay |
| Pretrained weights | `yolov8n.pt` (COCO) | Transfer learning from general object detection |

### Why `best.pt` and Not `last.pt`

The checkpoint saved at the epoch with the **highest validation mAP** is used — not the final epoch. Training loss can still decrease while validation mAP plateaus or dips due to overfitting. `best.pt` avoids this by capturing the generalization peak.

### Training Time

**0.681 hours** on Google Colab T4 GPU.

---

## Task 03 — Detection & Human Counting

### Inference Pipeline

For each input image:

1. Run `model.predict()` at `conf ≥ 0.25`
2. Iterate over all detected boxes
3. Draw colour-coded bounding boxes:
   - 🟢 **Green** — human
   - 🔵 **Blue** — car
4. Display confidence score on each box
5. Count all `class_id == 0` detections → human count
6. Overlay human count in the top-left corner
7. Save annotated image to Drive

### Confidence Threshold — Why 0.25?

`0.25` is YOLO's standard default. For aerial imagery with tiny objects:
- **Lower threshold** → catches more humans but increases false positives
- **Higher threshold** → cleaner results but misses small/occluded people

`0.25` strikes the right balance for a first deployment. In production, this would be tuned using a precision-recall curve on the validation set.

### Sample Detection Output

The pipeline was run on 6 randomly sampled validation images. Human counts ranged from **6 to 55 per image**, reflecting VisDrone's diversity of scenes — from sparse suburban roads to dense pedestrian plazas.

---

## Task 04 — Object Tracking with ByteTrack (Bonus)

### Why Tracking Adds Value

Per-frame detection counts every person independently in every frame. If a person appears across 35 frames, they are "detected" 35 times. **Tracking assigns persistent IDs** — the same person keeps the same ID as they move through the scene, enabling:

- Counting **unique individuals** rather than raw detections
- **Trajectory analysis** — where are people moving?
- **Re-identification** after brief occlusion

### ByteTrack — How It Works

ByteTrack is a multi-object tracker that unlike earlier trackers (SORT, DeepSORT) **does not discard low-confidence detections**. It associates both high- and low-confidence detections with existing tracks, significantly improving recall in crowded or partially occluded scenes — exactly the conditions VisDrone presents.

### Implementation

```python
results = model.track(
    source=frame_path,
    conf=0.25,
    tracker='bytetrack.yaml',  # bundled with Ultralytics
    device='cpu',
    persist=True,              # CRITICAL: maintains track state across frames
    verbose=False
)
```

`persist=True` is the single most important parameter. Without it, ByteTrack resets its internal state between calls, generating new IDs every frame — defeating the purpose of tracking entirely.

### Sequence Selection

The validation sequence with the **most consecutive frames** (`0000289`, 35 frames) was selected for the tracking demo. More frames = more meaningful demonstration of ID persistence.

### Visual Design

Each track ID is assigned a **deterministic unique colour** using an MD5 hash of the track ID. This ensures:
- The same object always has the same colour
- Colours are visually distinct
- No manual colour assignment is needed

### Output

A **5 FPS MP4 video** (`tracking_0000289.mp4`) is saved to Google Drive. VisDrone frames are sparsely sampled (not true 30fps video), so 5fps gives a natural viewing pace without appearing too fast or too slow.

---

## Task 05 — Evaluation & Visualization

### Metrics from Colab 2 Training

| Class | Precision | Recall | mAP@50 |
|---|---|---|---|
| Human | 0.400 | 0.300 | 0.236 |
| Car | 0.650 | 0.580 | 0.599 |
| **Overall** | **0.525** | **0.440** | **0.418** |

### Interpreting the Results

**Car mAP@50 = 0.599** — good result. Vehicles are large, structurally consistent, and appear at predictable scales from drone altitude.

**Human mAP@50 = 0.236** — lower, but expected and explainable:
- Pedestrians are 5–15 pixels tall at typical VisDrone altitude
- They are heavily occluded in crowd scenes
- The `human` class merges two distinct VisDrone classes (`pedestrian` + `people`) which have slightly different visual signatures
- Only 25 training epochs — insufficient for the model to fully converge on small-object features

**Overall mAP@50 = 0.418** — reasonable for a nano model, 25 epochs, on one of the most challenging aerial detection benchmarks publicly available.

---

## Results Summary

| Task | Deliverable | Status |
|---|---|---|
| Dataset preprocessing | Label remapping, YOLO conversion, filtering | ✅ |
| Model training | YOLOv8n, 25 epochs, T4 GPU | ✅ |
| Detection + counting | Bounding boxes + human count overlay | ✅ |
| Object tracking | ByteTrack MP4 video with persistent IDs | ✅ |
| Evaluation | mAP@50, Precision, Recall bar chart | ✅ |

---

## Strengths, Limitations & Future Work

### Strengths

- **End-to-end pipeline** — from raw VisDrone annotations to annotated video, fully reproducible in 3 Colab notebooks
- **Lightweight deployment** — `best.pt` is 5.9 MB and runs on CPU, no GPU required at inference time
- **ByteTrack bonus** — zero additional training required; integrated via Ultralytics in a clean, explainable way
- **Honest evaluation** — metrics are reported as-is without cherry-picking; limitations are explicitly discussed

### Limitations

- **25 epochs** is insufficient for full convergence on VisDrone's small objects; mAP would improve significantly with longer training
- **Human counting is per-frame** — the current counter resets each image. True unique-person counting across a video would require maintaining a set of seen track IDs
- **No SAHI** — Slicing Aided Hyper Inference would significantly improve small-object detection by tiling images before inference

### Future Improvements

| Improvement | Expected Impact |
|---|---|
| Train 100+ epochs with cosine LR | Higher mAP convergence |
| Use `imgsz=640` | Better small-object resolution |
| Apply SAHI tiling at inference | Large improvement in human recall |
| Upgrade to YOLOv8s or YOLOv8m | Higher capacity, better feature extraction |
| Cross-frame unique ID counting | True person counting in video |
| Export to ONNX/TensorRT | Real-time inference on edge devices |

---

## How to Run

### Prerequisites

- Google Account with Google Drive
- Google Colab (free tier works; T4 GPU recommended for training)

### Step 1 — Dataset

Download the preprocessed dataset from Google Drive:

> **[visdrone_yolo.zip — Google Drive Link]** *(add your link here)*

Place it in `MyDrive/Antlings_Drone_Project/visdrone_yolo.zip`.

### Step 2 — Run Notebooks in Order

| Notebook | Purpose | Runtime |
|---|---|---|
| `Drone_Part1_DataPrep.ipynb` | Download raw VisDrone, remap classes, convert labels | CPU, ~15 min |
| `Drone_Part2_Training.ipynb` | Train YOLOv8n for 25 epochs | T4 GPU, ~41 min |
| `Drone_Part3_Inference.ipynb` | Detection, counting, tracking, evaluation | CPU, ~10 min |

Each notebook mounts Google Drive at the start. All outputs are automatically saved to `MyDrive/Antlings_Drone_Project/outputs/`.

### Step 3 — View Outputs

After running all notebooks, your Drive will contain:

```
Antlings_Drone_Project/
├── best.pt                          # trained model weights
├── outputs/
│   ├── sample_detections_grid.png   # detection results
│   ├── tracking_0000289.mp4         # ByteTrack video
│   └── evaluation_metrics.png       # metrics chart
```

---

## Environment & Dependencies

```
Python          3.10+
ultralytics     8.x     (YOLOv8 + ByteTrack)
opencv-python-headless  (image/video I/O, annotation drawing)
numpy           1.24+
matplotlib      3.7+
torch           2.0+    (CPU inference)
```

Install in Colab:
```bash
pip install ultralytics opencv-python-headless --quiet
```

All other dependencies (`numpy`, `matplotlib`, `torch`) are pre-installed in the Colab environment.

---


