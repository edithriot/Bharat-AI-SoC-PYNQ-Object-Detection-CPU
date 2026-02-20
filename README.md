# Bharat AI-SoC Student Challenge  
## CPU-Only Object Detection Baseline on Xilinx PYNQ-Z2

---

## Overview

This project implements a complete **CPU-only object detection pipeline** on the PYNQ-Z2 platform under **Problem Statement 5** of the Bharat AI-SoC Student Challenge.  
The system runs entirely on the **Arm Cortex-A9 (Processing System)** of the **Zynq-7020 SoC**, without any FPGA acceleration.

This implementation serves as a **software reference baseline** for future hardware-accelerated (FPGA-based) designs.

---

## Objectives

- Develop a pedestrian detection system on embedded Linux  
- Execute the entire detection pipeline on the CPU  
- Measure latency and throughput on real hardware  
- Establish a quantitative performance baseline  

---

## Target Platform

- **Board**: Xilinx PYNQ-Z2  
- **SoC**: Zynq-7020  
- **Processor**: Dual-core Arm Cortex-A9  
- **Environment**: PYNQ Linux + Jupyter Notebook  

---

## Software Stack

- Python 3  
- OpenCV  
- NumPy  
- Matplotlib  
- PYNQ Framework  

---

## Detection Method

The system uses a classical computer vision approach:

1. Image loading  
2. Image resizing (aspect ratio preserved)  
3. HOG feature extraction  
4. SVM-based pedestrian classification  
5. Non-Maximum Suppression (NMS)  
6. Bounding box visualization  
7. Performance benchmarking  

All computation runs on the CPU.

---

# Complete CPU Object Detection Implementation

```python
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================
# Dataset Loading
# ==============================

img_dir = "/home/xilinx/jupyter_notebooks/images"

image_paths = sorted(
    glob.glob(img_dir + "/*.jpg") +
    glob.glob(img_dir + "/*.png") +
    glob.glob(img_dir + "/*.jpeg")
)

print("Total images:", len(image_paths))

# ==============================
# HOG + SVM Initialization
# ==============================

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ==============================
# Non-Maximum Suppression
# ==============================

def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes).astype("float")
    pick = []

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(
            idxs,
            np.concatenate(([len(idxs)-1], np.where(overlap > overlapThresh)[0]))
        )

    return boxes[pick].astype("int")

# ==============================
# Performance Tracking
# ==============================

latencies = []
total_start = time.time()

# ==============================
# Detection Loop
# ==============================

for img_path in image_paths:

    img = cv2.imread(img_path)
    if img is None:
        continue

    # Resize while preserving aspect ratio
    h, w = img.shape[:2]
    scale = 640.0 / w
    img = cv2.resize(img, (640, int(h * scale)))

    # Detection timing start
    start = time.time()

    rects, weights = hog.detectMultiScale(
        img,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.03
    )

    # Confidence filtering
    boxes = []
    for (x, y, w_box, h_box), weight in zip(rects, weights):
        if weight > 0.6:
            boxes.append([x, y, x + w_box, y + h_box])

    # Apply NMS
    final_boxes = non_max_suppression(boxes, overlapThresh=0.4)

    # Detection timing end
    end = time.time()
    latency = end - start
    latencies.append(latency)

    # Visualization
    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Person", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ==============================
# Performance Summary
# ==============================

total_end = time.time()
latencies = np.array(latencies)

avg_latency = latencies.mean()
min_latency = latencies.min()
max_latency = latencies.max()
fps = 1.0 / avg_latency
total_time = total_end - total_start

print("\n===== CPU-ONLY PERFORMANCE (PYNQ-Z2) =====")
print(f"Total images processed : {len(latencies)}")
print(f"Average latency        : {avg_latency*1000:.2f} ms")
print(f"Min latency            : {min_latency*1000:.2f} ms")
print(f"Max latency            : {max_latency*1000:.2f} ms")
print(f"Throughput (FPS)       : {fps:.2f}")
print(f"Total execution time   : {total_time:.2f} s")
```

---

## Results and Significance

This CPU-only implementation validates real-time object detection feasibility on the Arm Cortex-A9 of the PYNQ-Z2.

The measured metrics provide:

- Insight into CPU computational limitations  
- A quantitative reference for FPGA acceleration  
- A benchmarking foundation for future Vitis HLS-based hardware design  

This baseline will be used for comparison against upcoming FPGA-accelerated implementations.


## OUTPUTS

<img width="471" height="292" alt="Screenshot 2026-02-20 122338" src="https://github.com/user-attachments/assets/2c4d843c-6893-4bb5-902e-3fbbf7991c29" />
