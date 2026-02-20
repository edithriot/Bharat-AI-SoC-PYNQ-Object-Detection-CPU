# Bharat AI-SoC Student Challenge  
## Object Detection on Xilinx PYNQ-Z2 (CPU Baseline Implementation)

---

## Project Context

This project is developed under **Problem Statement 5** of the **Bharat AI-SoC Student Challenge**, which focuses on real-time object detection on embedded platforms using heterogeneous SoCs.

As part of this work, a **CPU-only baseline object detection system** has been implemented and evaluated on the **Xilinx PYNQ-Z2 FPGA board**. This implementation serves as a **software reference model** and performance baseline for future hardware-accelerated designs.

---

## Objective

- Implement an object detection pipeline on the **Arm Cortex-A9 processor** of the PYNQ-Z2 board  
- Measure inference latency and throughput on embedded hardware  
- Establish a **quantitative baseline** for comparison with future FPGA-accelerated implementations  

---

## Target Platform

- **Board**: Xilinx PYNQ-Z2  
- **SoC**: Zynq-7020  
- **Processor**: Dual-core Arm Cortex-A9  
- **Execution Environment**: PYNQ Linux + Jupyter Notebook  

---

## Implementation Overview

The current implementation performs **object detection entirely on the CPU** using Python and OpenCV. The system detects pedestrians in images using a classical computer vision pipeline.

### Detection Pipeline

1. Load images from a dataset  
2. Resize images while preserving aspect ratio  
3. Extract features using **Histogram of Oriented Gradients (HOG)**  
4. Perform classification using a **pre-trained Support Vector Machine (SVM)**  
5. Apply **Non-Maximum Suppression (NMS)** to remove redundant detections  
6. Draw bounding boxes and visualize results  
7. Measure inference latency and throughput  

All stages are executed on the Arm processor without FPGA acceleration.

---

## Software Stack

- Python 3  
- OpenCV  
- NumPy  
- Matplotlib  
- Jupyter Notebook (PYNQ Framework)  

---

## Performance Measurement Methodology

- Inference latency is measured **only for the detection stage**, excluding visualization overhead  
- Timing is captured for each image independently  
- The following metrics are computed:
  - Average inference latency (ms)
  - Minimum latency (ms)
  - Maximum latency (ms)
  - Throughput (Frames Per Second – FPS)
  - Total execution time  

This methodology ensures consistent and fair benchmarking on embedded hardware.

---

# CPU Object Detection Implementation (Module-wise Explanation)

This section describes the complete CPU-only object detection pipeline implemented on the Xilinx PYNQ-Z2 platform using Python and OpenCV.

---

## 1. Library Imports and Dependencies

```python
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
```
# Explanation

cv2 – OpenCV for image processing and detection

glob – Load multiple images from directory

numpy – Numerical operations

matplotlib – Visualization

time – Performance measurement

## 2. Image Dataset Loading Module
```
img_dir = "/home/xilinx/jupyter_notebooks/images"

image_paths = sorted(
    glob.glob(img_dir + "/*.jpg") +
    glob.glob(img_dir + "/*.png") +
    glob.glob(img_dir + "/*.jpeg")
)

print("Total images:", len(image_paths))
```
```
img_dir = "/home/xilinx/jupyter_notebooks/images"

image_paths = sorted(
    glob.glob(img_dir + "/*.jpg") +
    glob.glob(img_dir + "/*.png") +
    glob.glob(img_dir + "/*.jpeg")
)

print("Total images:", len(image_paths))
```

# Explanation

Specifies dataset directory

Loads .jpg, .png, and .jpeg files

Sorting ensures repeatable benchmarking

## 3. HOG + SVM Detector Initialization

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

# Explanation

Initializes HOG feature extractor

Uses pre-trained SVM pedestrian detector

Core detection engine executed on CPU

## 4. Non-Maximum Suppression (NMS)

```
 
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
```
# Explanation

Removes overlapping bounding boxes

Retains highest-confidence detections

Improves detection quality

## 5. Performance Initialization

```

latencies = []
total_start = time.time()
```

Stores per-image latency and tracks total runtime.

## 6. Detection Loop

```
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
```

Iterates through dataset and skips invalid images.

## 7. Image Preprocessing

```
python

h, w = img.shape[:2]
scale = 640.0 / w
img = cv2.resize(img, (640, int(h * scale)))
```

## 8. Inference Timing
```
start = time.time()

rects, weights = hog.detectMultiScale(
    img,
    winStride=(8, 8),
    padding=(16, 16),
    scale=1.03
)
```

Timing begins only for detection stage.

## 9. Bounding Box Filtering

```
boxes = []
for (x, y, w, h), weight in zip(rects, weights):
    if weight > 0.6:
        boxes.append([x, y, x + w, y + h])

final_boxes = non_max_suppression(boxes, overlapThresh=0.4)
```
Filters low-confidence detections and applies NMS.

## 10. Latency Measurement

```
end = time.time()
latency = end - start
latencies.append(latency)
```
Stores per-image inference time.

## 11. Visualization

```
for (x1, y1, x2, y2) in final_boxes:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "Person", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

## 12. Performance Summary

```
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

Computes and prints:

Average latency

Min / Max latency

Throughput (FPS)

Total runtime

Results Summary

The CPU-only implementation successfully performs object detection on the PYNQ-Z2 platform. Performance metrics obtained from this baseline provide:

Insight into CPU computational limits

Reference for future FPGA-based acceleration
