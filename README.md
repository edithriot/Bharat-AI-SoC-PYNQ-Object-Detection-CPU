# Bharat-AI-SoC-PYNQ-Object-Detection-CPU

## CPU-Only Object Detection (Baseline)

A software-only object detection pipeline is implemented on the Arm Cortex-A9 processor of the PYNQ-Z2 board using Python and OpenCV. This implementation serves as the performance baseline for comparison against the FPGA-accelerated design.

The baseline system uses a Histogram of Oriented Gradients (HOG) feature extractor combined with a pre-trained Support Vector Machine (SVM) classifier for person detection. The pipeline includes image loading, preprocessing, detection, non-maximum suppression, and result visualization.

All computation is performed on the CPU without hardware acceleration.
