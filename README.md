# Wafer Defect Classification C++ Inference Engine

A native C++ inference pipeline for semiconductor wafer defect classification using OpenCV and ONNX Runtime.

This project deploys a CNN trained with PyTorch on the WM-811K wafer map dataset. The trained model is exported to ONNX format and executed in a native C++ inference environment.

The project demonstrates an end-to-end machine learning deployment workflow:

```text
PyTorch Model Training
          |
          v
      ONNX Export
          |
          v
C++ Inference Deployment
          |
          v
Performance Benchmark
```

---

# Overview

The CNN model was trained using PyTorch to classify semiconductor wafer defect patterns.

After training, the best-performing model was exported to ONNX format and deployed using a C++ inference application.

The C++ pipeline reproduces the preprocessing steps used during Python inference:

```text
Input Wafer Image
        |
        v
OpenCV Preprocessing
(grayscale → float32 → normalization → resize 24x24)
        |
        v
Tensor [1, 1, 24, 24]
        |
        v
ONNX Runtime Inference
        |
        v
Wafer Defect Classification
```

The deployment pipeline was also validated by running the same ONNX model and input image through both Python and C++ ONNX Runtime environments.

---

# Architecture

```text
                PyTorch CNN
                     |
                     | ONNX Export
                     v
          wafer_fault_cnn.onnx
                     |
                     v
        +-------------------------+
        |   C++ Inference Engine  |
        +-------------------------+
                     |
          +----------+----------+
          |                     |
          v                     v
      OpenCV              ONNX Runtime
   Preprocessing            Inference
          |                     |
          +----------+----------+
                     |
                     v
          Wafer Defect Prediction
```

---

# Features

- C++17 inference application
- OpenCV-based image preprocessing
- ONNX Runtime inference
- CMake build system
- Native deployment of a PyTorch-trained CNN
- Python/C++ numerical consistency validation
- ONNX Runtime inference benchmarking
- End-to-end pipeline benchmarking
- 8-class wafer defect classification

---

# Inference Pipeline

## Input

```text
Wafer image (.png)
```

## Preprocessing

The C++ preprocessing pipeline consists of:

```text
Grayscale Conversion
        |
        v
Convert to float32
        |
        v
Normalize values to 0-1
        |
        v
Resize to 24 x 24
        |
        v
Create Tensor [1, 1, 24, 24]
```

## Model Input

```text
[1, 1, 24, 24]
```

## Output

The model predicts one of the following eight wafer defect classes:

```text
0 -> Center
1 -> Donut
2 -> Edge-Loc
3 -> Edge-Ring
4 -> Loc
5 -> Near-full
6 -> Random
7 -> Scratch
```

---

# Project Structure

```text
wafer-cpp-inference/

├── src/
│   └── main.cpp
│
├── models/
│   └── wafer_fault_cnn.onnx
│
├── results/
│   └── test_wafer.png
│
├── images/
│   └── inference_result.png
│
├── benchmarks/
├── CMakeLists.txt
└── README.md
```

---

# Environment

- C++17
- OpenCV 4.14
- ONNX Runtime 1.28.1
- CMake 3.20+
- Visual Studio 2022

---

# Build

From the project root:

```bash
cmake -S . -B build
```

Build the Debug configuration:

```bash
cmake --build build --config Debug
```

The executable is generated at:

```text
build/Debug/wafer_inference.exe
```

---

# Run

From the project root:

```bash
build\Debug\wafer_inference.exe
```

The application loads the test wafer image from:

```text
results/test_wafer.png
```

and the ONNX model from:

```text
models/wafer_fault_cnn.onnx
```

---

# Example Inference Result

The deployed C++ application successfully loaded the updated ONNX model and produced the following output:

```text
[SUCCESS] Image loaded: results/test_wafer.png

[SUCCESS] ONNX model loaded!

========================================
       Wafer Fault Prediction
========================================

Class 0 (Center): -1.92753
Class 1 (Donut): -1.52871
Class 2 (Edge-Loc): 0.0786493
Class 3 (Edge-Ring): -5.8068
Class 4 (Loc): 3.14454
Class 5 (Near-full): -16.1514
Class 6 (Random): -6.97627
Class 7 (Scratch): 2.31458

Predicted class: 4 (Loc)

========================================
```

![Inference Result](images/inference_result.png)

---

# Performance Benchmark

Inference performance was measured using two separate benchmarks.

## Benchmark Configuration

- Runtime: ONNX Runtime C++
- Input shape: `1 × 1 × 24 × 24`
- Warm-up runs: 100
- Benchmark runs: 1000

## Results

| Metric | Result |
|---|---:|
| ONNX Runtime latency | 0.031566 ms |
| ONNX Runtime throughput | 31,679.7 FPS |
| End-to-End latency | 0.334094 ms |
| End-to-End throughput | 2,993.17 FPS |

## ONNX Runtime Benchmark

The ONNX Runtime benchmark measures neural network execution only:

```text
Tensor Input
      |
      v
ONNX Runtime
      |
      v
Model Output
```

### Result

```text
Average latency: 0.031566 ms
FPS: 31679.7
```

## End-to-End Pipeline Benchmark

The end-to-end benchmark includes the complete inference pipeline:

```text
Image Loading
      |
      v
OpenCV Preprocessing
      |
      v
Tensor Creation
      |
      v
ONNX Runtime Inference
      |
      v
Prediction
```

### Result

```text
Average latency: 0.334094 ms
FPS: 2993.17
```

The separation between inference-only and end-to-end measurements helps distinguish neural network execution time from the overhead introduced by image preprocessing and tensor preparation.

---

# Numerical Consistency Validation

The updated ONNX model was evaluated using both Python and C++ ONNX Runtime with the same input wafer image.

## Prediction Comparison

| Runtime | Prediction |
|---|---|
| Python ONNX Runtime | Class 4 (Loc) |
| C++ ONNX Runtime | Class 4 (Loc) |

Both environments produced the same predicted class.

## Python ONNX Runtime

```text
Class 0 (Center): -1.92753243
Class 1 (Donut): -1.52870536
Class 2 (Edge-Loc): 0.07864933
Class 3 (Edge-Ring): -5.80679607
Class 4 (Loc): 3.14454126
Class 5 (Near-full): -16.15137672
Class 6 (Random): -6.97626686
Class 7 (Scratch): 2.31458020

Predicted class: 4 (Loc)
```

## C++ ONNX Runtime

```text
Class 0 (Center): -1.92753
Class 1 (Donut): -1.52871
Class 2 (Edge-Loc): 0.0786493
Class 3 (Edge-Ring): -5.8068
Class 4 (Loc): 3.14454
Class 5 (Near-full): -16.1514
Class 6 (Random): -6.97627
Class 7 (Scratch): 2.31458

Predicted class: 4 (Loc)
```

The outputs match to the displayed precision, and both runtimes produce the same prediction.

The Python and C++ preprocessing pipelines were also verified using the same input image and produced matching normalized pixel values for the inspected samples.

---

# Deployment Validation

The deployment process was validated across the complete model conversion pipeline:

```text
PyTorch Training
      |
      v
Best Model
      |
      v
ONNX Export
      |
      v
Python ONNX Runtime
      |
      | Same model + same input
      v
C++ ONNX Runtime
      |
      v
Matching Prediction
```

This confirms that the exported ONNX model can be executed successfully in the native C++ environment while preserving the classification result observed in Python.

---

# Related Project

## Model Training Repository

The PyTorch training, validation, test evaluation, and ONNX export pipeline is available here:

https://github.com/lucy980509/wafer-defect-classification
