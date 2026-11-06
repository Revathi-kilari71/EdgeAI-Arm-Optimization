# EdgeAI – Hardware-Aware Deep Learning Optimization

## Project Overview

EdgeAI is a machine learning software engineering project focused on optimizing lightweight deep learning models for efficient CPU-based edge inference.

The project uses **MobileNetV3** and **Fashion-MNIST** to investigate the impact of INT8 quantization on classification accuracy, model size, inference latency, and throughput.

The implementation covers model training, ONNX export, dynamic and static quantization, CPU benchmarking, Python and C++ inference comparison, and deployment validation.

**Evaluation environment:** Windows CPU using ONNX Runtime. Actual ARM64 hardware benchmarking is planned as future work.

## Project Objectives

- Train and evaluate a lightweight image classification model.
- Convert the trained PyTorch model into ONNX format.
- Implement dynamic and static INT8 quantization.
- Compare classification accuracy across FP32 and INT8 models.
- Measure inference latency, throughput, and model size.
- Investigate CPU thread optimization.
- Compare Python and C++ inference performance.
- Prepare and validate a portable deployment package.
- Extend benchmarking to ARM64 hardware.

## Technologies Used

- **Programming:** Python, C++
- **Deep Learning:** PyTorch, Torchvision, MobileNetV3
- **Dataset:** Fashion-MNIST
- **Optimization:** ONNX, ONNX Runtime, INT8 Quantization
- **Evaluation:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib
- **Development:** Visual Studio Code, Git, GitHub, Windows PowerShell

## Project Workflow

1. Dataset preparation and preprocessing
2. MobileNetV3 model training
3. Model evaluation
4. PyTorch to ONNX conversion
5. Dynamic INT8 quantization
6. Static INT8 quantization
7. Model accuracy comparison
8. CPU performance benchmarking
9. CPU thread optimization
10. Python and C++ inference comparison
11. Deployment packaging and validation

## Model Accuracy Comparison

The models were evaluated using the Fashion-MNIST test dataset containing 10,000 images.

| Model | Accuracy | Weighted F1 | Model Size (MiB) |
|---|---:|---:|---:|
| FP32 | 83.79% | 0.8380 | 5.84 |
| Dynamic INT8 | 83.84% | 0.8388 | 1.62 |
| Static INT8 | 84.03% | 0.8405 | 1.65 |

Static INT8 achieved the highest measured classification accuracy. The small accuracy differences do not establish that quantization consistently improves accuracy.

## Final CPU Benchmark Results

| Metric | FP32 | Static INT8 |
|---|---:|---:|
| Mean latency | 0.9832 ms | 0.8016 ms |
| P95 latency | 1.6887 ms | 1.1990 ms |
| Throughput | 1017.13 FPS | 1247.51 FPS |
| Model size | 5.84 MiB | 1.65 MiB |

### Optimization Results

- **Mean latency reduction:** 18.47%
- **Throughput improvement:** 22.65%
- **Model size reduction:** 71.78%
- **Static INT8 throughput speedup:** approximately 1.23×

These measurements represent the tested Windows CPU environment, not ARM64 hardware performance.

## CPU Thread Optimization

ONNX Runtime inference was evaluated using the following thread configurations:

- 1 thread
- 2 threads
- 4 threads
- 8 threads

Two threads produced the fastest measured configuration for both models in the thread optimization experiment. Results may differ across processors and runtime environments.

## Python vs C++ Inference Benchmark

The project compared Python and C++ inference performance using the tested ONNX Runtime setup.

| Implementation | Mean Latency |
|---|---:|
| Python | 0.9404 ms |
| C++ | 0.6166 ms |

**Measured C++ speedup: approximately 1.53×**

The comparison used 10 warm-up iterations and 100 measured runs. This result applies to the tested implementations and does not establish a universal C++ performance advantage.

## Deployment Validation

The Static INT8 ONNX model was selected for deployment validation.

Validation covered:

- ONNX model verification
- ONNX Runtime session initialization
- Input and output shape checking
- Single-image inference
- Batch inference
- Full test-set evaluation
- Deployment package preparation

### Full Test-Set Results

| Metric | Result |
|---|---:|
| Test images | 10,000 |
| Correct predictions | 8,403 |
| Incorrect predictions | 1,597 |
| Accuracy | 84.03% |
| Weighted F1 score | 0.8405 |

**Model input:** `[1, 3, 96, 96]`

**Model output:** `[1, 10]`

The deployment model successfully completed inference evaluation on the Fashion-MNIST test dataset.

## Installation

Clone the repository:

```powershell
git clone https://github.com/Revathi-kilari71/EdgeAI-Arm-Optimization.git
cd EdgeAI-Arm-Optimization
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running the Project

### Model Training

```powershell
python src/train_improved.py
```

### Model Evaluation

```powershell
python src/evaluate_model.py
```

### Model Accuracy Comparison

```powershell
python src/compare_accuracy.py
```

### Final CPU Benchmark

```powershell
python benchmarks/benchmark_final.py
```

### CPU Thread Optimization

```powershell
python benchmarks/benchmark_threads.py
```

### Deployment Preparation

```powershell
python benchmarks/deployment/prepare_deployment.py
```

### Deployment Validation

```powershell
python deployment/validate_deployment.py
```

**Note:** Some commands require Fashion-MNIST data and generated model files. Large datasets and model artifacts may need to be generated locally because they are excluded from the repository.

## Current Project Status

**Completed:**

- MobileNetV3 training and evaluation
- FP32 ONNX conversion
- Dynamic INT8 quantization
- Static INT8 quantization
- Model accuracy and F1 comparison
- CPU latency and throughput benchmarking
- CPU thread optimization
- Python and C++ inference benchmarking
- Deployment preparation and validation
- Git version control and GitHub publication

## Limitations

- Benchmarks were performed on a Windows CPU.
- ARM64 hardware inference has not yet been evaluated.
- Performance results depend on processor configuration and runtime settings.
- Fashion-MNIST is a relatively simple benchmark dataset and does not represent all real-world edge AI workloads.

## Future Improvements

- Evaluate inference on actual ARM64 hardware.
- Benchmark additional ARM-compatible inference backends.
- Investigate hardware-specific optimizations.
- Measure memory usage and power consumption.
- Test the optimized inference pipeline on more complex image classification datasets.

## Conclusion

This project demonstrates a practical workflow for converting and optimizing a lightweight deep learning model for CPU-based edge inference.

Static INT8 quantization reduced model size by approximately 71.78% and mean inference latency by 18.47% in the tested environment while maintaining approximately 84% classification accuracy.

The project combines deep learning, model optimization, ONNX Runtime, Python, C++, benchmarking, and deployment engineering to establish a foundation for future ARM64 edge AI evaluation.