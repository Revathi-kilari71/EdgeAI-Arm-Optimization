# EdgeAI – Hardware-Aware Deep Learning Optimization

## Project Overview

EdgeAI is a machine learning software engineering project focused on optimizing deep learning models for efficient edge inference.

The project uses MobileNetV3 and Fashion-MNIST to investigate the effects of model quantization on classification accuracy, model size, inference latency, and CPU throughput.

The implementation includes model training, ONNX conversion, dynamic and static INT8 quantization, performance benchmarking, and deployment validation.

The current experiments were performed on Windows using ONNX Runtime CPU Execution Provider. ARM64 hardware evaluation remains future work.

## Project Objectives

- Train and evaluate a lightweight image classification model.
- Export the trained model to ONNX format.
- Implement dynamic and static INT8 quantization.
- Compare FP32, Dynamic INT8, and Static INT8 models.
- Measure inference latency, throughput, and model size.
- Investigate CPU thread optimization.
- Prepare and validate a portable deployment package.
- Extend the implementation to ARM64 hardware and C++ inference.

## Technologies Used

- Python
- PyTorch and Torchvision
- MobileNetV3
- Fashion-MNIST
- ONNX and ONNX Runtime
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- Git and GitHub
- Windows PowerShell and Visual Studio Code

## Model Accuracy Comparison

| Model | Accuracy | F1 Score | Size |
|---|---:|---:|---:|
| FP32 | 83.79% | 0.8380 | 5.84 MB |
| Dynamic INT8 | 83.84% | 0.8388 | 1.62 MB |
| Static INT8 | 84.03% | 0.8405 | 1.65 MB |

Static INT8 achieved the highest measured classification accuracy, while Dynamic INT8 produced the smallest model file.

## Final CPU Benchmark Results

| Model | Mean Latency | P95 Latency | Throughput |
|---|---:|---:|---:|
| FP32 | 0.9832 ms | 1.6887 ms | 1017.13 FPS |
| Static INT8 | 0.8016 ms | 1.1990 ms | 1247.51 FPS |

The final benchmark reported:

- Static INT8 speedup: 1.227×
- Mean latency reduction: 18.47%
- Approximately 71.8% smaller model than FP32

These measurements represent performance on the tested Windows CPU environment and should not be interpreted as ARM64 hardware results.

## CPU Thread Optimization

The project evaluated different ONNX Runtime CPU thread configurations:

- 1 thread
- 2 threads
- 4 threads
- 8 threads

The thread optimization experiment identified two threads as the fastest tested configuration for each model in that experiment.

## Deployment Validation

The Static INT8 model was selected for deployment packaging.

Deployment validation included:

- ONNX model structure verification
- ONNX Runtime session creation
- Model input and output shape verification
- Metadata and requirements validation
- Single-image inference testing
- Batch inference testing
- Deployment ZIP package creation

### Deployment Batch Test

| Metric | Result |
|---|---:|
| Images tested | 100 |
| Correct predictions | 84 |
| Incorrect predictions | 16 |
| Accuracy | 84.00% |
| Weighted F1 Score | 0.8373 |

The deployment validation completed successfully.

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

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running the Project

Model training:

```powershell
python src/train_improved.py
```

Model evaluation:

```powershell
python src/evaluate_model.py
```

Final CPU benchmarking:

```powershell
python benchmarks/benchmark_final.py
```

CPU thread optimization:

```powershell
python benchmarks/benchmark_threads.py
```

Deployment preparation:

```powershell
python benchmarks/deployment/prepare_deployment.py
```

Deployment validation:

```powershell
python deployment/validate_deployment.py
```

**Note:** Training and model preparation require the appropriate dataset and generated model files. The benchmark and deployment commands depend on these files being available. Large model and dataset artifacts are excluded from the repository.

## Current Project Status

Completed:

- Model training and evaluation
- ONNX model export
- Dynamic INT8 quantization
- Static INT8 quantization
- Accuracy and F1 comparison
- CPU latency and throughput benchmarking
- CPU thread optimization
- Deployment package preparation
- Deployment inference and validation
- Git version control and repository publication