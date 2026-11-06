
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <stdexcept>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EdgeAI");

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        Ort::Session session(
            env,
            L"deployment/mobilenetv3_static_int8.onnx",
            options
        );

        std::ifstream file("cpp/test_image.bin", std::ios::binary);

        if (!file) {
            throw std::runtime_error("Image file not found");
        }

        std::vector<float> input_data(3 * 96 * 96);

        file.read(
            reinterpret_cast<char*>(input_data.data()),
            input_data.size() * sizeof(float)
        );

        if (!file) {
            throw std::runtime_error("Invalid image data");
        }

        std::vector<int64_t> shape = {1, 3, 96, 96};

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            shape.data(),
            shape.size()
        );

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};

        const int warmup_runs = 10;
        const int benchmark_runs = 100;

        std::cout << "\nEDGEAI C++ INFERENCE BENCHMARK\n";

        for (int i = 0; i < warmup_runs; i++) {
            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );
        }

        std::vector<double> latencies;

        for (int i = 0; i < benchmark_runs; i++) {
            auto start = std::chrono::steady_clock::now();

            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            auto end = std::chrono::steady_clock::now();

            double latency =
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count();

            latencies.push_back(latency);
        }

        double total = std::accumulate(
            latencies.begin(),
            latencies.end(),
            0.0
        );

        double average = total / benchmark_runs;

        double minimum = *std::min_element(
            latencies.begin(), latencies.end()
        );

        double maximum = *std::max_element(
            latencies.begin(), latencies.end()
        );

        double throughput = 1000.0 / average;

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "Warm-up runs: " << warmup_runs << "\n";
        std::cout << "Benchmark runs: " << benchmark_runs << "\n";
        std::cout << "Average latency: " << average << " ms\n";
        std::cout << "Minimum latency: " << minimum << " ms\n";
        std::cout << "Maximum latency: " << maximum << " ms\n";
        std::cout << "Throughput: " << throughput
                  << " images/sec\n";

        std::cout << "\nC++ Benchmark Completed Successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
