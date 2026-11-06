
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
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
            throw std::runtime_error("Test image file not found");
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

        auto start = std::chrono::high_resolution_clock::now();

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        auto end = std::chrono::high_resolution_clock::now();

        double latency =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        auto output_info =
            outputs[0].GetTensorTypeAndShapeInfo();

        size_t count = output_info.GetElementCount();
        float* scores = outputs[0].GetTensorMutableData<float>();

        int predicted = static_cast<int>(std::distance(
            scores,
            std::max_element(scores, scores + count)
        ));

        std::vector<std::string> classes = {
            "T-shirt/top", "Trouser", "Pullover", "Dress",
            "Coat", "Sandal", "Shirt", "Sneaker",
            "Bag", "Ankle boot"
        };

        int actual = 9;

        std::cout << "\nEDGEAI REAL IMAGE INFERENCE\n";
        std::cout << "Actual class: " << classes[actual] << "\n";
        std::cout << "Predicted class: " << classes.at(predicted) << "\n";
        std::cout << "Prediction correct: "
                  << (predicted == actual ? "YES" : "NO") << "\n";
        std::cout << "Inference latency: " << latency << " ms\n";
        std::cout << "C++ inference completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
