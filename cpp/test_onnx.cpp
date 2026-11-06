
#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "EdgeAI ONNX Runtime C++ Test\n";
    std::cout << "ONNX Runtime Version: "
              << OrtGetApiBase()->GetVersionString()
              << std::endl;

    return 0;
}
