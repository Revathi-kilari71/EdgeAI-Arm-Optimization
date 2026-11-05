
import onnx
from collections import Counter

models = {
    "FP32": "models/mobilenetv3_fp32.onnx",
    "INT8": "models/mobilenetv3_int8.onnx"
}

for name, path in models.items():
    model = onnx.load(path)
    onnx.checker.check_model(model)

    operations = Counter(
        node.op_type for node in model.graph.node
    )

    print(f"\n{name} MODEL OPERATIONS")
    print("-" * 40)

    for operation, count in operations.most_common():
        print(f"{operation}: {count}")

    print("Total Nodes:", len(model.graph.node))

print("\nONNX Operator Analysis Completed!")
