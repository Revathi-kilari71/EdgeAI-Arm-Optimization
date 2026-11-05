
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def create_model(num_classes=10):
    model = mobilenet_v3_small(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes
    )

    return model


if __name__ == "__main__":
    model = create_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sample_input = torch.randn(1, 3, 96, 96).to(device)

    with torch.no_grad():
        output = model(sample_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Model: MobileNetV3-Small")
    print("Device:", device)
    print("Input Shape:", sample_input.shape)
    print("Output Shape:", output.shape)
    print("Total Parameters:", f"{total_params:,}")
    print("Trainable Parameters:", f"{trainable_params:,}")
    print("Model Architecture Created Successfully!")
