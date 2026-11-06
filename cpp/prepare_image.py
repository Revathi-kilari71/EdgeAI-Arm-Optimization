
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=transform
)

image, label = dataset[0]

output = Path("cpp/test_image.bin")
output.parent.mkdir(parents=True, exist_ok=True)

image.numpy().astype(np.float32).tofile(output)

print("Test image prepared successfully!")
print("Actual label:", label)
print("Class name:", dataset.classes[label])
print("Input shape:", tuple(image.shape))
print("Pixel range:", image.min().item(), image.max().item())
print("Saved to:", output)
