
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    image, label = dataset[i]
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(dataset.classes[label])
    ax.axis("off")

plt.suptitle("Fashion-MNIST Sample Images")
plt.tight_layout()

Path("results").mkdir(exist_ok=True)
plt.savefig("results/sample_images.png", dpi=300)
plt.show()

print("Dataset Visualization Completed!")
