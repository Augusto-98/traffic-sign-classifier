import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Download automático do GTSRB
print("A fazer download do GTSRB...")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_data = datasets.GTSRB(
    root='./data',
    split='train',
    download=True,
    transform=transform
)

test_data = datasets.GTSRB(
    root='./data',
    split='test',
    download=True,
    transform=transform
)

print(f"Train samples: {len(train_data)}")
print(f"Test samples:  {len(test_data)}")
print("Download completo!")