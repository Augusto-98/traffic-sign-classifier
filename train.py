import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Config
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
NUM_CLASSES = 43
DEVICE = torch.device("cpu")
SAVE_PATH = "model_weights/efficientnet_gtsrb.pt"

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.3337, 0.3064, 0.3171],
                         [0.2672, 0.2564, 0.2629])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.3337, 0.3064, 0.3171],
                         [0.2672, 0.2564, 0.2629])
])

# Data
train_data = datasets.GTSRB(root='./data', split='train', download=False, transform=train_transform)
test_data  = datasets.GTSRB(root='./data', split='test',  download=False, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

# Model
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Test Acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        os.makedirs("model_weights", exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  → Model saved (best acc: {best_acc:.4f})")

print(f"\n Training complete. Best accuracy: {best_acc:.4f}")