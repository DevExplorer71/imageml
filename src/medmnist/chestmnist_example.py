

import medmnist
from medmnist import ChestMNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Transform to convert PIL images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])


# Download and load ChestMNIST with transform
train_dataset = ChestMNIST(split='train', download=True, transform=transform)
val_dataset = ChestMNIST(split='val', download=True, transform=transform)
test_dataset = ChestMNIST(split='test', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Data Visualization ---
# Label distribution
import numpy as np
labels = train_dataset.labels
label_sums = np.sum(labels, axis=0)
plt.figure(figsize=(10,4))
plt.bar(range(14), label_sums)
plt.title('ChestMNIST Label Distribution (Train)')
plt.xlabel('Label Index')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chestmnist_label_distribution.png')
plt.close()

# Sample images
fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flatten()):
    img = train_dataset[i][0].squeeze().numpy()
    label = train_dataset[i][1]  # Already a numpy array
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Labels: {np.where(label==1)[0].tolist()}')
    ax.axis('off')
plt.suptitle('Sample ChestMNIST Images')
plt.tight_layout()
plt.savefig('chestmnist_sample_images.png')
plt.close()

# Simple CNN model for MedMNIST
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training loop
model = SimpleCNN(num_classes=14)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).float().sum().item()
            total_samples += labels.numel()
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

for epoch in range(1, 4):
    train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch} completed. Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# --- Prediction Visualization ---
model.eval()
images, labels = next(iter(test_loader))
outputs = model(images)
preds = (torch.sigmoid(outputs) > 0.5).float()
fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flatten()):
    img = images[i].squeeze().numpy()
    true_labels = np.where(labels[i].numpy() == 1)[0].tolist()
    pred_labels = np.where(preds[i].numpy() == 1)[0].tolist()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"T:{true_labels} | P:{pred_labels}")
    ax.axis('off')
plt.suptitle('MedMNIST Predictions (Test)')
plt.tight_layout()
plt.savefig('chestmnist_test_predictions.png')
plt.close()
