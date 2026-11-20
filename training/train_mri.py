import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix

# Allow Python to find the utils folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import MRIDataset

# Use CPU or GPU automatically
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# Load datasets
print("Loading training dataset...")
train_dataset = MRIDataset(
    root_dir="datasets/mri/raw/Training",
    train=True,
)

print("Loading testing dataset...")
test_dataset = MRIDataset(
    root_dir="datasets/mri/raw/Testing",
    train=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

print("Classes:", train_dataset.class_names)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

# Create model
print("Creating model...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

best_accuracy = 0.0
epochs = 5

print("Starting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    batch_number = 0

    for images, labels in train_loader:
        batch_number += 1

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_number % 20 == 0:
            print(
                f"Epoch {epoch + 1}, "
                f"Batch {batch_number}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print("Evaluating...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    print(f"Epoch {epoch + 1}/{epochs} finished")
    print(f"Average training loss: {total_loss / len(train_loader):.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("-" * 50)

    if accuracy > best_accuracy:
        best_accuracy = accuracy

        os.makedirs("models/mri_model", exist_ok=True)
        torch.save(
            model.state_dict(),
            "models/mri_model/best_mri_model.pt",
        )
        print("Saved new best model.")

print("Training complete.")
print("Best test accuracy:", best_accuracy)

print("Final classification report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=train_dataset.class_names,
    )
)

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))