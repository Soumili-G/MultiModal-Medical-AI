import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.multi_ultrasound_dataset import MultiUltrasoundDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

ULTRASOUND_CONFIGS = {
    "breast": {
        "data_dir": "datasets/ultrasound/raw/busi/Dataset_BUSI_with_GT",
        "model_save_path": "models/ultrasound_models/breast_model.pt",
        "epochs": 5,
        "batch_size": 16,
    },
    "ovarian": {
        "data_dir": "datasets/ultrasound/raw/ovarian/Ovarian_US",
        "model_save_path": "models/ultrasound_models/ovarian_model.pt",
        "epochs": 5,
        "batch_size": 16,
    },
    "abdominal": {
        "data_dir": "datasets/ultrasound/raw/abdominal/dataset publish/organ_classification_1",
        "model_save_path": "models/ultrasound_models/abdominal_model.pt",
        "epochs": 5,
        "batch_size": 16,
    },
    "fetal": {
        "data_dir": "datasets/ultrasound/raw/fetal/REPLACE_WITH_FETAL_FOLDER",
        "model_save_path": "models/ultrasound_models/fetal_model.pt",
        "epochs": 3,
        "batch_size": 16,
    },
}

if len(sys.argv) < 2:
    print("Please choose a dataset.")
    print("Usage:")
    print("  python training/train_ultrasound_model.py breast")
    print("  python training/train_ultrasound_model.py ovarian")
    print("  python training/train_ultrasound_model.py abdominal")
    print("  python training/train_ultrasound_model.py fetal")
    sys.exit()

dataset_name = sys.argv[1].lower()

if dataset_name not in ULTRASOUND_CONFIGS:
    print("Unknown dataset:", dataset_name)
    print("Available options:", list(ULTRASOUND_CONFIGS.keys()))
    sys.exit()

config = ULTRASOUND_CONFIGS[dataset_name]

data_dir = config["data_dir"]
model_save_path = config["model_save_path"]
epochs = config["epochs"]
batch_size = config["batch_size"]

print("Training ultrasound model:", dataset_name)
print("Using device:", device)
print("Data folder:", data_dir)

if not os.path.exists(data_dir):
    print("ERROR: Data folder does not exist.")
    print("Please update `data_dir` in ULTRASOUND_CONFIGS.")
    print("Current folder:", data_dir)
    sys.exit()

print("Loading dataset...")
full_dataset = MultiUltrasoundDataset(
    root_dir=data_dir,
    dataset_name=dataset_name,
    train=True,
)

print("Classes:", full_dataset.class_names)
print("Total images:", len(full_dataset))

if len(full_dataset) == 0:
    print("ERROR: No images found.")
    print("Check that `data_dir` points to the folder containing class folders.")
    sys.exit()

labels = full_dataset.labels

train_indices, test_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=labels,
    random_state=42,
)

train_dataset = Subset(full_dataset, train_indices)

eval_dataset = MultiUltrasoundDataset(
    root_dir=data_dir,
    dataset_name=dataset_name,
    train=False,
)

test_dataset = Subset(eval_dataset, test_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

print("Creating model...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

best_accuracy = 0.0

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

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print("Saved new best model.")

print("Training complete.")
print("Best test accuracy:", best_accuracy)

print("Final classification report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=full_dataset.class_names,
    )
)

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))