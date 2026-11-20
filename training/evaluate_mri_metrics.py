
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from utils.dataset import MRIDataset

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TEST_DIR = "datasets/mri/raw/Testing"
MODEL_PATH = "models/mri_model/best_mri_model.pt"
OUTPUT_DIR = "results/metrics/mri"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("MRI MODEL EVALUATION")
print("=" * 65)
print("Device:", device)

# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

test_dataset = MRIDataset(
    root_dir=TEST_DIR,
    train=False,
)

class_names = test_dataset.class_names
num_classes = len(class_names)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

print("Classes:", class_names)
print("Total test images:", len(test_dataset))

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

print("\nModel loaded successfully.")
print("Running inference...")

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for batch_number, (images, labels) in enumerate(test_loader, 1):

        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        if batch_number % 20 == 0:
            print(
                f"Processed {batch_number}/{len(test_loader)} batches"
            )

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)

# --------------------------------------------------
# CLASSIFICATION METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

balanced_acc = balanced_accuracy_score(y_true, y_pred)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)

# --------------------------------------------------
# ROC-AUC
# --------------------------------------------------

try:
    auc_macro = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=np.arange(num_classes),
    )

    auc_weighted = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="weighted",
        labels=np.arange(num_classes),
    )

except ValueError as error:
    print("ROC-AUC could not be calculated:", error)
    auc_macro = np.nan
    auc_weighted = np.nan

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("MRI EVALUATION RESULTS")
print("=" * 65)

metrics = {
    "model": "MRI",
    "test_images": len(y_true),
    "accuracy": accuracy,
    "balanced_accuracy": balanced_acc,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_precision": weighted_precision,
    "weighted_recall": weighted_recall,
    "weighted_f1": weighted_f1,
    "macro_roc_auc_ovr": auc_macro,
    "weighted_roc_auc_ovr": auc_weighted,
}

for name, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        zero_division=0,
    )
)

report_df.to_csv(
    os.path.join(OUTPUT_DIR, "mri_classification_report.csv")
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)

print("\nConfusion matrix:")
print(cm)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "mri_confusion_matrix.csv")
)

# --------------------------------------------------
# SAVE SUMMARY METRICS
# --------------------------------------------------

pd.DataFrame([metrics]).to_csv(
    os.path.join(OUTPUT_DIR, "mri_overall_metrics.csv"),
    index=False,
)

# --------------------------------------------------
# SAVE IMAGE-LEVEL PREDICTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "image_path": [
        str(path) for path in test_dataset.image_paths
    ],
    "true_label": [
        class_names[label] for label in y_true
    ],
    "predicted_label": [
        class_names[label] for label in y_pred
    ],
    "correct": y_true == y_pred,
    "confidence": y_prob.max(axis=1),
})

for index, class_name in enumerate(class_names):
    prediction_df[
        f"probability_{class_name}"
    ] = y_prob[:, index]

prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "mri_image_predictions.csv"),
    index=False,
)

print("\n" + "=" * 65)
print("MRI EVALUATION COMPLETED")
print("=" * 65)
print("Results saved in:", os.path.abspath(OUTPUT_DIR))
