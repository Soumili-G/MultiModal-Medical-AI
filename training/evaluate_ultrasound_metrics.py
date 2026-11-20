
import os
import sys
import csv
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# PROJECT ROOT / IMPORT PATH
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from utils.multi_ultrasound_dataset import MultiUltrasoundDataset


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CONFIGS = {
    "breast": {
        "data_dir": "datasets/ultrasound/raw/Dataset_BUSI_with_GT",
        "model_path": "models/ultrasound_models/breast_model.pt",
    },

    "ovarian": {
        "data_dir": "datasets/ultrasound/raw/ovarian/Ovarian_US",
        "model_path": "models/ultrasound_models/ovarian_model.pt",
    },

    "abdominal": {
        "data_dir": "datasets/ultrasound/raw/abdominal/dataset publish/organ_classification_1",
        "model_path": "models/ultrasound_models/abdominal_model.pt",
    },
}

BATCH_SIZE = 32

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics",
    "ultrasound",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("ULTRASOUND MODEL EVALUATION")
print("=" * 65)

print("Device:", device)


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(name, config):

    print("\n" + "=" * 65)
    print("EVALUATING:", name.upper())
    print("=" * 65)

    data_dir = os.path.join(
        PROJECT_ROOT,
        config["data_dir"],
    )

    model_path = os.path.join(
        PROJECT_ROOT,
        config["model_path"],
    )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}"
        )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    # Load dataset and original class mapping
    full_dataset = MultiUltrasoundDataset(
        root_dir=data_dir,
        dataset_name=name,
        train=True,
    )

    class_names = full_dataset.class_names
    labels = full_dataset.labels

    print("Classes:", class_names)
    print("Total images:", len(full_dataset))

    # Recreate original 80:20 stratified split
    indices = np.arange(len(full_dataset))

    _, test_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    # Evaluation dataset without random augmentation
    eval_dataset = MultiUltrasoundDataset(
        root_dir=data_dir,
        dataset_name=name,
        train=False,
    )

    test_dataset = Subset(
        eval_dataset,
        test_indices,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Test images:", len(test_dataset))

    # Build ResNet18
    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names),
    )

    # Load trained checkpoint
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    y_true = []
    y_pred = []

    print("\nRunning inference...")

    with torch.no_grad():

        for batch_index, (images, labels_batch) in enumerate(test_loader):

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(
                labels_batch.numpy().tolist()
            )

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

            if (batch_index + 1) % 20 == 0:

                print(
                    f"Processed {batch_index + 1}/"
                    f"{len(test_loader)} batches"
                )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

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
    # PER-CLASS METRICS
    # --------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []

    for class_name in class_names:

        metrics = report[class_name]

        report_rows.append({
            "class": class_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": metrics["support"],
        })

    for average_name in ["macro avg", "weighted avg"]:

        metrics = report[average_name]

        report_rows.append({
            "class": average_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": metrics["support"],
        })

    report_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_classification_report.csv",
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class",
                "precision",
                "recall",
                "f1_score",
                "support",
            ],
        )

        writer.writeheader()
        writer.writerows(report_rows)

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    cm_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_confusion_matrix.csv",
    )

    with open(
        cm_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Actual / Predicted"] + class_names
        )

        for class_name, row in zip(class_names, cm):

            writer.writerow(
                [class_name] + row.tolist()
            )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    result = {
        "model": name,
        "test_images": len(y_true),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    print("\n" + "-" * 55)
    print(name.upper(), "RESULTS")
    print("-" * 55)

    for key, value in result.items():

        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(cm)

    print("\nSaved:", report_path)
    print("Saved:", cm_path)

    return result


# --------------------------------------------------
# RUN ALL THREE MODELS
# --------------------------------------------------

all_results = []

for model_name, config in CONFIGS.items():

    try:

        result = evaluate_model(
            model_name,
            config,
        )

        if result is not None:
            all_results.append(result)

    except Exception as error:

        print("\nERROR evaluating:", model_name)
        print(type(error).__name__, error)


# --------------------------------------------------
# SAVE COMBINED METRICS
# --------------------------------------------------

if all_results:

    summary_path = os.path.join(
        OUTPUT_DIR,
        "ultrasound_overall_metrics.csv",
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 65)
    print("ULTRASOUND EVALUATION COMPLETED")
    print("=" * 65)

    print("Models successfully evaluated:", len(all_results))
    print("Combined results:", summary_path)

else:

    print("\nNo models were successfully evaluated.")

print("\nFinished.")
