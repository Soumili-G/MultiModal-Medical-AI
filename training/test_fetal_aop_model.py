
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CSV_PATH = "datasets/ultrasound/raw/fetal/Dataset/Training/Labeled cases/label.csv"
IMAGE_DIR = "datasets/ultrasound/raw/fetal/Dataset/Training/Labeled cases"
MODEL_PATH = "models/ultrasound_models/fetal_aop_model.pt"
OUTPUT_DIR = "results/metrics/fetal_aop"

IMAGE_SIZE = 224
BATCH_SIZE = 16
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("FETAL AOP REGRESSION EVALUATION")
print("=" * 65)
print("Device:", device)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

class FetalAOPDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = os.path.join(
            self.image_dir,
            row["Filename"]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        angle = torch.tensor(
            float(row["AOP"]),
            dtype=torch.float32
        )

        return image, angle, row["Filename"]


# --------------------------------------------------
# LOAD AND FILTER DATA
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

df = df.dropna(
    subset=["Filename", "AOP"]
)

valid_rows = []

for _, row in df.iterrows():

    image_path = os.path.join(
        IMAGE_DIR,
        row["Filename"]
    )

    if os.path.exists(image_path):
        valid_rows.append(row)

df = pd.DataFrame(valid_rows).reset_index(drop=True)

print("Total valid labeled images:", len(df))

# Recreate the original training/validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    shuffle=True
)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_dataset = FetalAOPDataset(
    val_df,
    IMAGE_DIR,
    transform
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

aop_mean = checkpoint["aop_mean"]
aop_std = checkpoint["aop_std"]

print("\nModel loaded successfully.")
print("AOP mean:", aop_mean)
print("AOP standard deviation:", aop_std)

# --------------------------------------------------
# RUN VALIDATION
# --------------------------------------------------

all_actual = []
all_predicted = []
all_filenames = []

print("\nRunning inference...")

with torch.no_grad():

    for batch_number, (images, angles, filenames) in enumerate(
        validation_loader, 1
    ):

        images = images.to(device)

        outputs = model(images).squeeze(1).cpu()

        predicted_degrees = (
            outputs * aop_std + aop_mean
        )

        all_actual.extend(angles.numpy().tolist())
        all_predicted.extend(predicted_degrees.numpy().tolist())
        all_filenames.extend(list(filenames))

        if batch_number % 20 == 0:
            print(
                f"Processed {batch_number}/"
                f"{len(validation_loader)} batches"
            )

y_true = np.array(all_actual)
y_pred = np.array(all_predicted)

# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

errors = y_pred - y_true
absolute_errors = np.abs(errors)

mae = mean_absolute_error(y_true, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

r2 = r2_score(y_true, y_pred)

median_error = median_absolute_error(
    y_true,
    y_pred
)

mean_bias = np.mean(errors)

within_2 = np.mean(absolute_errors <= 2) * 100
within_5 = np.mean(absolute_errors <= 5) * 100
within_10 = np.mean(absolute_errors <= 10) * 100

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

metrics = {
    "model": "Fetal AOP Regression",
    "evaluation_images": len(y_true),
    "MAE_degrees": mae,
    "RMSE_degrees": rmse,
    "R2_score": r2,
    "Median_absolute_error_degrees": median_error,
    "Mean_bias_degrees": mean_bias,
    "Predictions_within_2_degrees_percent": within_2,
    "Predictions_within_5_degrees_percent": within_5,
    "Predictions_within_10_degrees_percent": within_10,
}

print("\n" + "=" * 65)
print("FETAL AOP EVALUATION RESULTS")
print("=" * 65)

for name, value in metrics.items():

    if isinstance(value, (float, np.floating)):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

# --------------------------------------------------
# SAVE IMAGE-LEVEL PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "Filename": all_filenames,
    "Actual_AOP": y_true,
    "Predicted_AOP": y_pred,
    "Error": errors,
    "Absolute_Error": absolute_errors,
})

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fetal_aop_predictions.csv"
    ),
    index=False
)

# --------------------------------------------------
# SAVE OVERALL METRICS
# --------------------------------------------------

pd.DataFrame([metrics]).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fetal_aop_overall_metrics.csv"
    ),
    index=False
)

# --------------------------------------------------
# DISPLAY SAMPLE PREDICTIONS
# --------------------------------------------------

print("\nFirst 20 predictions:")

print(
    results.head(20).round(3).to_string(index=False)
)

print("\nBest absolute error:", absolute_errors.min())
print("Worst absolute error:", absolute_errors.max())

print("\n" + "=" * 65)
print("FETAL AOP EVALUATION COMPLETED")
print("=" * 65)

print(
    "Results saved in:",
    os.path.abspath(OUTPUT_DIR)
)
