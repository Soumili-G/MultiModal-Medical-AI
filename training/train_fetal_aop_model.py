import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

CSV_PATH = "datasets/ultrasound/raw/fetal/Dataset/Training/Labeled cases/label.csv"
IMAGE_DIR = "datasets/ultrasound/raw/fetal/Dataset/Training/Labeled cases"
MODEL_SAVE_PATH = "models/ultrasound_models/fetal_aop_model.pt"

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.00005
IMAGE_SIZE = 224
VAL_SPLIT = 0.2
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FetalAOPDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, aop_mean=0.0, aop_std=1.0):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.aop_mean = aop_mean
        self.aop_std = aop_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row["Filename"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        aop = float(row["AOP"])
        normalized_aop = (aop - self.aop_mean) / self.aop_std

        return image, torch.tensor(normalized_aop, dtype=torch.float32)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Filename", "AOP"])

valid_rows = []
for _, row in df.iterrows():
    if os.path.exists(os.path.join(IMAGE_DIR, row["Filename"])):
        valid_rows.append(row)

df = pd.DataFrame(valid_rows).reset_index(drop=True)
print(f"Found {len(df)} valid labeled fetal images")

train_df, val_df = train_test_split(
    df,
    test_size=VAL_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

aop_mean = train_df["AOP"].mean()
aop_std = train_df["AOP"].std()

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"AOP mean: {aop_mean:.2f} degrees")
print(f"AOP standard deviation: {aop_std:.2f} degrees")

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = FetalAOPDataset(
    train_df, IMAGE_DIR, train_transform, aop_mean, aop_std
)

val_dataset = FetalAOPDataset(
    val_df, IMAGE_DIR, val_transform, aop_mean, aop_std
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_mae = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, normalized_angles in train_loader:
        images = images.to(device)
        normalized_angles = normalized_angles.unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, normalized_angles)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    val_mae = 0.0

    with torch.no_grad():
        for images, normalized_angles in val_loader:
            images = images.to(device)
            normalized_angles = normalized_angles.unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, normalized_angles)

            # Convert back to degrees
            predicted_degrees = outputs.squeeze(1) * aop_std + aop_mean
            actual_degrees = normalized_angles.squeeze(1) * aop_std + aop_mean

            val_loss += loss.item() * images.size(0)
            val_mae += torch.abs(predicted_degrees - actual_degrees).sum().item()

    val_loss /= len(val_dataset)
    val_mae /= len(val_dataset)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Val MSE: {val_loss:.4f} | "
        f"Val MAE: {val_mae:.2f} degrees"
    )

    if val_mae < best_val_mae:
        best_val_mae = val_mae

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "aop_mean": aop_mean,
            "aop_std": aop_std
        }

        torch.save(checkpoint, MODEL_SAVE_PATH)
        print(f"Saved improved model to {MODEL_SAVE_PATH}")

print("\nTraining complete.")
print(f"Best validation MAE: {best_val_mae:.2f} degrees")
print(f"Model saved to: {MODEL_SAVE_PATH}")