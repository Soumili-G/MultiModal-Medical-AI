
import torch
import torch.nn as nn
from torchvision import models

CHECKPOINT_PATH = "models/xray_model/resnet18/best_resnet18.pth"

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the ResNet18 architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))

# Load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

# Remove DataParallel prefix
state_dict = {
    key.replace("module.", "", 1): value
    for key, value in checkpoint["model_state_dict"].items()
}

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("ResNet18 loaded successfully!")
print("Best epoch:", checkpoint["epoch"])
print("Validation Macro AUC:", checkpoint["macro_auc"])
print("Device:", device)
print("Output classes:", len(LABELS))
