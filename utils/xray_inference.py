
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "xray_model"
    / "resnet18" / "best_resnet18.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_xray_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"X-ray checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    # Trusted checkpoint from our own Kaggle training
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]

    # Remove DataParallel prefix
    cleaned_state_dict = {
        key.replace("module.", "", 1)
        if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

    model.load_state_dict(cleaned_state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

def predict_xray(image, model):

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image = image.convert("RGB")

    input_tensor = (
        TRANSFORM(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0]

    disease_scores = {
        name: float(score.item())
        for name, score in zip(CLASS_NAMES, probabilities)
    }

    return {
        "modality": "X-ray",
        "prediction_type": "Multi-label classification",
        "disease_scores": disease_scores,
    }


# --------------------------------------------------
# QUICK CHECK
# --------------------------------------------------

if __name__ == "__main__":

    model = load_xray_model()

    print("X-ray ResNet18 loaded successfully!")
    print("Device:", DEVICE)
    print("Output classes:", len(CLASS_NAMES))
    print("Checkpoint:", CHECKPOINT_PATH.name)
