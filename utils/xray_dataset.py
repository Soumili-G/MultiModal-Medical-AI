
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/xray_model/resnet18/best_resnet18.pth"

LABELS = [
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
    "Hernia"
]

IMAGE_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_xray_model():

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in checkpoint["model_state_dict"].items()
    }

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


model = load_xray_model()


def predict_xray(image_path):

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output)[0]

    results = {
        label: round(probability.item() * 100, 2)
        for label, probability in zip(LABELS, probabilities)
    }

    return dict(
        sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )


if __name__ == "__main__":
    print("X-ray model loaded successfully.")
    print("Device:", device)
    print("Number of disease outputs:", len(LABELS))
