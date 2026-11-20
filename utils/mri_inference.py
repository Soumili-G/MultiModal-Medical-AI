
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "mri_model" / "best_mri_model.pt"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_mri_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"MRI checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


def predict_mri(image, model):
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image = image.convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    return {
        "modality": "MRI",
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "class_probabilities": {
            name: float(probability.item())
            for name, probability in zip(CLASS_NAMES, probabilities)
        },
    }


if __name__ == "__main__":
    model = load_mri_model()
    print("MRI model loaded successfully!")
    print("Device:", DEVICE)
    print("Classes:", CLASS_NAMES)
