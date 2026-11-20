
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "ultrasound_models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = {
    "breast": [
        "benign",
        "malignant",
        "normal",
    ],

    "ovarian": [
        "complex_cyst",
        "dominant_follicle",
        "healthy",
        "poly_cyst",
        "simple_cyst",
    ],

    "abdominal": [
        "AA",
        "GB",
        "Hepatic",
        "Kidney",
        "Liver",
        "Ovaries",
        "Pancreas",
        "Portal",
        "Spleen",
        "UB+prostrate+uterous+cervix",
    ],
}

CHECKPOINTS = {
    "breast": MODEL_DIR / "breast_model.pt",
    "ovarian": MODEL_DIR / "ovarian_model.pt",
    "abdominal": MODEL_DIR / "abdominal_model.pt",
    "fetal": MODEL_DIR / "fetal_aop_model.pt",
}


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def get_transform(model_type):

    steps = [
        transforms.Resize((224, 224))
    ]

    if model_type == "abdominal":
        steps.append(
            transforms.Grayscale(num_output_channels=3)
        )

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(steps)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_ultrasound_model(model_type):

    model_type = model_type.lower()

    if model_type not in CHECKPOINTS:
        raise ValueError(
            f"Unknown ultrasound model: {model_type}"
        )

    checkpoint_path = CHECKPOINTS[model_type]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = models.resnet18(weights=None)

    aop_mean = None
    aop_std = None

    # FETAL AOP REGRESSION MODEL
    if model_type == "fetal":

        model.fc = nn.Linear(
            model.fc.in_features,
            1
        )

        # Trusted checkpoint created by our own training script.
        # weights_only=False is needed because it contains NumPy scalars.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        aop_mean = float(checkpoint["aop_mean"])
        aop_std = float(checkpoint["aop_std"])

        if aop_std <= 0:
            raise ValueError("Invalid fetal AOP standard deviation.")

    # BREAST / OVARIAN / ABDOMINAL CLASSIFICATION MODELS
    else:

        classes = CLASS_NAMES[model_type]

        model.fc = nn.Linear(
            model.fc.in_features,
            len(classes)
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return {
        "model": model,
        "model_type": model_type,
        "device": DEVICE,
        "transform": get_transform(model_type),
        "class_names": CLASS_NAMES.get(model_type),
        "aop_mean": aop_mean,
        "aop_std": aop_std,
    }


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_ultrasound(image, model_bundle):

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    model_type = model_bundle["model_type"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    image = image.convert("RGB")

    input_tensor = (
        model_bundle["transform"](image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = model(input_tensor)

        # FETAL AOP REGRESSION
        if model_type == "fetal":

            normalized_aop = output.item()

            predicted_aop = (
                normalized_aop * model_bundle["aop_std"]
                + model_bundle["aop_mean"]
            )

            return {
                "modality": "Ultrasound",
                "subtype": "Fetal",
                "prediction_type": "AOP regression",
                "predicted_aop_degrees": float(predicted_aop),
            }

        # BREAST / OVARIAN / ABDOMINAL CLASSIFICATION
        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        predicted_index = probabilities.argmax().item()

        classes = model_bundle["class_names"]

        return {
            "modality": "Ultrasound",
            "subtype": model_type,
            "predicted_class": classes[predicted_index],
            "confidence": float(
                probabilities[predicted_index].item()
            ),
            "class_probabilities": {
                name: float(probability.item())
                for name, probability in zip(
                    classes,
                    probabilities
                )
            },
        }


# --------------------------------------------------
# QUICK CHECK
# --------------------------------------------------

if __name__ == "__main__":

    model_types = [
        "breast",
        "ovarian",
        "abdominal",
        "fetal",
    ]

    for model_type in model_types:

        bundle = load_ultrasound_model(model_type)

        print(
            f"{model_type.capitalize()} model loaded successfully!"
        )

        if model_type == "fetal":

            print("Task: AOP regression")
            print("AOP mean:", bundle["aop_mean"])
            print("AOP std:", bundle["aop_std"])

        else:

            print("Classes:", bundle["class_names"])

        print("Device:", bundle["device"])
        print("-" * 50)

    print("\nAll four ultrasound models loaded successfully!")
