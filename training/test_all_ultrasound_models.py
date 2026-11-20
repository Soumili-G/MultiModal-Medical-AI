import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

IMAGE_SIZE = 224

standard_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

abdominal_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def predict_classification(model, image_tensor, classes):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()
        return classes[predicted_index], confidence


def load_classification_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model


def test_classification_model(task_name, data_dir, model_path, transform):
    print("=" * 70)
    print(f"TESTING {task_name.upper()} CLASSIFICATION MODEL")
    print("=" * 70)

    if not os.path.exists(data_dir):
        print(f"Dataset folder not found: {data_dir}\n")
        return

    classes = sorted(
        entry for entry in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, entry))
    )

    if not classes:
        print(f"No class folders found in: {data_dir}\n")
        return

    print(f"Classes: {classes}\n")

    model = load_classification_model(model_path, len(classes))

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        images = [
            file for file in os.listdir(class_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Exclude segmentation masks from the breast dataset
        if task_name == "breast":
            images = [
                file for file in images
                if "_mask" not in file.lower()
            ]

        if not images:
            continue

        image_file = random.choice(images)
        image_path = os.path.join(class_dir, image_file)
        image_tensor = load_image(image_path, transform)

        predicted_class, confidence = predict_classification(
            model, image_tensor, classes
        )

        result = "CORRECT" if predicted_class == class_name else "WRONG"

        print(f"Image: {image_path}")
        print(f"True label: {class_name}")
        print(f"Predicted label: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Result: {result}\n")


def test_fetal_model():
    print("=" * 70)
    print("TESTING FETAL AOP REGRESSION MODEL")
    print("=" * 70)

    csv_path = "datasets/ultrasound/raw/fetal/Dataset/Training/Labeled cases/label.csv"
    image_dir = "datasets/ultrasound/raw/fetal/Dataset/Training/Labeled cases"
    model_path = "models/ultrasound_models/fetal_aop_model.pt"

    if not os.path.exists(csv_path):
        print(f"Fetal label CSV not found: {csv_path}\n")
        return

    if not os.path.exists(model_path):
        print(f"Fetal model not found: {model_path}\n")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Filename", "AOP"])

    valid_rows = []
    for _, row in df.iterrows():
        if os.path.exists(os.path.join(image_dir, row["Filename"])):
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows).reset_index(drop=True)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    aop_mean = checkpoint["aop_mean"]
    aop_std = checkpoint["aop_std"]

    sample_rows = df.sample(n=5, random_state=42)

    errors = []

    for _, row in sample_rows.iterrows():
        image_path = os.path.join(image_dir, row["Filename"])
        image_tensor = load_image(image_path, standard_transform)

        with torch.no_grad():
            output = model(image_tensor).squeeze().cpu().item()

        predicted_aop = output * aop_std + aop_mean
        actual_aop = float(row["AOP"])
        error = abs(predicted_aop - actual_aop)
        errors.append(error)

        print(f"Image: {image_path}")
        print(f"Actual AOP: {actual_aop:.2f} degrees")
        print(f"Predicted AOP: {predicted_aop:.2f} degrees")
        print(f"Absolute error: {error:.2f} degrees\n")

    print(f"Fetal sample MAE: {sum(errors) / len(errors):.2f} degrees\n")


# -------------------------
# CONFIGURATION
# -------------------------
breast_dir = "datasets/ultrasound/raw/busi/Dataset_BUSI_with_GT"
breast_model = "models/ultrasound_models/breast_model.pt"

ovarian_dir = "datasets/ultrasound/raw/ovarian/Ovarian_US"
ovarian_model = "models/ultrasound_models/ovarian_model.pt"

abdominal_dir = (
    "datasets/ultrasound/raw/abdominal/"
    "dataset publish/organ_classification_1"
)
abdominal_model = "models/ultrasound_models/abdominal_model.pt"

# -------------------------
# RUN ALL TESTS
# -------------------------
test_classification_model(
    "breast",
    breast_dir,
    breast_model,
    standard_transform
)

test_classification_model(
    "ovarian",
    ovarian_dir,
    ovarian_model,
    standard_transform
)

test_classification_model(
    "abdominal",
    abdominal_dir,
    abdominal_model,
    abdominal_transform
)

test_fetal_model()

print("All ultrasound model tests completed.")