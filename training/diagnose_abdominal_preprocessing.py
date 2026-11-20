import os
import random
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DATA_DIR = "datasets/ultrasound/raw/abdominal/dataset publish/organ_classification_1"
MODEL_PATH = "models/ultrasound_models/abdominal_model.pt"

IMAGE_SIZE = 224
SAMPLES_PER_CLASS = 20
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

classes = sorted(
    entry for entry in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, entry))
)

print(f"Classes: {classes}\n")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()

preprocessing_variants = {
    "ImageNet RGB normalization": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    "RGB without normalization": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ]),
    "Grayscale converted to 3 channels, ImageNet normalization": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    "Grayscale converted to 3 channels, no normalization": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ]),
    "Simple 0-1 scaling with mean 0.5 and std 0.5": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
}

random.seed(RANDOM_SEED)

sample_images = []

for true_class in classes:
    class_dir = os.path.join(DATA_DIR, true_class)
    images = [
        file for file in os.listdir(class_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    sampled_images = random.sample(
        images,
        min(SAMPLES_PER_CLASS, len(images))
    )

    for image_file in sampled_images:
        sample_images.append((
            os.path.join(class_dir, image_file),
            true_class
        ))

print(f"Testing {len(sample_images)} abdominal images.\n")

results = {}

for variant_name, transform in preprocessing_variants.items():
    correct = 0
    total = 0

    for image_path, true_class in sample_images:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            predicted_index = torch.argmax(output, dim=1).item()

        predicted_class = classes[predicted_index]

        if predicted_class == true_class:
            correct += 1

        total += 1

    accuracy = correct / total * 100
    results[variant_name] = accuracy

    print(f"{variant_name}:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%\n")

best_variant = max(results, key=results.get)

print("=" * 70)
print("BEST PREPROCESSING SETTINGS")
print("=" * 70)
print(f"Best variant: {best_variant}")
print(f"Best accuracy: {results[best_variant]:.2f}%")