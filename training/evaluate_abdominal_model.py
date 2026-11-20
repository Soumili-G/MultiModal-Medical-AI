import os
import random
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict

DATA_DIR = "datasets/ultrasound/raw/abdominal/dataset publish/organ_classification_1"
MODEL_PATH = "models/ultrasound_models/abdominal_model.pt"

IMAGE_SIZE = 224
SAMPLES_PER_CLASS = 30
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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

random.seed(RANDOM_SEED)

correct = 0
total = 0
class_results = {}

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

    class_correct = 0

    for image_file in sampled_images:
        image_path = os.path.join(class_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            predicted_index = torch.argmax(output, dim=1).item()

        predicted_class = classes[predicted_index]

        if predicted_class == true_class:
            correct += 1
            class_correct += 1

        total += 1

    class_results[true_class] = {
        "correct": class_correct,
        "total": len(sampled_images),
        "accuracy": class_correct / len(sampled_images) * 100
    }

print("=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

for class_name, result in class_results.items():
    print(
        f"{class_name}: "
        f"{result['correct']}/{result['total']} correct "
        f"({result['accuracy']:.2f}%)"
    )

print("\n" + "=" * 60)
print("OVERALL RESULT")
print("=" * 60)
print(f"Total images tested: {total}")
print(f"Correct predictions: {correct}")
print(f"Overall accuracy: {correct / total * 100:.2f}%")