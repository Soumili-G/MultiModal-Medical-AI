import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import MRIDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# Load class names
temp_dataset = MRIDataset(
    root_dir="datasets/mri/raw/Training",
    train=False,
)

class_names = temp_dataset.class_names
print("Classes:", class_names)

# Load trained model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(
    torch.load(
        "models/mri_model/best_mri_model.pt",
        map_location=device,
    )
)

model = model.to(device)
model.eval()

# Test one image from every class
testing_root = "datasets/mri/raw/Testing"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

for class_name in class_names:
    class_folder = os.path.join(testing_root, class_name)

    image_files = [
        file for file in os.listdir(class_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_path = os.path.join(class_folder, image_files[0])
    true_label = class_name

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_class = class_names[probabilities.argmax().item()]
    confidence = probabilities.max().item()

    print("\n" + "=" * 60)
    print("Image:", image_path)
    print("True label:", true_label)
    print("Predicted label:", predicted_class)
    print(f"Confidence: {confidence:.2%}")

    if predicted_class == true_label:
        print("Result: CORRECT")
    else:
        print("Result: WRONG")

    print("Probabilities:")
    for name, probability in zip(class_names, probabilities):
        print(f"  {name}: {probability.item():.4f}")