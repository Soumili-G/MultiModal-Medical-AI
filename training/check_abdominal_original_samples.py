import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DATA_DIR = "datasets/ultrasound/raw/abdominal/dataset publish/organ_classification_1"
MODEL_PATH = "models/ultrasound_models/abdominal_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = sorted(
    entry for entry in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, entry))
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()

sample_files = {
    "AA": "1.jpg",
    "GB": "1434.jpg",
    "Hepatic": "385.jpg",
    "Kidney": "463.jpg",
    "Liver": "1000.jpg",
    "Ovaries": "1205.jpg",
    "Pancreas": "1651.jpg",
    "Portal": "1823.jpg",
    "Spleen": "1919.jpg",
    "UB+prostrate+uterous+cervix": "2082.jpg"
}

for true_class, filename in sample_files.items():
    path = os.path.join(DATA_DIR, true_class, filename)
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    predicted_class = classes[predicted_index]

    print(
        f"True: {true_class} | "
        f"Predicted: {predicted_class} | "
        f"Confidence: {probabilities[predicted_index].item() * 100:.2f}%"
    )