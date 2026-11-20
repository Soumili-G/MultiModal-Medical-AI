from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, image_size=224, train=True):
        self.root_dir = Path(root_dir)
        self.image_paths = []
        self.labels = []

        self.class_names = sorted(
            [folder.name for folder in self.root_dir.iterdir() if folder.is_dir()]
        )

        self.class_to_index = {
            name: index for index, name in enumerate(self.class_names)
        }

        for class_name in self.class_names:
            class_folder = self.root_dir / class_name

            for image_path in class_folder.glob("*"):
                filename = image_path.name.lower()

                # Skip segmentation masks
                if "mask" in filename:
                    continue

                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.image_paths.append(image_path)
                    self.labels.append(self.class_to_index[class_name])

        if train:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label