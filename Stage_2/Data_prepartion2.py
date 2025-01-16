import os
import torch
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations import Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_data(path):
    image_list, label_list = [], []

    for snake_species_folder in sorted(os.listdir(path)):
        print(f"Defining paths for: {snake_species_folder}.")
        snake_species_folder_image_list = os.listdir(os.path.join(path, snake_species_folder))

        for image in snake_species_folder_image_list:
            image_path = os.path.join(path, snake_species_folder, image)
            if image_path.lower().endswith((".jpg", ".png",".jpeg",".JPG",".JPEG")):
                image_list.append(image_path)
                label_list.append(snake_species_folder)
    print(f"\n Nombre total d'images trouvées: {len(image_list)}\n")
    return image_list, label_list

path = "..\data\moroccan_snakes_V2"
image_list, label_list = extract_data(path)

label_binarizer = LabelBinarizer()
label_list = label_binarizer.fit_transform(label_list)

HEIGHT = 224
WIDTH = 224

batch_size=64

class CustomDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert("RGB")  
        label = self.label_list[idx]

        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented['image']

        return image, label

train_images, test_images, train_labels, test_labels = train_test_split(image_list, label_list, test_size=0.2, random_state=7)

def get_transforms(*, data):
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            RandomResizedCrop(WIDTH, HEIGHT, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(WIDTH, HEIGHT),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
train_transform = get_transforms(data='train')
valid_transform = get_transforms(data='valid')

train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
test_dataset = CustomDataset(test_images, test_labels, transform=valid_transform)

train_loader_stage2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader_stage2 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
