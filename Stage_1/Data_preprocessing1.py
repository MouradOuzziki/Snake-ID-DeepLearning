import cv2
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from albumentations import Compose, Normalize, Resize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from albumentations import  HorizontalFlip, VerticalFlip, RandomBrightnessContrast, RandomResizedCrop

# Check if GPU is available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



metadata = pd.read_csv("SnakeCLEF2021_train_metadata_PROD.csv")
min_train_metadata = pd.read_csv("SnakeCLEF2021_min-train_metadata_PROD.csv")


# List of old paths and corresponding new paths for each folder
old_paths = ['/Datasets/SnakeCLEF-2021/inaturalist/', '/Datasets/SnakeCLEF-2021/flickr/', '/Datasets/SnakeCLEF-2021/herpmapper/']
new_paths = ['..\\Datasets\\SnakeCLEF-2021\\inaturalist\\',
             '..\\Datasets\\SnakeCLEF-2021\\flickr\\',
             '..\\Datasets\\SnakeCLEF-2021\\herpmapper\\']

# Update the paths in the "image_path" column for each folder
for old_path, new_path in zip(old_paths, new_paths):
    metadata['image_path'] = metadata['image_path'].apply(lambda x: x.replace(old_path, new_path))


train_metadata = min_train_metadata
val_metadata = metadata[metadata['subset'] == 'val']


HEIGHT = 224
WIDTH = 224
N_CLASSES = 772
BATCH_SIZE = 128
num_workers = 8

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['image_path'].values[idx]
        label = self.df['class_id'].values[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label
    
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
    
train_dataset_stage1 = TrainDataset(train_metadata, transform=get_transforms(data='train'))
valid_dataset_stage1 = TrainDataset(val_metadata, transform=get_transforms(data='valid'))

train_loader = DataLoader(train_dataset_stage1, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset_stage1, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)