import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random
import shutil

# Transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom dataset class
class WeatherDataset(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx] + '.jpg')
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Load the labels from the file name
        climate_label = self.load_climate_label(self.file_list[idx])

        return image, climate_label

    def load_climate_label(self, file_id):
        try:
            weather_condition = file_id.split('_')[1][0]  # 첫 번째 문자가 기후 정보
            if weather_condition == 'N':  # Normal
                label = 0
            elif weather_condition == "S":  # Snowy
                label = 1
            elif weather_condition == "R":  # Rainy
                label = 2
            elif weather_condition == "H":  # Hazy
                label = 3
            else:
                label = -1  # Unknown Label, ERROR
        except IndexError:
            label = -1  # 파일 이름이 예상한 형식과 다르면 에러 처리
        return label
    
class NewWeatherDataset(Dataset):
    def __init__(self, file_list, data_dir, transform=None):
        self.file_list = file_list
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name, gt = self.file_list[idx].split(" ")
        
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(gt)


# 파일 리스트 로드 함수
def load_file_list(filepath):
    with open(filepath, 'r') as file:
        return file.read().splitlines()

# Custom collate function
def custom_collate_fn(batch):
    images, labels = zip(*batch)

    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
    else:
        images = torch.stack([transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img for img in images])

    # labels = torch.tensor(labels, dtype=torch.long)
    # images = torch.stack(images)
    custom_labels = torch.tensor(labels, dtype=torch.long)

    return images, custom_labels

def print_class_distribution(labels, dataset_name):
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"Class distribution in {dataset_name} dataset: {distribution}")

def save_class_distribution(labels, dataset_name, save_dir):
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    with open(os.path.join(save_dir, f"{dataset_name}_class_distribution.txt"), 'w') as f:
        f.write(f"Class distribution in {dataset_name} dataset: {distribution}\n")
    print(f"Class distribution for {dataset_name} saved to {save_dir}")

def get_data_loaders(train_files, val_files, test_files, transform, batch_size=16):
    # Create datasets
    train_dataset = NewWeatherDataset(train_files, transform)
    val_dataset = NewWeatherDataset(val_files, transform)
    test_dataset = NewWeatherDataset(test_files, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader, test_loader

def balanced_sampling(file_list, labels):
    unique, counts = np.unique(labels, return_counts=True)
    min_count = min(counts)
    sample_size = int(min_count * 0.2)

    sampled_files = []
    sampled_labels = []

    for label in unique:
        label_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        sampled_indices = np.random.choice(label_indices, size=sample_size, replace=False)
        sampled_files.extend([file_list[i] for i in sampled_indices])
        sampled_labels.extend([labels[i] for i in sampled_indices])
    
    return sampled_files, sampled_labels

def sample_files_by_class(files, labels, n_samples=10):
    sampled_files = {0: [], 1: [], 2: [], 3: []}  # assuming four classes: 0-Normal, 1-Snowy, 2-Rainy, 3-Hazy
    for label in sampled_files.keys():
        label_files = [file for file, lbl in zip(files, labels) if lbl == label]
        sampled_files[label] = random.sample(label_files, min(n_samples, len(label_files)))
    return sampled_files

def save_sampled_images(sampled_files, root_dir, save_dir):
    for label, files in sampled_files.items():
        if label == 0:
            label_dir = os.path.join(save_dir, "Normal")
        elif label == 1:
            label_dir = os.path.join(save_dir, "Snowy")
        elif label == 2:
            label_dir = os.path.join(save_dir, "Rainy")
        elif label == 3:
            label_dir = os.path.join(save_dir, "Hazy")
        os.makedirs(label_dir, exist_ok=True)
        # remove existing files
        existing_files = os.listdir(label_dir)
        for file in existing_files:
            os.remove(os.path.join(label_dir, file))
            
        for file in files:
            src = os.path.join(root_dir, file + '.jpg')
            dst = os.path.join(label_dir, file + '.jpg')
            shutil.copyfile(src, dst)
