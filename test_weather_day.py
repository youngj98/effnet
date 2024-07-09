import os
import torch
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

# 경로 설정
test_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/testing/image_2'
imagesets_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/ImageSets'

# Transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
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
        day_label = self.load_day_label(self.file_list[idx])

        return image, (climate_label, day_label)

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
    
    def load_day_label(self, file_id):
        try:
            day_condition = file_id.split('_')[1][1]  # 두 번째 문자가 낮/밤 정보
            if day_condition == 'D':  # Day
                label = 0
            elif day_condition == "N":  # Night
                label = 1
            else:
                label = -1  # Unknown Label, ERROR
        except IndexError:
            label = -1  # 파일 이름이 예상한 형식과 다르면 에러 처리
        return label

# 파일 리스트 로드 함수
def load_file_list(filepath):
    with open(filepath, 'r') as file:
        return file.read().splitlines()

# 파일 리스트 로드
test_files = load_file_list(os.path.join(imagesets_dir, 'test.txt'))

# 10% 샘플링
test_files_sampled, _ = train_test_split(test_files, train_size=0.1, random_state=42)

# Custom collate function
def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    climate_labels, day_labels = zip(*labels)
    climate_labels = torch.tensor(climate_labels)
    day_labels = torch.tensor(day_labels)

    return images, (climate_labels, day_labels)

# 데이터셋 및 데이터로더 생성
test_dataset = WeatherDataset(test_files_sampled, test_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

# Load the EfficientNet model
model = EfficientNet.from_name('efficientnet-b5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
num_ftrs = model._fc.in_features

# Define a custom head for the model to predict climate and day
class CustomHead(torch.nn.Module):
    def __init__(self, num_ftrs):
        super(CustomHead, self).__init__()
        self.fc_climate = torch.nn.Linear(num_ftrs, 4)  # Assuming 4 classes for climate
        self.fc_day = torch.nn.Linear(num_ftrs, 2)  # Assuming 2 classes for day

    def forward(self, x):
        climate = self.fc_climate(x)
        day = self.fc_day(x)
        return climate, day

model._fc = CustomHead(num_ftrs)
model._fc = model._fc.to(device)  # Ensure the final layer is on the correct device

# 저장된 모델 가중치를 로드
model.load_state_dict(torch.load('best_weather_model.pth'))

# Evaluation on test dataset
model.eval()
correct_climate = 0
correct_day = 0
total = 0

with torch.no_grad():
    for inputs, (climate_labels, day_labels) in test_loader:
        inputs, climate_labels, day_labels = inputs.to(device), climate_labels.to(device), day_labels.to(device)
        with autocast():
            outputs_climate, outputs_day = model(inputs)
        _, predicted_climate = torch.max(outputs_climate, 1)
        _, predicted_day = torch.max(outputs_day, 1)
        total += climate_labels.size(0)
        correct_climate += (predicted_climate == climate_labels).sum().item()
        correct_day += (predicted_day == day_labels).sum().item()

print(f'Accuracy on test dataset: Climate: {100 * correct_climate / total}%, Day: {100 * correct_day / total}%')

# 각 이미지에 대한 결과 출력
for i, (inputs, (climate_labels, day_labels)) in enumerate(test_loader):
    inputs = inputs.to(device)
    with torch.no_grad():
        with autocast():
            outputs_climate, outputs_day = model(inputs)
        _, predicted_climate = torch.max(outputs_climate, 1)
        _, predicted_day = torch.max(outputs_day, 1)
    
    for j in range(inputs.size(0)):
        print(f"Image: {test_files_sampled[i * test_loader.batch_size + j]}.jpg, Predicted Climate: {predicted_climate[j].item()}, True Climate: {climate_labels[j].item()}, Predicted Day: {predicted_day[j].item()}, True Day: {day_labels[j].item()}")
