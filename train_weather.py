import os
import torch
import numpy as np
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import gc
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from data_loader import load_file_list, get_data_loaders, print_class_distribution, balanced_sampling, WeatherDataset, transform
from metric import precision_recall_f1score, plot_confusion_matrix

# 경로 설정
train_val_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/training/image_2'
test_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/testing/image_2'
imagesets_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/ImageSets'

# 파일 리스트 로드
train_files = load_file_list(os.path.join(imagesets_dir, 'train.txt'))
val_files = load_file_list(os.path.join(imagesets_dir, 'val.txt'))
test_files = load_file_list(os.path.join(imagesets_dir, 'test.txt'))

# 클래스 라벨 추출
train_labels = [WeatherDataset([f], train_val_dir, None).load_climate_label(f) for f in train_files]
val_labels = [WeatherDataset([f], train_val_dir, None).load_climate_label(f) for f in val_files]
test_labels = [WeatherDataset([f], test_dir, None).load_climate_label(f) for f in test_files]

# 클래스 분포 출력
print_class_distribution(train_labels, "Train")
print_class_distribution(val_labels, "Validation")
print_class_distribution(test_labels, "Test")

# 각 데이터셋에서 클래스별 분포를 유지하면서 10% 샘플링
train_files_sampled, train_labels_sampled = balanced_sampling(train_files, train_labels)
val_files_sampled, val_labels_sampled = balanced_sampling(val_files, val_labels)
test_files_sampled, test_labels_sampled = balanced_sampling(test_files, test_labels)

# 샘플링 후 클래스 분포 출력
print_class_distribution(train_labels_sampled, "Sampled Train")
print_class_distribution(val_labels_sampled, "Sampled Validation")
print_class_distribution(test_labels_sampled, "Sampled Test")

# 데이터로더 생성
train_loader, val_loader, test_loader = get_data_loaders(train_files_sampled, val_files_sampled, test_files_sampled, train_val_dir, test_dir, transform)

# Load the EfficientNet model
model = EfficientNet.from_name('efficientnet-b5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
num_ftrs = model._fc.in_features

# Define a custom head for the model to predict climate
class CustomHead(torch.nn.Module):
    def __init__(self, num_ftrs):
        super(CustomHead, self).__init__()
        self.fc_climate = torch.nn.Linear(num_ftrs, 4)  # Assuming 4 classes for climate

    def forward(self, x):
        climate = self.fc_climate(x)
        return climate

model._fc = CustomHead(num_ftrs)
model._fc = model._fc.to(device)  # Ensure the final layer is on the correct device

# Define loss and optimizer
criterion_climate = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

# Training loop with Gradient Accumulation and AMP
num_epochs = 10
accumulation_steps = 4  # number of steps to accumulate gradients

best_val_loss = float('inf')
best_model_wts = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    train_preds = []
    train_true = []
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for i, (inputs, climate_labels) in enumerate(train_loader):
            inputs, climate_labels = inputs.to(device), climate_labels.to(device)
            
            with autocast():
                outputs_climate = model(inputs)
                loss_climate = criterion_climate(outputs_climate, climate_labels)
                loss = loss_climate / accumulation_steps  # Normalize the loss to account for accumulation
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})
            pbar.update(1)
            
            _, predicted_climate = torch.max(outputs_climate, 1)
            train_preds.extend(predicted_climate.cpu().numpy())
            train_true.extend(climate_labels.cpu().numpy())
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_climate = 0
    total = 0
    val_preds = []
    val_true = []
    with torch.no_grad():
        for inputs, climate_labels in tqdm(val_loader, desc='Validation', unit='batch'):
            inputs, climate_labels = inputs.to(device), climate_labels.to(device)
            with autocast():
                outputs_climate = model(inputs)
                loss_climate = criterion_climate(outputs_climate, climate_labels)
            val_loss += loss_climate.item()
            _, predicted_climate = torch.max(outputs_climate, 1)
            total += climate_labels.size(0)
            correct_climate += (predicted_climate == climate_labels).sum().item()
            val_preds.extend(predicted_climate.cpu().numpy())
            val_true.extend(climate_labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    precision, recall, f1_score = precision_recall_f1score(val_preds, val_true, average='macro')
    print(f'Validation Loss: {avg_val_loss}, Climate Accuracy: {100 * correct_climate / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

    # 가장 좋은 모델 가중치 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = model.state_dict()

# 가장 좋은 모델 가중치 저장
torch.save(best_model_wts, 'best_weather_model.pth')

print('Finished Training')

# Confusion Matrix for Train
plot_confusion_matrix(train_true, train_preds, classes=['Normal', 'Snowy', 'Rainy', 'Hazy'], name='Train')

# 저장된 가장 좋은 모델 가중치를 로드
model.load_state_dict(torch.load('best_weather_model.pth'))

# Evaluation on test dataset
model.eval()
correct_climate = 0
total = 0
test_preds = []
test_true = []

with torch.no_grad():
    for inputs, climate_labels in tqdm(test_loader, desc='Testing', unit='batch'):
        inputs, climate_labels = inputs.to(device), climate_labels.to(device)
        with autocast():
            outputs_climate = model(inputs)
        _, predicted_climate = torch.max(outputs_climate, 1)
        total += climate_labels.size(0)
        correct_climate += (predicted_climate == climate_labels).sum().item()
        test_preds.extend(predicted_climate.cpu().numpy())
        test_true.extend(climate_labels.cpu().numpy())

precision, recall, f1_score = precision_recall_f1score(test_preds, test_true, average='macro')
print(f'Accuracy on test dataset: Climate: {100 * correct_climate / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

# Confusion Matrix for Test
plot_confusion_matrix(test_true, test_preds, classes=['Normal', 'Snowy', 'Rainy', 'Hazy'], name='Test')
