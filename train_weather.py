import os
import torch
import numpy as np
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import gc
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from data_loader import load_file_list, get_data_loaders, print_class_distribution, save_class_distribution, balanced_sampling, sample_files_by_class, save_sampled_images, WeatherDataset, transform
from metric import precision_recall_f1score, plot_confusion_matrix
from plot import plot_metrics, plot_precision_recall_curve

# 경로 설정
train_val_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/training/image_2'
test_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/testing/image_2'
imagesets_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/ImageSets'
train_setting = 'e2_lr0001_s01'
metrics_save_dir = f'results/train/{train_setting}'
# 샘플 이미지 저장 경로
train_save_dir = 'samples/train'
val_save_dir = 'samples/val'
test_save_dir = 'samples/test'

# 파일 리스트 로드
train_files = load_file_list(os.path.join(imagesets_dir, 'train.txt'))
val_files = load_file_list(os.path.join(imagesets_dir, 'val.txt'))
test_files = load_file_list(os.path.join(imagesets_dir, 'test.txt'))

# 클래스 라벨 추출
train_labels = [WeatherDataset([f], train_val_dir, None).load_climate_label(f) for f in train_files]
val_labels = [WeatherDataset([f], train_val_dir, None).load_climate_label(f) for f in val_files]
test_labels = [WeatherDataset([f], test_dir, None).load_climate_label(f) for f in test_files]

# 클래스 분포 출력 및 저장
print_class_distribution(train_labels, "Train")
print_class_distribution(val_labels, "Validation")
print_class_distribution(test_labels, "Test")

# 기존 샘플링 방법 유지
train_files_sampled, train_labels_sampled = balanced_sampling(train_files, train_labels)
val_files_sampled, val_labels_sampled = balanced_sampling(val_files, val_labels)
test_files_sampled, test_labels_sampled = balanced_sampling(test_files, test_labels)

# 클래스 분포 출력 및 저장
print_class_distribution(train_labels_sampled, "Sampled Train")
print_class_distribution(val_labels_sampled, "Sampled Validation")
print_class_distribution(test_labels_sampled, "Sampled Test")

# 새로운 클래스별 샘플링 (각 클래스별 10개씩)
train_samples = sample_files_by_class(train_files, train_labels, n_samples=10)
val_samples = sample_files_by_class(val_files, val_labels, n_samples=10)
test_samples = sample_files_by_class(test_files, test_labels, n_samples=10)

# 샘플 이미지 저장
save_sampled_images(train_samples, train_val_dir, train_save_dir)
save_sampled_images(val_samples, train_val_dir, val_save_dir)
save_sampled_images(test_samples, test_dir, test_save_dir)

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

# Metrics dictionary
metrics = {
    'train_loss': [],
    'val_loss': [],
    'train_precision': [],
    'val_precision': [],
    'train_recall': [],
    'val_recall': [],
    'train_f1': [],
    'val_f1': []
}

# Training loop without Gradient Accumulation and with AMP
num_epochs = 2

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
            
            scaler.scale(loss_climate).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            running_loss += loss_climate.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})
            pbar.update(1)
            
            _, predicted_climate = torch.max(outputs_climate, 1)
            train_preds.extend(predicted_climate.cpu().numpy())
            train_true.extend(climate_labels.cpu().numpy())
    
    train_loss = running_loss / len(train_loader)
    train_precision, train_recall, train_f1_score = precision_recall_f1score(train_preds, train_true, average='macro')

    metrics['train_loss'].append(train_loss)
    metrics['train_precision'].append(train_precision)
    metrics['train_recall'].append(train_recall)
    metrics['train_f1'].append(train_f1_score)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}')

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
    val_outputs = []
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
            val_outputs.extend(outputs_climate.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    precision, recall, f1_score = precision_recall_f1score(val_preds, val_true, average='macro')

    metrics['val_loss'].append(avg_val_loss)
    metrics['val_precision'].append(precision)
    metrics['val_recall'].append(recall)
    metrics['val_f1'].append(f1_score)

    print(f'Validation Loss: {avg_val_loss}, Climate Accuracy: {100 * correct_climate / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

    # 가장 좋은 모델 가중치 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = model.state_dict()

# 가장 좋은 모델 가중치 저장
torch.save(best_model_wts, f'best_weather_model_{train_setting}.pth')

print('Finished Training')

# Save metrics plots
os.makedirs(metrics_save_dir, exist_ok=True)
plot_metrics(metrics, metrics_save_dir)

# Plot Precision-Recall curve
plot_precision_recall_curve(np.array(val_true), np.array(val_outputs), metrics_save_dir)

# Confusion Matrix for Train
plot_confusion_matrix(train_true, train_preds, classes=['Normal', 'Snowy', 'Rainy', 'Hazy'], train_setting=train_setting,  name='Train')

# 저장된 가장 좋은 모델 가중치를 로드
model.load_state_dict(torch.load(f'best_weather_model_{train_setting}.pth'))

# Evaluation on test dataset
model.eval()
correct_climate = 0
total = 0
test_preds = []
test_true = []
test_outputs = []

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
        test_outputs.extend(outputs_climate.cpu().numpy())

precision, recall, f1_score = precision_recall_f1score(test_preds, test_true, average='macro')
print(f'Accuracy on test dataset: Climate: {100 * correct_climate / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

# save class distribution and test metrics to txt file
with open(f'results/train/{train_setting}/metrics_and_class_distribution.txt', 'w') as f:
    f.write("Class\n")
    f.write("0-Normal, 1-Snowy, 2-Rainy, 3-Hazy\n\n")
    f.write("Class distribution:\n")
    unique, counts = np.unique(train_labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    f.write(f"Train: {distribution}\n")
    
    unique, counts = np.unique(val_labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    f.write(f"Validation: {distribution}\n")
    
    unique, counts = np.unique(test_labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    f.write(f"Test: {distribution}\n")
    
    unique, counts = np.unique(train_labels_sampled, return_counts=True)
    distribution = dict(zip(unique, counts))
    f.write(f"Sampled Train: {distribution}\n")
    
    unique, counts = np.unique(val_labels_sampled, return_counts=True)
    distribution = dict(zip(unique, counts))
    f.write(f"Sampled Validation: {distribution}\n")
    
    unique, counts = np.unique(test_labels_sampled, return_counts=True)
    distribution = dict(zip(unique, counts))
    f.write(f"Sampled Test: {distribution}\n")
    
    f.write("\nMetrics on test dataset:\n")
    f.write(f'Accuracy: {100 * correct_climate / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

# Confusion Matrix for Test
plot_confusion_matrix(test_true, test_preds, classes=['Normal', 'Snowy', 'Rainy', 'Hazy'], train_setting=train_setting, name='Test')
