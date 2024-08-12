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


train_setting = 'e10_lr0001_s01'
metrics_save_dir = f'results/train/{train_setting}_time'

os.makedirs(metrics_save_dir, exist_ok = True)

# 파일 리스트 로드
imagesets_dir = "./data"
train_files = load_file_list(os.path.join(imagesets_dir, 'train_time_1.txt'))
val_files = load_file_list(os.path.join(imagesets_dir, 'valid_time_1.txt'))
test_files = load_file_list(os.path.join(imagesets_dir, 'test_time_1.txt'))

# 데이터로더 생성
train_loader, val_loader, test_loader = get_data_loaders(train_files, val_files, test_files, transform)

# Load the EfficientNet model
model = EfficientNet.from_name('efficientnet-b5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
num_ftrs = model._fc.in_features

# Define a custom head for the model to predict time
class CustomHead(torch.nn.Module):
    def __init__(self, num_ftrs):
        super(CustomHead, self).__init__()
        self.fc_time = torch.nn.Linear(num_ftrs, 2)  # Assuming 3 classes for time [daytime, dawn/dusk, night]

    def forward(self, x):
        time = self.fc_time(x)
        return time

model._fc = CustomHead(num_ftrs)
model._fc = model._fc.to(device)  # Ensure the final layer is on the correct device

# Define loss and optimizer
criterion_time = torch.nn.CrossEntropyLoss()
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
num_epochs = 10

best_val_loss = float('inf') 
best_model_wts = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    train_preds = []
    train_true = []
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs_time = model(inputs)
                loss_time = criterion_time(outputs_time, labels)
            
            scaler.scale(loss_time).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            running_loss += loss_time.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})
            pbar.update(1)
            
            _, predicted_time = torch.max(outputs_time, 1)
            train_preds.extend(predicted_time.cpu().numpy())
            train_true.extend(labels.cpu().numpy())
    
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
    correct_time = 0
    total = 0
    val_preds = []
    val_true = []
    val_outputs = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs_time = model(inputs)
                loss_time = criterion_time(outputs_time, labels)
            val_loss += loss_time.item()
            _, predicted_time = torch.max(outputs_time, 1)
            total += labels.size(0)
            correct_time += (predicted_time == labels).sum().item()
            val_preds.extend(predicted_time.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            val_outputs.extend(outputs_time.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    precision, recall, f1_score = precision_recall_f1score(val_preds, val_true, average='macro')

    metrics['val_loss'].append(avg_val_loss)
    metrics['val_precision'].append(precision)
    metrics['val_recall'].append(recall)
    metrics['val_f1'].append(f1_score)


    print(f'Validation Loss: {avg_val_loss}, Time Accuracy: {100 * correct_time / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

    # 가장 좋은 모델 가중치 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = model.state_dict()

# 가장 좋은 모델 가중치 저장
torch.save(best_model_wts, f'best_time_model_{train_setting}.pth')

print('Finished Training')

# Save metrics plots
os.makedirs(metrics_save_dir, exist_ok=True)
plot_metrics(metrics, metrics_save_dir)

# Plot Precision-Recall curve
plot_precision_recall_curve(np.array(val_true), np.array(val_outputs), metrics_save_dir)

# Confusion Matrix for Train
plot_confusion_matrix(train_true, train_preds, classes=['daytime', 'night'], train_setting=train_setting,  name='Train')

# 저장된 가장 좋은 모델 가중치를 로드
model.load_state_dict(torch.load(f'best_time_model_{train_setting}.pth'))

# Evaluation on test dataset
model.eval()
correct_time = 0
total = 0
test_preds = []
test_true = []
test_outputs = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast():
            outputs_time = model(inputs)
        _, predicted_time = torch.max(outputs_time, 1)
        total += labels.size(0)
        correct_time += (predicted_time == labels).sum().item()
        test_preds.extend(predicted_time.cpu().numpy())
        test_true.extend(labels.cpu().numpy())
        test_outputs.extend(outputs_time.cpu().numpy())

precision, recall, f1_score = precision_recall_f1score(test_preds, test_true, average='macro')
print(f'Accuracy on test dataset: time: {100 * correct_time / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

# Confusion Matrix for Test
plot_confusion_matrix(test_true, test_preds, classes=['daytime', 'night'], train_setting=train_setting, name='Test')
