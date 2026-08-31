import os
import torch
import numpy as np
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import gc
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from data_loader import load_file_list, get_data_loaders, print_class_distribution, save_class_distribution, balanced_sampling, sample_files_by_class, save_sampled_images, WeatherDataset, transform, train_transform, eval_transform
from metric import precision_recall_f1score, plot_confusion_matrix
from plot import plot_metrics, plot_precision_recall_curve
import wandb
import argparse
import copy

# Define a custom head for the model to predict climate
class CustomHead(torch.nn.Module):
    def __init__(self, num_ftrs, num_cls):
        super(CustomHead, self).__init__()
        self.fc_output = torch.nn.Linear(num_ftrs, num_cls)  # Assuming 4 classes for climate

    def forward(self, x):
        result = self.fc_output(x)
        return result
        
def main(args):
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 64
    train_setting = f'time_e{num_epochs}_lr{str(learning_rate).replace(".", "")}'
    metrics_save_dir = f'results/train/{train_setting}'
    
    classes=['Daytime', 'Night']
    num_cls = len(classes)
    wandb.init(project="time_classification", config={
        "learning_rate": learning_rate,
        "architecture": "EfficientNet-B3",
        "dataset": "Time",
        "epochs": num_epochs,
    })

    # 파일 리스트 로드
    imagesets_dir = "./data"
    train_files = load_file_list(os.path.join(imagesets_dir, 'train_time_1.txt'))
    val_files = load_file_list(os.path.join(imagesets_dir, 'valid_time_1.txt'))
    test_files = load_file_list(os.path.join(imagesets_dir, 'test_time_1.txt'))

    # 데이터로더 생성
    train_loader, val_loader, test_loader = get_data_loaders(train_files, val_files, test_files, train_transform, eval_transform, batch_size)

    # Load the EfficientNet model
    model = EfficientNet.from_name('efficientnet-b3')
    # model = EfficientNet.from_pretrained('efficientnet-b3')
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpus[0]}')  # 첫 번째 GPU를 메인 디바이스로 설정
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    num_ftrs = model._fc.in_features
    model._fc = CustomHead(num_ftrs, num_cls)
    model._fc = model._fc.to(device)  # Ensure the final layer is on the correct device
    
    if torch.cuda.is_available() and len(args.gpus) > 1:
        # 여러 개의 GPU를 사용하는 경우 DataParallel 사용
        model = torch.nn.DataParallel(model, device_ids=args.gpus)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # LR 스케줄: warmup(lr x0.1 -> x1.0) 후 cosine annealing으로 eta_min까지 감쇠
    warmup_epochs = max(1, int(num_epochs * 0.03))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs])

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

    best_val_loss = float('inf')
    best_epoch = -1
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
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (i + 1)})
                pbar.update(1)
                
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_true.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_precision, train_recall, train_f1_score = precision_recall_f1score(train_preds, train_true, average='macro')

        metrics['train_loss'].append(train_loss)
        metrics['train_precision'].append(train_precision)
        metrics['train_recall'].append(train_recall)
        metrics['train_f1'].append(train_f1_score)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_score': train_f1_score
        }, step=epoch + 1)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}')

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_true = []
        val_outputs = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation', unit='batch'):
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                val_outputs.extend(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        precision, recall, f1_score = precision_recall_f1score(val_preds, val_true, average='macro')

        metrics['val_loss'].append(avg_val_loss)
        metrics['val_precision'].append(precision)
        metrics['val_recall'].append(recall)
        metrics['val_f1'].append(f1_score)

        wandb.log({
            'epoch': epoch + 1,
            'val_loss': avg_val_loss,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1_score': f1_score
        }, step=epoch + 1)

        print(f'Validation Loss: {avg_val_loss}, Time Accuracy: {100 * correct / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

        # 가장 좋은 모델 가중치 저장
        # state_dict()는 텐서의 참조만 돌려주므로 deepcopy가 없으면 이후 에폭의 갱신이
        # 그대로 반영되어 결국 '최종 에폭' 가중치가 저장된다.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'  -> best updated (epoch {best_epoch}, val_loss {best_val_loss:.4f})')

        # LR 스케줄 갱신: 배치 루프가 아니라 에폭 단위로 1회
        wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch + 1)
        scheduler.step()

    # 가장 좋은 모델 가중치 저장
    print(f'Best epoch: {best_epoch} (val_loss {best_val_loss:.4f})')
    torch.save(best_model_wts, f'best_model_{train_setting}_1.pth')

    print('Finished Training')

    # Save metrics plots
    os.makedirs(metrics_save_dir, exist_ok=True)
    plot_metrics(metrics, metrics_save_dir)

    # Plot Precision-Recall curve
    plot_precision_recall_curve(np.array(val_true), np.array(val_outputs), classes, metrics_save_dir)

    # Confusion Matrix for Train
    plot_confusion_matrix(train_true, train_preds, classes=classes, folder_name=train_setting,  name='Train')

    # 저장된 가장 좋은 모델 가중치를 로드
    model.load_state_dict(torch.load(f'best_model_{train_setting}_1.pth'))

    # Evaluation on test dataset
    model.eval()
    correct = 0
    total = 0
    test_preds = []
    test_true = []
    test_outputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
            test_outputs.extend(outputs.cpu().numpy())

    precision, recall, f1_score = precision_recall_f1score(test_preds, test_true, average='macro')
    print(f'Accuracy on test dataset: Time: {100 * correct / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

    wandb.log({
        'test_accuracy': 100 * correct / total,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1_score': f1_score
    })

    # save class distribution and test metrics to txt file
    # with open(f'results/train/{train_setting}/metrics_and_class_distribution.txt', 'w') as f:
    #     f.write("Class\n")
    #     f.write("0-Normal, 1-Snowy, 2-Rainy, 3-Hazy\n\n")
    #     f.write("Class distribution:\n")
    #     unique, counts = np.unique(train_labels, return_counts=True)
    #     distribution = dict(zip(unique, counts))
    #     f.write(f"Train: {distribution}\n")
        
    #     unique, counts = np.unique(val_labels, return_counts=True)
    #     distribution = dict(zip(unique, counts))
    #     f.write(f"Validation: {distribution}\n")
        
    #     unique, counts = np.unique(test_labels, return_counts=True)
    #     distribution = dict(zip(unique, counts))
    #     f.write(f"Test: {distribution}\n")
        
    #     unique, counts = np.unique(train_labels_sampled, return_counts=True)
    #     distribution = dict(zip(unique, counts))
    #     f.write(f"Sampled Train: {distribution}\n")
        
    #     unique, counts = np.unique(val_labels_sampled, return_counts=True)
    #     distribution = dict(zip(unique, counts))
    #     f.write(f"Sampled Validation: {distribution}\n")
        
    #     unique, counts = np.unique(test_labels_sampled, return_counts=True)
    #     distribution = dict(zip(unique, counts))
    #     f.write(f"Sampled Test: {distribution}\n")
        
    #     f.write("\nMetrics on test dataset:\n")
    #     f.write(f'Accuracy: {100 * correct_climate / total}%, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

    # Confusion Matrix for Test
    plot_confusion_matrix(test_true, test_preds, classes=classes, folder_name=train_setting, name='Test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Classification Training")
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    
    main(args)