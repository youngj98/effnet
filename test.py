import os
import torch
import numpy as np
import argparse
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
from data_loader import load_file_list, get_data_loaders, print_class_distribution, save_class_distribution, balanced_sampling, sample_files_by_class, save_sampled_images, NewWeatherDataset, transform, custom_collate_fn
import gc
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from metric import precision_recall_f1score, save_confusion_matrix
from plot import plot_metrics, plot_precision_recall_curve

def main(args):
    learning_rate = 0.001
    num_epochs = 10
    task = args.task
    train_setting = f'e{num_epochs}_lr{str(learning_rate).replace(".", "")}_{task}_02' 
    metrics_save_dir = f'results/test/{train_setting}'     
    os.makedirs(metrics_save_dir, exist_ok=True)
    
    # 파일 리스트 로드
    test_dir = '/home/ailab/AILabDataset/01_Open_Dataset/05_nuScenes/nuScenes/samples/CAM_FRONT'
    imagesets_dir = "./data"

    if task == "weather":
        test_files = load_file_list(os.path.join(imagesets_dir, 'test.txt'))
        class_names = ['clear', 'overcast', 'foggy', 'rainy']
    elif task == "time":
        test_files = load_file_list(os.path.join(imagesets_dir, 'test_time.txt'))
        class_names = ['daytime', 'night'] # dawn/dusk
    num_cls = len(class_names)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 데이터셋 및 데이터로더 생성
    test_dataset = NewWeatherDataset(test_files, test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    # Load the EfficientNet model
    model = EfficientNet.from_name('efficientnet-b5')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_ftrs = model._fc.in_features

    # Define a custom head for the model to predict time
    class CustomHead(torch.nn.Module):
        def __init__(self, num_ftrs, num_cls):
            super(CustomHead, self).__init__()
            self.fc_custom = torch.nn.Linear(num_ftrs, num_cls)
            
        def forward(self, x):
            return self.fc_custom(x)

    model._fc = CustomHead(num_ftrs, num_cls)
    model._fc = model._fc.to(device)  # Ensure the final layer is on the correct device

    # Evaluation on test dataset
    model.load_state_dict(torch.load(args.model), strict=False)

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
    
    # Precision-Recall Graph for Test
    plot_precision_recall_curve(np.array(test_true), np.array(test_outputs), class_names, metrics_save_dir)

    # Confusion Matrix for Test
    save_confusion_matrix(test_true, test_preds, classes=class_names, metrics_save_dir=metrics_save_dir+'/confusion_matrix') # classes = class_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time, Weather Classification Inference")
    parser.add_argument("--task", choices=["weather", "time"], help="choose the task : 'weather of 'time'")
    parser.add_argument("--model", type=str, required=True, help="Path to the model weights")
    args = parser.parse_args()
    
    main(args)
