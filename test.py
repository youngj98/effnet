import os
import gc
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt 
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
from data_loader import load_file_list, get_data_loaders, print_class_distribution, save_class_distribution, balanced_sampling, sample_files_by_class, save_sampled_images, NewWeatherDataset, transform, custom_collate_fn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from metric import precision_recall_f1score, save_confusion_matrix
from plot import plot_metrics, plot_precision_recall_curve, plot_image_with_predictions


def main(args):
    learning_rate = 0.001
    num_epochs = 10
    # threshold = 0.8
    task = args.task
    train_setting = f'e{num_epochs}_lr{str(learning_rate).replace(".", "")}_{task}' 
    metrics_save_dir = f'results/test/{train_setting}'     
    os.makedirs(metrics_save_dir, exist_ok=True)
    
    # 파일 리스트 로드
    # test_dir = '/home/ailab/AILabDataset/01_Open_Dataset/05_nuScenes/nuScenes/samples/CAM_FRONT'
    # test_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/303.특수환경_자율주행_3D_데이터_고도화/01-1.정식개방데이터/Training/01.원천데이터/image'
    imagesets_dir = "./data"

    if task == "weather":
        # test_files = load_file_list(os.path.join(imagesets_dir, 'test.txt'))
        class_names = ['clear', 'overcast', 'foggy', 'rainy']
    elif task == "time":
        # test_files = load_file_list(os.path.join(imagesets_dir, 'test_time_1.txt'))
        test_dir = "/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/303.특수환경_자율주행_3D_데이터_고도화/01-1.정식개방데이터/Training/01.원천데이터/image"
        class_names = ['daytime', 'night'] # dawn/dusk
    num_cls = len(class_names)
    test_files = [f"{os.path.join(test_dir, f)} 0" for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # 데이터셋 및 데이터로더 생성
    test_dataset = NewWeatherDataset(test_files, transform)
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False, collate_fn=custom_collate_fn)

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

    output_folder = os.path.join(metrics_save_dir, f'output_images_{task}_AIhub')
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)

            print(f"True Labels: {labels.cpu().numpy()}")

            with autocast():
                outputs = model(inputs)
            
            print("Raw outputs :", outputs.cpu().numpy())

            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
            print("Probabilities:", probabilities)

            _, predicted = torch.max(outputs, 1)
            
            print("Predicted Labels:", predicted.cpu().numpy())
            
            total += labels.size(0)
            correct_time += (predicted == labels).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
            test_outputs.extend(outputs.cpu().numpy())

            for i in range(inputs.size(0)):
                image_tensor = inputs[i]  
                # prob = probabilities[i].cpu().numpy() 
                prob = probabilities[i]
                true_label = labels[i].item()  
                pred_label = predicted[i].item()  
                file_name = os.path.join(output_folder, f'image_{i}.png')

                plt.figure()
                plot_image_with_predictions(image_tensor, prob, true_label, pred_label, file_name, class_names)
                plt.close()

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
