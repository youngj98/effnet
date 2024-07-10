import os
import numpy as np
from data_loader import load_file_list, get_data_loaders, print_class_distribution, save_class_distribution, balanced_sampling, sample_files_by_class, save_sampled_images, WeatherDataset, transform
from metric import precision_recall_f1score, plot_confusion_matrix
from plot import plot_metrics, plot_precision_recall_curve

# 경로 설정
train_val_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/training/image_2'
test_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/testing/image_2'
imagesets_dir = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/Climate/085.다양한 기상 상황 주행 데이터/01.RawDataset/ImageSets'
# 샘플 이미지 저장 경로
train_save_dir = 'data_sampling/train'
val_save_dir = 'data_sampling/val'
test_save_dir = 'data_sampling/test'

# 파일 리스트 로드
train_files = load_file_list(os.path.join(imagesets_dir, 'train.txt'))
val_files = load_file_list(os.path.join(imagesets_dir, 'val.txt'))
test_files = load_file_list(os.path.join(imagesets_dir, 'test.txt'))

# 클래스 라벨 추출
train_labels = [WeatherDataset([f], train_val_dir, None).load_climate_label(f) for f in train_files]
val_labels = [WeatherDataset([f], train_val_dir, None).load_climate_label(f) for f in val_files]
test_labels = [WeatherDataset([f], test_dir, None).load_climate_label(f) for f in test_files]

# 기존 샘플링 방법 유지
train_files_sampled, train_labels_sampled = balanced_sampling(train_files, train_labels)
val_files_sampled, val_labels_sampled = balanced_sampling(val_files, val_labels)
test_files_sampled, test_labels_sampled = balanced_sampling(test_files, test_labels)

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

# save class distribution and test metrics to txt file
with open(f'data_sampling/class_distribution.txt', 'w') as f:
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
