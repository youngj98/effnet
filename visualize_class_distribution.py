import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse

def load_class_labels(file_path):
    """
    텍스트 파일에서 클래스 레이블을 추출합니다.
    각 행의 마지막 값이 클래스입니다.
    """
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # 공백으로 분리하여 마지막 값을 클래스로 추출
                parts = line.split()
                if len(parts) >= 2:
                    labels.append(int(parts[-1]))
    return labels

def get_class_distribution(labels):
    """
    클래스 분포를 계산합니다.
    """
    return Counter(labels)

def plot_class_distribution(train_dist, valid_dist, test_dist, class_names=None, save_dir='./'):
    """
    Train, Valid, Test 데이터셋의 클래스 분포를 시각화합니다.
    """
    # 모든 클래스 수집
    all_classes = sorted(set(list(train_dist.keys()) + list(valid_dist.keys()) + list(test_dist.keys())))
    
    # 클래스 이름 설정
    if class_names is None:
        class_names = [f'Class {i}' for i in all_classes]
    
    # 각 데이터셋의 분포를 배열로 변환
    train_counts = [train_dist.get(cls, 0) for cls in all_classes]
    valid_counts = [valid_dist.get(cls, 0) for cls in all_classes]
    test_counts = [test_dist.get(cls, 0) for cls in all_classes]
    
    # 그래프 설정
    x = np.arange(len(all_classes))
    width = 0.25
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 전체 비교 (Bar Chart)
    ax1 = axes[0, 0]
    ax1.bar(x - width, train_counts, width, label='Train', alpha=0.8)
    ax1.bar(x, valid_counts, width, label='Valid', alpha=0.8)
    ax1.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Train 데이터셋 분포 (Pie Chart)
    ax2 = axes[0, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_classes)))
    ax2.pie(train_counts, labels=class_names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Train Dataset Distribution', fontsize=14, fontweight='bold')
    
    # 3. Valid 데이터셋 분포 (Pie Chart)
    ax3 = axes[1, 0]
    ax3.pie(valid_counts, labels=class_names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Validation Dataset Distribution', fontsize=14, fontweight='bold')
    
    # 4. Test 데이터셋 분포 (Pie Chart)
    ax4 = axes[1, 1]
    ax4.pie(test_counts, labels=class_names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title('Test Dataset Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 저장
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved class distribution plot to {save_path}')
    plt.show()
    
    # 통계 정보 출력 및 저장
    stats_path = os.path.join(save_dir, 'class_distribution_stats.txt')
    with open(stats_path, 'w') as f:
        total_train = sum(train_counts)
        total_valid = sum(valid_counts)
        total_test = sum(test_counts)
        
        output = []
        output.append("=" * 60)
        output.append("Class Distribution Statistics")
        output.append("=" * 60)
        output.append("")
        
        output.append(f"Total samples - Train: {total_train}, Valid: {total_valid}, Test: {total_test}")
        output.append("")
        
        output.append("-" * 60)
        output.append(f"{'Class':<15} {'Train':<15} {'Valid':<15} {'Test':<15}")
        output.append("-" * 60)
        
        for i, cls in enumerate(all_classes):
            class_name = class_names[i] if i < len(class_names) else f'Class {cls}'
            train_pct = (train_counts[i] / total_train * 100) if total_train > 0 else 0
            valid_pct = (valid_counts[i] / total_valid * 100) if total_valid > 0 else 0
            test_pct = (test_counts[i] / total_test * 100) if total_test > 0 else 0
            
            output.append(f"{class_name:<15} {train_counts[i]:<6} ({train_pct:5.2f}%)  "
                         f"{valid_counts[i]:<6} ({valid_pct:5.2f}%)  "
                         f"{test_counts[i]:<6} ({test_pct:5.2f}%)")
        
        output.append("-" * 60)
        
        # 화면 및 파일에 출력
        for line in output:
            print(line)
            f.write(line + '\n')
    
    print(f'\nSaved statistics to {stats_path}')

def main():
    parser = argparse.ArgumentParser(description='Visualize class distribution from train, valid, test txt files')
    parser.add_argument('--train', type=str, default='./data/train.txt', help='Path to train.txt file')
    parser.add_argument('--valid', type=str, default='./data/valid.txt', help='Path to valid.txt file')
    parser.add_argument('--test', type=str, default='./data/test.txt', help='Path to test.txt file')
    parser.add_argument('--class-names', nargs='+', type=str, default=None, 
                        help='Class names (e.g., --class-names Clear Snowy Rainy Foggy)')
    parser.add_argument('--save-dir', type=str, default='./class_distribution', 
                        help='Directory to save the plots')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    for file_path, name in [(args.train, 'train'), (args.valid, 'valid'), (args.test, 'test')]:
        if not os.path.exists(file_path):
            print(f"Error: {name} file not found at {file_path}")
            return
    
    # 클래스 레이블 로드
    print("Loading class labels...")
    train_labels = load_class_labels(args.train)
    valid_labels = load_class_labels(args.valid)
    test_labels = load_class_labels(args.test)
    
    print(f"Loaded {len(train_labels)} train samples")
    print(f"Loaded {len(valid_labels)} validation samples")
    print(f"Loaded {len(test_labels)} test samples")
    print()
    
    # 클래스 분포 계산
    train_dist = get_class_distribution(train_labels)
    valid_dist = get_class_distribution(valid_labels)
    test_dist = get_class_distribution(test_labels)
    
    # 시각화
    plot_class_distribution(train_dist, valid_dist, test_dist, 
                           class_names=args.class_names, 
                           save_dir=args.save_dir)

if __name__ == '__main__':
    main()
