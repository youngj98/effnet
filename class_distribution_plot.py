import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

def load_xml(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_txt(image_path, gt_info, dst):
    with open(dst, 'w') as file:
        for image, gt in zip(image_path, gt_info):
            file.write(image + " " + str(int(gt)) + "\n")

def plot_distribution(class_counts, class_labels, title, save_path):
    plt.figure(figsize=(10, 6))
    # plt.bar(class_labels, class_counts, color=['blue', 'orange'])
    plt.bar(class_labels, class_counts, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()  

def count_classes(gt_info):
    class_counts = np.zeros(2)
    for gt in gt_info:
        class_counts[int(gt)] += 1
    return class_counts

def write_class_distribution_txt(class_counts, class_labels, save_path):
    total_count = np.sum(class_counts)
    with open(save_path, 'w') as file:
        file.write("Class Distribution\n")
        for label, count in zip(class_labels, class_counts):
            percentage = (count / total_count) * 100
            file.write(f"{label}: {int(count)}장 ({percentage:.2f}%)\n")

if __name__ == "__main__":
    target_folders = ["daytime_clear", "daytime_foggy", "dusk_rainy", "night_rainy", "Night-Sunny"]

    class_labels = ['daytime', 'night']
    # class_labels = ['clear', 'foggy', 'rainy']

    all_image_path = np.array([])
    all_gt_info = np.array([])
    
    for target_folder in target_folders:
        data_paths = glob("/home/ailab/AILabDataset/01_Open_Dataset/37_WeatherDataset/DWD/{}/VOC2007/Annotations/*.xml".format(target_folder))
        for data_path in tqdm(data_paths):
            image_path = data_path.replace("Annotations", "JPEGImages").replace("xml", "jpg")
            if not os.path.isfile(image_path):
                raise FileNotFoundError("Image file not found: ", image_path)
            
            xml_meta = load_xml(data_path)
            info_start = xml_meta.find("<timeofday>") + 11
            info_end = xml_meta.find("</timeofday>")
            # info_start = xml_meta.find("<weather>") + 9
            # info_end = xml_meta.find("</weather>")
            
            if info_end == -1:  # No </timeofday> tag found
                print(f"No <timeofday> tag found in XML: {data_path}")
                continue
            else:
                # store in gt
                gt = xml_meta[info_start:info_end]

            # if info_end == -1:
            #     gt = data_path.split(os.sep)[-1].split("-")[0]
            #     if "target" in gt:
            #         gt = "foggy"
            # else:
            #     gt = xml_meta[info_start:info_end]
            
            # if gt == 'clear' or gt == 'partly cloudy':
            #     gt_int = 0
            # elif gt == 'overcast':
            #     gt_int = 0
            # elif gt == 'foggy' or gt == 'haze' or gt == 'mist':
            #     gt_int = 1
            # elif gt == 'rainy' or gt == 'rain_storm':
            #     gt_int = 2
            # else:
            #     raise ValueError("Unknown weather type: ", gt)

            if gt == 'daytime':
                gt_int = 0
            elif gt == 'dawn/dusk':
                gt_int = 0
            elif gt == 'night':
                gt_int = 1
            else:
                raise ValueError("Unknown weather type: ", gt)

            all_image_path = np.append(all_image_path, image_path)
            all_gt_info = np.append(all_gt_info, gt_int)
    
    idx = np.arange(len(all_image_path))
    np.random.shuffle(idx)
    all_image_path = all_image_path[idx]
    all_gt_info = all_gt_info[idx]
    
    train_image_path, train_gt_info = np.array([]), np.array([])
    valid_image_path, valid_gt_info = np.array([]), np.array([])
    test_image_path, test_gt_info = np.array([]), np.array([])
    
    for i in range(4):
        idx = np.where(all_gt_info == i)[0]
        
        train_idx = idx[:int(len(idx) * 0.7)]
        valid_idx = idx[int(len(idx) * 0.7):int(len(idx) * 0.9)]
        test_idx = idx[int(len(idx) * 0.9):]
        
        train_image_path = np.append(train_image_path, all_image_path[train_idx])
        train_gt_info = np.append(train_gt_info, all_gt_info[train_idx])
        
        valid_image_path = np.append(valid_image_path, all_image_path[valid_idx])
        valid_gt_info = np.append(valid_gt_info, all_gt_info[valid_idx])
        
        test_image_path = np.append(test_image_path, all_image_path[test_idx])
        test_gt_info = np.append(test_gt_info, all_gt_info[test_idx])
    
    idx = np.arange(len(train_image_path))
    np.random.shuffle(idx)
    train_image_path = train_image_path[idx]
    train_gt_info = train_gt_info[idx]
    
    idx = np.arange(len(valid_image_path))
    np.random.shuffle(idx)
    valid_image_path = valid_image_path[idx]
    valid_gt_info = valid_gt_info[idx]
    
    idx = np.arange(len(test_image_path))
    np.random.shuffle(idx)
    test_image_path = test_image_path[idx]
    test_gt_info = test_gt_info[idx]
    
    os.makedirs('./data', exist_ok=True)
    write_txt(train_image_path, train_gt_info, './data/train_time.txt')
    write_txt(valid_image_path, valid_gt_info, './data/valid_time.txt')
    write_txt(test_image_path, test_gt_info, './data/test_time.txt')

    train_class_counts = count_classes(train_gt_info)
    valid_class_counts = count_classes(valid_gt_info)
    test_class_counts = count_classes(test_gt_info)
    
    plot_distribution(train_class_counts, class_labels, 'Train Dataset Class Distribution', '/home/ailab/young/EfficientNet-PyTorch/class_distri//time/train_class_distribution_time.jpg')
    plot_distribution(valid_class_counts, class_labels, 'Validation Dataset Class Distribution', '/home/ailab/young/EfficientNet-PyTorch/class_distri/time/valid_class_distribution_time.jpg')
    plot_distribution(test_class_counts, class_labels, 'Test Dataset Class Distribution', '/home/ailab/young/EfficientNet-PyTorch/class_distri/time/test_class_distribution_time.jpg')
    
    with open('/home/ailab/young/EfficientNet-PyTorch/class_distri/time/class_distribution.txt', 'w') as f:
        f.write("Class Distribution\n")
        for label, count in zip(class_labels, train_class_counts):
            f.write(f"Train {label}: {int(count)}\n")
        for label, count in zip(class_labels, valid_class_counts):
            f.write(f"Validation {label}: {int(count)}\n")
        for label, count in zip(class_labels, test_class_counts):
            f.write(f"Test {label}: {int(count)}\n")

    total_class_counts = count_classes(all_gt_info)

    write_class_distribution_txt(total_class_counts, class_labels, '/home/ailab/young/EfficientNet-PyTorch/class_distri/time/total_class_distribution.txt')
    plot_distribution(total_class_counts, class_labels, 'Total Dataset Class Distribution', '/home/ailab/young/EfficientNet-PyTorch/class_distri/time/total_class_distribution_time.jpg')
