# accum count number per class(class3, class4) in the dataset and save the distribution to a file name(XXX_class1class2_class3_class4_XX_XX.jpg) in the directory.
# count separately for class1, class2, class3 and class4
# example: class1: 01: ##, 02: ##, class2: 01: ##, 02: ##, 03: ##, 04: ##, 05: ##, 06: ##, 07: ##, class3: day: ##, sunrising: ##, sunset: ##, night: ##, class4: clear: ##, cloudy: ##, rainy: ##, backlight: ##
# class1: 01, 02
# class2: 01, 02, 03, 04, 05, 06, 07
# class3: 'day', 'sunrising', 'sunset', 'night'
# class4: 'clear', 'cloudy', 'rainy', 'backlight'
# example: gyeonggi_0106_day_clear_0048_0234.jpg class1: 01, class2: 06, class3: day, class4: clear
# plot the distribution of the classes in the dataset and save the plot to a file name(class_distribution.jpg) in the directory and save the txt file to a file name(class_distribution.txt).

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
            
def count_class_per_image(folder_path):
    # 폴더 안의 파일 이름을 리스트로 가져옵니다.
    file_names = sorted([file for file in os.listdir(folder_path) if file.endswith('.jpg')])
    class1 = {'01': 0, '02': 0}
    class2 = {'01': 0, '02': 0, '03': 0, '04': 0, '05': 0, '06': 0, '07': 0}
    class3 = {'day': 0, 'sunrising': 0, 'sunset': 0, 'night': 0}
    class4 = {'clear': 0, 'cloudy': 0, 'rainy': 0, 'snow': 0, 'foggy': 0, 'backlight': 0}
    
    for i, file_name in enumerate(file_names):
        class1[file_name.split('_')[1][:2]] += 1
        class2[file_name.split('_')[1][2:]] += 1
        class3[file_name.split('_')[2]] += 1
        class4[file_name.split('_')[3]] += 1
            
    return class1, class2, class3, class4, len(file_names)

# 이미지가 있는 폴더 경로를 입력합니다.
folder_path = '/home/ailab/AILabDataset/01_Open_Dataset/13_AIHUB/303.특수환경_자율주행_3D_데이터_고도화/01-1.정식개방데이터/Training/01.원천데이터/image'
class1, class2, class3, class4, total_images = count_class_per_image(folder_path)

def calculate_ratios(class_dict, total):
    return {key: (value, round((value / total) * 100, 2)) for key, value in class_dict.items()}

class1_ratios = calculate_ratios(class1, total_images)
class2_ratios = calculate_ratios(class2, total_images)
class3_ratios = calculate_ratios(class3, total_images)
class4_ratios = calculate_ratios(class4, total_images)

# Plot the distribution of the classes in the dataset
plt.figure(figsize=(10, 6))
plt.bar(class1.keys(), class1.values(), color='b', label='class1')
plt.xlabel('class1')
plt.ylabel('count')
plt.title('Class1 Distribution')
plt.legend()
plt.savefig('/home/ailab/young/EfficientNet-PyTorch/class_distribution/class1_distribution.jpg')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(class2.keys(), class2.values(), color='g', label='class2')
plt.xlabel('class2')
plt.ylabel('count')
plt.title('Class2 Distribution')
plt.legend()
plt.savefig('/home/ailab/young/EfficientNet-PyTorch/class_distribution/class2_distribution.jpg')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(class3.keys(), class3.values(), color='r', label='class3')
plt.xlabel('class3')
plt.ylabel('count')
plt.title('Class3 Distribution')
plt.legend()
plt.savefig('/home/ailab/young/EfficientNet-PyTorch/class_distribution/class3_distribution.jpg')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(class4.keys(), class4.values(), color='y', label='class4')
plt.xlabel('class4')
plt.ylabel('count')
plt.title('Class4 Distribution')
plt.legend()
plt.savefig('/home/ailab/young/EfficientNet-PyTorch/class_distribution/class4_distribution.jpg')
plt.show()

# Save the txt file (include the distribution(count and ratio))
with open('/home/ailab/young/EfficientNet-PyTorch/class_distribution/class_distribution.txt', 'w') as f:
    f.write("Class\n")
    f.write("class1: 01, 02\n")
    f.write("class2: 01, 02, 03, 04, 05, 06, 07\n")
    f.write("class3: 'day', 'sunrising', 'sunset', 'night'\n")
    f.write("class4: 'clear', 'cloudy', 'rainy', 'snow', 'foggy', 'backlight'\n\n")
    f.write("Class distribution(#, %):\n")
    f.write(f"class1: {class1_ratios}\n")
    f.write(f"class2: {class2_ratios}\n")
    f.write(f"class3: {class3_ratios}\n")
    f.write(f"class4: {class4_ratios}\n")
    
# Save the distribution of the classes in the dataset, subplot(2,2)
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax[0, 0].bar(class1.keys(), class1.values(), color='b')
ax[0, 0].set_title('Class1 Distribution')
ax[0, 1].bar(class2.keys(), class2.values(), color='g')
ax[0, 1].set_title('Class2 Distribution')
ax[1, 0].bar(class3.keys(), class3.values(), color='r')
ax[1, 0].set_title('Class3 Distribution')
ax[1, 1].bar(class4.keys(), class4.values(), color='y')
ax[1, 1].set_title('Class4 Distribution')
plt.tight_layout()
plt.savefig('/home/ailab/young/EfficientNet-PyTorch/class_distribution/class_distribution.jpg')
plt.show()
