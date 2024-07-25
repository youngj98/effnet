import os
import matplotlib.pyplot as plt
from PIL import Image

def load_and_plot_images(folder_path):
    # 폴더 안의 파일 이름을 리스트로 가져옵니다.
    file_names = sorted([file for file in os.listdir(folder_path) if file.endswith('.jpg')])

    # 각 이미지를 순서대로 불러와서 10번에 1번 plot합니다.
    for i, file_name in enumerate(file_names):
        if i % 50 == 0:
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title(file_name)
            plt.show()
            plt.close()

# 이미지가 있는 폴더 경로를 입력합니다.
folder_path = '/home/ailab/young/303.특수환경_자율주행_3D_데이터_고도화/01-1.정식개방데이터/Validation/01.원천데이터/0104'
load_and_plot_images(folder_path)
