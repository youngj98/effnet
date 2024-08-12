from glob import glob
import numpy as np
import os, sys
from tqdm import tqdm

def load_xml(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def write_txt(image_path, gt_info, dst):
    with open(dst, 'w') as file:
        for image, gt in zip(image_path, gt_info):
            file.write(image + " " + str(int(gt)) + "\n")
            
if __name__ == "__main__":
    # Load all xml files in target folder
    target_folders = ["daytime_clear", "daytime_foggy","dusk_rainy", "night_rainy", "Night-Sunny"]
    
    
    all_image_path = np.array([])
    all_gt_info = np.array([])
    
    # add list all image path and gt info
    for target_folder in target_folders:
        data_paths = glob("/home/ailab/AILabDataset/01_Open_Dataset/37_WeatherDataset/DWD/{}/VOC2007/Annotations/*.xml".format(target_folder))
        for data_path in tqdm(data_paths):
            
            image_path = data_path.replace("Annotations", "JPEGImages").replace("xml", "jpg")
            if os.path.isfile(image_path) == False:
                raise("Image file not found: ", image_path)
            xml_meta = load_xml(data_path)
            
            # add length of "<weather>" to get the start of the weather info
            info_start = xml_meta.find("<weather>") + 9
            info_end = xml_meta.find("</weather>")
            
            # Can't find </weather> info
            if info_end == -1:
                # if other xml file, load label from other strategy
                gt = data_path.split(os.sep)[-1].split("-")[0]
                # some of daytime_foggy situtation
                if "target" in gt:
                    gt = "foggy"
            else:
                gt = xml_meta[info_start:info_end]
                
            # gt mapping
            if gt == 'clear' or gt =='partly cloudy':
                gt_int = 0
            elif gt == 'overcast':
                gt_int = 1
            elif gt == 'foggy' or gt == 'haze' or gt == 'mist':
                gt_int = 2
            elif gt == 'rainy' or gt == 'rain_storm':
                gt_int = 3
            else:
                raise("Unknown weather type: ", gt)

            all_image_path = np.append(all_image_path, image_path)
            all_gt_info = np.append(all_gt_info, gt_int)
            
    # shuffle all data
    idx = np.arange(len(all_image_path))
    np.random.shuffle(idx)
    all_image_path = all_image_path[idx]
    all_gt_info = all_gt_info[idx]
        
    train_image_path = np.array([])
    train_gt_info = np.array([])
    
    valid_image_path = np.array([])
    valid_gt_info = np.array([])
    
    test_image_path = np.array([])
    test_gt_info = np.array([])
    
    '''
    class 0 : clear
    class 1 : overcast
    class 2 : foggy
    class 3 : rainy
    '''
    for i in range(4):
        # get index of each class
        idx = np.where(all_gt_info == i)[0]
        
        # split each class to train, valid, test
        train_idx = idx[:int(len(idx) * 0.7)]
        valid_idx = idx[int(len(idx) * 0.7):int(len(idx) * 0.9)]
        test_idx = idx[int(len(idx) * 0.9):]
        
        train_image_path = np.append(train_image_path, all_image_path[train_idx])
        train_gt_info = np.append(train_gt_info, all_gt_info[train_idx])
        
        valid_image_path = np.append(valid_image_path, all_image_path[valid_idx])
        valid_gt_info = np.append(valid_gt_info, all_gt_info[valid_idx])
        
        test_image_path = np.append(test_image_path, all_image_path[test_idx])
        test_gt_info = np.append(test_gt_info, all_gt_info[test_idx])
        
    # shuffle all data again
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
    
    
    # save to txt file
    dst = "./data/train.txt"
    write_txt(train_image_path, train_gt_info, dst)

    dst = "./data/valid.txt"
    write_txt(valid_image_path, valid_gt_info, dst)
    
    dst = "./data/test.txt"
    write_txt(test_image_path, test_gt_info, dst)
