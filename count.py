import os
import numpy as np
from glob import glob
from tqdm import tqdm
from xml.etree import ElementTree as ET

def load_xml(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def extract_time_of_day(xml_content):
    tree = ET.fromstring(xml_content)
    for elem in tree.iter('timeofday'):
        return elem.text.strip()  # 수정: text에서 공백 제거
    return None  # Return None if <timeofday> tag is not found

def count_time_of_day_in_folder(folder_path):
    xml_paths = glob(os.path.join(folder_path, "*.xml"))
    time_of_day_counts = {'daytime': 0, 'dawn/dusk': 0, 'night': 0}
    
    for xml_path in tqdm(xml_paths, desc=f"Processing {os.path.basename(folder_path)}"):
        xml_content = load_xml(xml_path)
        time_of_day = extract_time_of_day(xml_content)
        
        if time_of_day and time_of_day in time_of_day_counts:
            time_of_day_counts[time_of_day] += 1
    
    return time_of_day_counts

if __name__ == "__main__":
    base_path = "/home/ailab/AILabDataset/01_Open_Dataset/37_WeatherDataset/DWD"
    target_folders = ["daytime_clear", "Night-Sunny", "dusk_rainy", "night_rainy"]

    overall_counts = {'daytime': 0, 'dawn/dusk': 0, 'night': 0}
    for folder in target_folders:
        folder_path = os.path.join(base_path, folder, "VOC2007", "Annotations")
        counts = count_time_of_day_in_folder(folder_path)
        for key, value in counts.items():
            overall_counts[key] += value
        print(f"Counts for {folder}: {counts}")
    
    print(f"Overall counts across all folders: {overall_counts}")
