"""
--------------------------------------------------
Author:                           Louis(111550132)
--------------------------------------------------
"""

import pandas as pd
import numpy as np
import cv2
import os
import random

def read_image(gray_img = False, coordinate = False):
    dataset = []
    folder_paths = ['../data/1800~1970']
    #folder_paths = ['../data/~1500', '../data/1400~1800', '../data/1800~1970', '../data/1970~']
    for folder_path in folder_paths:
        floder_name = folder_path.split('/')[-1]
        df = pd.read_csv(f'../data/csv_data/{floder_name}.csv')
        i = 0
        for filename in df['picture']:
            building_info = df.iloc[i]
            i = i + 1
            filepath = os.path.join(folder_path, filename)
            if gray_img:
                image_array = np.array(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE))
            else:
                image_array = np.array(cv2.imread(filepath))
            if coordinate:
                dataset.append((image_array, building_info['year'], building_info['longitude'], building_info['latitude']))
            else:
                dataset.append((image_array, building_info['year']))
            print(building_info['year'], building_info['longitude'], building_info['latitude'])
    # train test split
    SPLIT_RATIO = 0.7
    random.shuffle(dataset)
    train_dataset = dataset[:int(SPLIT_RATIO * len(dataset))]
    test_dataset = dataset[int(SPLIT_RATIO * len(dataset)):]

    return train_dataset, test_dataset

if __name__ == "__main__":
    read_image()