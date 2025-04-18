import os
import sys
import cv2
import json
import pickle
import base64
import shapely
import argparse
import rasterio
import numpy as np
import pandas as pd
from skimage import measure
from datetime import datetime
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, MultiPolygon

import post_process
#data clearning

# Calculates the Dice coefficient between two masks (a measure of similarity)
def dice_coefficient(mask1, mask2):
    mask1 = mask1 > 0  # Convert to binary
    mask2 = mask2 > 0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    dice = 2. * intersection / (intersection + union)  # Avoid division by zero
    return dice

def perfrom_postprocess(df_path):
    df = pd.read_csv(df_path)#'/home/datadisk/pipe/results/sam_box/output.csv')
    
    df['area'] = df['roi'].apply(
        lambda x: (
            (coords := list(map(int, x.strip('[]').split(','))))  # Extract and convert to int
            and ((coords[2] - coords[0]) * (coords[3] - coords[1]))  # Compute area
    ))

    dataset_name = 'toyset'
    file_testing_name = 'post'
    csv_path = "/home/datadisk/pipe/results/sam_box/post_"+datetime.now().strftime('%Y%m%d%H%M%S%f')+"/"+dataset_name+"/"
    post_box_dir = csv_path+file_testing_name+"/"
    if not os.path.exists(post_box_dir):
        # Create the directory
        os.makedirs(post_box_dir)
        
    df['post_dice'] = np.nan
    df['post_path'] = np.nan
    
    #index = 5
    for index, row in df.iterrows():
        gtp = row['gt_path'].replace('/home','')
        print (gtp)
        gt_mask = cv2.imread(gtp)
        mask = cv2.imread(row['path'])
        post_mask = post_process.post_process(mask,row['area'],row['class'])
        
        post_final_path = post_box_dir+datetime.now().strftime('%Y%m%d%H%M%S%f')+".jpg"
        
        # calculate dice
        #save post_mask result
        
        df.iloc[index, df.columns.get_loc('post_dice')] = dice_coefficient(gt_mask, post_mask)
        df.iloc[index, df.columns.get_loc('post_path')] = post_final_path


        cv2.imwrite(post_final_path, post_mask)

    df.to_csv(csv_path+'output.csv', index=False)
    return csv_path+'output.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/datadisk/pipe/results/sam_box/20250402114250408401/toy_set/Annotations_val/output.csv")


    args = parser.parse_args()

    # Use the parsed arguments

    print (perfrom_postprocess(args.csv_path))

    

