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


def perfrom_postprocess(df_path = "/datadisk/xyy/codes/Evaluation/post_output.csv"):
    df = pd.read_csv(df_path)#'/home/datadisk/pipe/results/sam_box/output.csv')
    
    df['area'] = df['roi'].apply(
        lambda x: (
            (coords := list(map(int, x.strip('[]').split(','))))  # Extract and convert to int
            and ((coords[2] - coords[0]) * (coords[3] - coords[1]))  # Compute area
    ))

    df['post_dice'] = np.nan
    df['post_path'] = np.nan
    
    #index = 5
    for index, row in df.iterrows():
        
        mask = cv2.imread(row['path'].replace("toy_set","toy_set2"))
        post_mask = post_process.post_process(mask,row['area'],row['class'])
        
        # calculate dice
        #save post_mask result
        row['post_dice'] = 0
        row['post_path'] = 'aaaa'

    #save df