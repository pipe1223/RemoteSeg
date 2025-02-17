import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import supervision as sv
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import json

from huggingface_hub import hf_hub_download
import supervision as sv
from pycocotools import mask as maskUtils
from pycocotools import mask
from skimage import measure

import time

import argparse

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def polygon_to_mask(polygon, image_height, image_width):
    # Create an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # Create a COCO-style RLE (Run-Length Encoding) mask from polygon
    rle = maskUtils.frPyObjects(polygon, image_height, image_width)
    
    # Decode the RLE mask to binary mask
    mask = maskUtils.decode(rle)
    
    return mask

def dice_coefficient(mask1, mask2):
    mask1 = mask1>0
    mask2 = mask2>0
    
    # Ensure masks are binary and have the same shape
    mask1 = np.asarray(mask1, dtype=np.bool_)
    mask2 = np.asarray(mask2, dtype=np.bool_)

    intersection = np.logical_and(mask1, mask2).sum()
    
    union = np.logical_or(mask1, mask2).sum()
    
    # Compute Dice coefficient
    dice = 2. * intersection / (intersection+union)  # Handle case where union is 0
    return dice



def perform_segmentation(predictor, image_path, box_list, expanded_rate=0.15, dice_cal=True, segmentation_poly="", 
                         label_list = "", visualization=True, show_box=True, show_text=True):
    #load image
    image = cv2.imread(image_path)
    image_source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image_source.shape
    
    #set image
    sam_predictor = predictor
    sam_predictor.set_image(image_source)

    #setup image for visualization
    main_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #crop roi
    roi_list = box_list
    
    dice_list = []
    roi_mask_list = []
    # Iterate over the ROI list and crop the image
    for indexing, (x, y, w, h) in enumerate((roi_list)):

        isError = False
        
        if w > h:
            expanded = int(h*expanded_rate)
        else:
            expanded = int(w*expanded_rate)  
        #expanded = int(w*0.15)
        if w > W*0.75 or h > H*0.75:
            if w > h:
                expanded = int(h*0.05)
            else:
                expanded = int(w*0.05)  

        #GT polygon to mask
        if segmentation_poly!="":
            polygon = segmentation_poly[indexing]
            binary_mask = polygon_to_mask(polygon, H, W)

        # Crop the image using numpy slicing
        cropped_image = image_source[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]
        
        if segmentation_poly!="":
            cropped_gt = binary_mask[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]

        try:
            sam_predictor.set_image(cropped_image)
        except:
            expanded = 0
            cropped_image = image_source[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]
            sam_predictor.set_image(cropped_image)
            
            if segmentation_poly!="":
                cropped_gt = binary_mask[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]

        bbox = [expanded, expanded, w+expanded, h+expanded]

        input_point = np.array([[int(w/2)+expanded, int(h/2)+expanded]])
        input_label = np.array([1])
        
        
        #sam_predictor.set_image(image_source)
        input_box = np.array(bbox)



        masks, _, _ = sam_predictor.predict(
            point_coords=None,#input_point,
            point_labels=None,#input_label,
            box=input_box[None, :],
            multimask_output=False,
        )


        roi_mask = masks[0]
        roi_mask_list = masks


        #calcualate dice
        dice = ''
        if dice_cal:
            try:
                cropped_gt = cropped_gt.squeeze(axis=-1)
                dice = dice_coefficient(roi_mask, cropped_gt)
            except:
                dice = 0
            dice_list.append(dice)

        if visualization:
            #visualize overall
            over_y = int(y)-expanded
            over_x = int(x)-expanded
            over_w = int(w) + (expanded*2)
            over_h = int(h) + (expanded*2)


            random_color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            alpha = 0.7


            for i in range(over_h-1):
                for j in range(over_w-1):
                    try:
                        if roi_mask[i, j]:
                            existing_pixel = main_image[over_y + i, over_x + j].astype(float)
                            blended_pixel = (1 - alpha) * existing_pixel + alpha * random_color
                            main_image[over_y + i, over_x + j] = np.clip(blended_pixel, 0, 255).astype(np.uint8)
                    except:
                        continue



            #GT mask visualization
            if segmentation_poly!="":
                for i in range(over_h):
                    for j in range(over_w):
                        try:
                            if binary_mask[over_y + i, over_x + j]:
                                existing_pixel = gt_image[over_y + i, over_x + j].astype(float)
                                blended_pixel = (1 - alpha) * existing_pixel + alpha * random_color
                                gt_image[over_y + i, over_x + j] = np.clip(blended_pixel, 0, 255).astype(np.uint8)
                        except Exception as e:
                            continue


            #draw bbox
            if show_box:
                cv2.rectangle(main_image, (over_x, over_y), (over_x + over_w, over_y + over_h), (255, 0, 0), 1)  # Red color, thickness=2
                #cv2.rectangle(gt_image, (over_x, over_y), (over_x + over_w, over_y + over_h), (255, 0, 0), 1)  # Red color, thickness=2

            # Add text to the bounding box
            if show_text:
                text = label_list[indexing] +":"+ str(np.around(dice, decimals=2))
                font_scale = 0.3
                font_color = (255, 255, 255)  # White color for the text
                font_thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = over_x + 5
                text_y = over_y + text_size[1] + 5

                cv2.putText(main_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

#     if visualization:
#         # create figure 
#         fig = plt.figure(figsize=(15, 20)) 

#         rows = 2
#         columns = 2


#         # Adds a subplot at the 1st position 
#         fig.add_subplot(rows, columns, 1) 
#         plt.imshow(image_source)
#         plt.title("Original") 

#         # Adds a subplot at the 2nd position 
#         fig.add_subplot(rows, columns, 2) 
#         plt.imshow(main_image)
#         plt.title("Mask")

#         # Adds a subplot at the 2nd position 
#         fig.add_subplot(rows, columns, 3) 
#         plt.imshow(gt_image)
#         plt.title("GT")

    return image_source, main_image, gt_image, dice_list, box_list, roi_mask_list


def SAM_segmentatio_json(sam_predictor, json_path, dataset_path):
    dataset_path = dataset_path#'/home/datadisk/evaluation/data/Benchmark/datasets/'
    # Load the JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    data_list = json_data['data']
    dice_lists = []
    counter = 0
    for data in data_list:
        image_path = data['image'][0]
        print (f'index[{counter}]: {image_path}')
        counter = counter+1
        annotations = data['annotations']
        
        box_list = []
        label_list = []
        segmentation_poly = []
        for annotation in annotations:
            box_list.append(annotation['bbox'])
            label_list.append(annotation['label'])
            segmentation_poly.append(annotation['segmentation'])

        image_path = data['image'][0]
        local_image_path = dataset_path+image_path
        
        image_source, main_image, gt_image, dice_list, box_list, roi_mask_list = perform_segmentation(sam_predictor, local_image_path, 
                                                                                                      box_list, dice_cal=True, 
                                                                                                      segmentation_poly=segmentation_poly,
                                                                                                      label_list = label_list, 
                                                                                                      visualization=False, show_box=True, 
                                                                                                      show_text=False)
        
        #print ("DICE: ",np.sum(dice_list)/len(dice_list))
        dice_lists = dice_lists + dice_list
    return dice_lists

def sementation(args):
    
    #load sam
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sam_checkpoint = '/home/datadisk/evaluation/models/SAM/sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    
    #select type of input
    json_path = args.json_path
    dataset_path = args.data_path
    #image
    if json_path == '':
        image_path = args.image_path
        bounding_box = args.bounding_box
        image_source, main_image, gt_image, dice_list, box_list, roi_mask_list = perform_segmentation(sam_predictor, image_path, 
                                                                                                      [box_list], dice_cal=True, 
                                                                                                      segmentation_poly='',
                                                                                                      label_list = "", 
                                                                                                      visualization=False, show_box=True, 
                                                                                                      show_text=False)
        
    #json
    else:
        ice_lists = SAM_segmentatio_json(sam_predictor, json_path, dataset_path)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default="")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--bounding-box", type=str, default="")
    parser.add_argument("--data-path", type=str, default="/home/datadisk/evaluation/data/Benchmark/datasets/")
    
    args = parser.parse_args()
    
    sementation(args)
    
    