###PPPPP

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
import matplotlib.pyplot as plt
import PIL
import requests
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import json
from huggingface_hub import hf_hub_download
from pycocotools import mask as maskUtils
from skimage import measure
import time
import rasterio
import base64
import pickle
import shapely
from rasterio.features import geometry_mask

# Function to display the segmentation mask overlay on an axis
def show_mask(mask, ax, random_color=False):
    if random_color:
        # Generate a random color with alpha transparency
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Default color (light blue) with alpha transparency
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image

# Function to display key points on the image
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]  # Positive points
    neg_points = coords[labels == 0]  # Negative points
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# Function to display a bounding box on the image
def show_box(box, ax):
    x0, y0 = box[0], box[1]  # Top-left corner
    w, h = box[2] - box[0], box[3] - box[1]  # Width and height
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def parse_coord(coord):
    # Remove any trailing parentheses or unwanted characters
    coord = coord.strip(")")
    coord = coord.strip("(")
    # Split into x and y components and convert to float
    return tuple(map(float, coord.split()))
    
# segmentation multipolygon to box
def multipolygon_to_box(obj):
                         
    coords_str = str(obj).replace("MULTIPOLYGON (((", "").replace(")))", "")
    # Split into individual coordinate pairs
    coords_list = coords_str.split(", ")

    # Parse coordinates into a list of (x, y) tuples
    #coords = [tuple(map(int, coord.split())) for coord in coords_list]
    #coords = [tuple(map(float, coord.split())) for coord in coords_list]
    coords = [parse_coord(coord) for coord in coords_list]
    
    # Find min and max x and y values
    min_x = min(coord[0] for coord in coords)
    max_x = max(coord[0] for coord in coords)
    min_y = min(coord[1] for coord in coords)
    max_y = max(coord[1] for coord in coords)

    # Define the bounding box
    bounding_box = [min_x,min_y,max_x,max_y]
    return bounding_box

# Converts a polygon to a binary mask
def polygon_to_mask(polygon, image_height, image_width):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    rle = maskUtils.frPyObjects(polygon, image_height, image_width)  # Generate RLE mask
    mask = maskUtils.decode(rle)  # Decode RLE to binary mask
    return mask

def poly_sting_to_multi_polygon(poly_string):
    pickled_bytes = base64.b64decode(poly_string.encode("utf-8"))
    obj = pickle.loads(pickled_bytes)
    return obj

def multi_polygon_to_mask(multi_polygon,height,width):
    height, width = height, width  
    transform = rasterio.transform.from_origin(0, height, 1, 1)  

    mask = geometry_mask(
        [multi_polygon], 
        transform=transform, 
        out_shape=(height, width), 
        invert=True  
    )
    return mask

def str_poly_to_mask(polygon, image_height, image_width):
    multi_poly = poly_sting_to_multi_polygon(polygon)
    mask = multi_polygon_to_mask(multi_poly,image_height, image_width)
    mask = np.flip(mask, axis=0) ##dont know why the result from rasterio is not correct so need to flip
    return mask
    

# Calculates the Dice coefficient between two masks (a measure of similarity)
def dice_coefficient(mask1, mask2):
    mask1 = mask1 > 0  # Convert to binary
    mask2 = mask2 > 0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    dice = 2. * intersection / (intersection + union)  # Avoid division by zero
    return dice

# Perform segmentation on the image based on ROIs and optional ground truth masks
def perform_segmentation(predictor, image_path, box_list, expanded_rate=0.15, dice_cal=True, segmentation_poly="", 
                         label_list = "", visualization=True, show_box=True, show_text=False, crop =[]):
    #load image
    #print(image_path)
    image = cv2.imread(image_path)
    
    if crop!=[]:
        image = image[crop[1]:crop[3], crop[0]:crop[2]]
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
    overlay_box_list = []
    # Iterate over the ROI list and crop the image
    for indexing, (x, y, w, h) in enumerate((roi_list)):
        try:
            x, y, w, h = float(x), float(y), float(w), float(h)
            isError = False
    
            if w > h:
                expanded = int(h*expanded_rate)
            else:
                expanded = int(w*expanded_rate)  
            #expanded = int(w*0.15)
            if w > W*0.75 or h > H*0.75:
                if w > h:
                    expanded = int(h*0.025)
                else:
                    expanded = int(w*0.025)  
    
            #GT polygon to mask
              
                
            if segmentation_poly!="":
                polygon = segmentation_poly[indexing]
                if isinstance(polygon, list):
#                     print ('list poly')
                    binary_mask = polygon_to_mask(polygon, H, W)
                    
                elif type(polygon) == str:
#                     print ('str poly')
                    binary_mask = str_poly_to_mask(polygon, H, W)

                else:
                    print('print error nothing')
            else:
                print('print error segmentation_poly')
            
#             print (binary_mask)
    
            # Crop the image using numpy slicing
            cropped_image = image_source[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]
            
#             print (binary_mask.shape)
#             print (image_source.shape)
            
#             plt.imshow(binary_mask)
#             plt.show()
            if segmentation_poly!="":
                cropped_gt = binary_mask[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]
#                 if isinstance(polygon, list):
#                     cropped_gt = binary_mask[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]
#                 elif type(polygon) == str:
#                     cropped_gt = binary_mask[int(x)-expanded:int(x+w)+expanded, int(y)-expanded:int(y+h)+expanded]
    
            try:
                sam_predictor.set_image(cropped_image)
            except:
                expanded = 0
                cropped_image = image_source[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]
                sam_predictor.set_image(cropped_image)
                
                if segmentation_poly!="":
                    cropped_gt = binary_mask[int(y)-expanded:int(y+h)+expanded, int(x)-expanded:int(x+w)+expanded]

#             plt.imshow(cropped_image)
#             plt.show()
#             plt.imshow(cropped_gt)
#             plt.show()
            bbox = [expanded, expanded, w+expanded, h+expanded]
    
            input_point = np.array([[int(w/2)+expanded, int(h/2)+expanded]])
            input_label = np.array([1])
            
            
            #sam_predictor.set_image(image_source)
            input_box = np.array(bbox)
    
    
    
            masks, _, _ = sam_predictor.predict(
                point_coords=None,#input_point,
                point_labels=label_list[indexing],#None,#input_label,
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
                    try:
                        dice = dice_coefficient(roi_mask, cropped_gt)
                    except:
                        #make an error
                        dice = dice_coefficient(roi_mask, cropped_gt)
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
                                #main image
                                existing_pixel = main_image[over_y + i, over_x + j].astype(float)
                                blended_pixel = (1 - alpha) * existing_pixel + alpha * random_color
                                main_image[over_y + i, over_x + j] = np.clip(blended_pixel, 0, 255).astype(np.uint8)
                                
                                #overlab coop
                                existing_pixel = cropped_image[i, j].astype(float)
                                blended_pixel = (1 - alpha) * existing_pixel + alpha * random_color
                                cropped_image[i, j] = np.clip(blended_pixel, 0, 255).astype(np.uint8)
                                
                        except:
                            continue
                
#                 plt.imshow(cropped_image)
#                 plt.show()
                overlay_box_list.append(cropped_image)
                
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
                    font_scale = 1#0.3
                    font_color = (255, 255, 255)  # White color for the text
                    font_thickness = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    text_x = over_x + 5
                    text_y = over_y + text_size[1] + 5
    
                    cv2.putText(main_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

        except Exception as e:
            print(f"Error: {e}")
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

    return image_source, main_image, gt_image, dice_list, box_list, roi_mask_list, overlay_box_list

# Process multiple images and annotations from a JSON file
def SAM_segmentatio_json(sam_predictor, data_list, dataset_path):
    # Load JSON data
#     with open(json_path, 'r') as f:
#         json_data = json.load(f)

    data_list = data_list#json_data['data']
    image_sources = []
    main_images = []
    gt_images = []
    dice_lists = []
    box_lists = []
    local_image_paths = []
    roi_mask_lists = []
    for data in data_list:
        image_path = data['image'][0]
        annotations = data['object_annotations']
        box_list = [anno['bbox'] for anno in annotations]
        label_list = [anno['label'] for anno in annotations]
        segmentation_poly = [anno['segmentation'] for anno in annotations]
        local_image_path = os.path.join(dataset_path, image_path)
        local_image_paths.append(local_image_path)
        xywh_boxes = []
#         print ("box list: ", box_list)
        if box_list[0] != []:
            for box in box_list:
                x1, y1, x2, y2 =  box  # Convert strings to integers
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                xywh_boxes.append([x, y, w, h])

            #print ('xywh_boxes',xywh_boxes)
            # Perform segmentation
            
            image_source, main_image, gt_image, dice_list, box_list, roi_mask_list = perform_segmentation(sam_predictor, local_image_path, xywh_boxes, dice_cal=False, segmentation_poly="", label_list="")

            image_sources.append(image_source)
            main_images.append(main_image)
            gt_images.append(gt_image)
            dice_lists.append(dice_list)
            box_lists.append(box_list)
            roi_mask_lists.append(roi_mask_list)
        else:
            image = cv2.imread(local_image_path)
            image_source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_sources.append(image_source)
            main_images.append(image_source)
        
    return image_sources, main_images, gt_images, dice_lists, box_lists, roi_mask_lists, local_image_paths


def open_image_with_nothing(file_path):
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']
    data_paths = ['/goss/Datasets/', '/home/oss/Datasets/']
    final_path = ''
    
    for ext in extensions:
        for data_path in data_paths:
            if file_path.endswith(ext):
                local_image_path = data_path+ file_path
            else:
                local_image_path = data_path+ file_path + ext
            final_path = local_image_path
            if os.path.exists(local_image_path):
                return local_image_path
            

#working on this ne
def SAM_segmentatio_json_detect(sam_predictor, data_list, dataset_path):
    # Load JSON data
#     with open(json_path, 'r') as f:
#         json_data = json.load(f)

    data_list = data_list#json_data['data']
    image_sources = []
    main_images = []
    gt_images = []
    dice_lists = []
    box_lists = []
    local_image_paths = []
    roi_mask_lists = []
    label_list = []
    data = data_list
# for data in data_list:
    image_path = data['image'][0].replace("@","/")
    image_size = data['size']
    box_list = []
    segmentation_poly = []
    object_annotations = data['object_annotations']
    if object_annotations != []:
        for object_annotation in object_annotations:
            label_list.append(object_annotation['label'])
            segmentation_poly.append(object_annotation['segmentation'])

            
            if object_annotation['bbox'] == "":
                box_list.append(multipolygon_to_box(poly_sting_to_multi_polygon(object_annotation['segmentation'])))
            else:
                box_list.append(object_annotation['bbox'])
            
            
            
            
#             for seg_obj, box_obj in zip(segmentation_poly, box_list):
#                 if box_obj =="":
#                     roi_list.append(multipolygon_to_box(poly_sting_to_multi_polygon(seg_obj)))
#                 else:
#                     roi_list.append(box_obj)
            

        #label_list = [data['score']]

        local_image_path =  open_image_with_nothing(image_path)



        local_image_paths.append(local_image_path)

        xywh_boxes = []
        if box_list[0] != []:
            for box in box_list:
                x1, y1, x2, y2 =  box  # Convert strings to integers
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                xywh_boxes.append([x, y, w, h])

            #print ('xywh_boxes',xywh_boxes)
            # Perform segmentation

            image_source, main_image, gt_image, dice_list, box_list, roi_mask_list, overlay_box_list = perform_segmentation(sam_predictor, local_image_path, xywh_boxes, dice_cal=True, segmentation_poly=segmentation_poly, label_list=label_list, show_text=True)

            image_sources.append(image_source)
            main_images.append(main_image)
            gt_images.append(gt_image)
            dice_lists.append(dice_list)
            box_lists.append(box_list)
            roi_mask_lists.append(roi_mask_list)
        else:
            image = cv2.imread(local_image_path)
            image_source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_sources.append(image_source)
            main_images.append(image_source)
        
    return image_sources, main_images, gt_images, dice_lists, box_lists, roi_mask_lists, local_image_paths, label_list, overlay_box_list

#hbb_detection_toyset
def SAM_segmentatio_json_detect_Backup(sam_predictor, data_list, dataset_path):
    # Load JSON data
#     with open(json_path, 'r') as f:
#         json_data = json.load(f)

    data_list = data_list#json_data['data']
    image_sources = []
    main_images = []
    gt_images = []
    dice_lists = []
    box_lists = []
    local_image_paths = []
    roi_mask_lists = []
    label_list = []
    for data in data_list:
        print ("DATA:", data)
        image_path = data['image'][0].replace("@","/")
        try:
            crop=data['crop']
        except:
            crop =[]
        box_list = []
        object_annotations = data['object_annotations']
        if object_annotations != []:
            for object_annotation in object_annotations:
                box_list.append(object_annotation['bbox'])
                label_list.append(object_annotation['label'])
            #label_list = [data['score']]

            local_image_path =  open_image_with_nothing(image_path)



            local_image_paths.append(local_image_path)

            xywh_boxes = []
            if box_list[0] != []:
                for box in box_list:
                    x1, y1, x2, y2 =  box  # Convert strings to integers
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    xywh_boxes.append([x, y, w, h])

                #print ('xywh_boxes',xywh_boxes)
                # Perform segmentation

                image_source, main_image, gt_image, dice_list, box_list, roi_mask_list = perform_segmentation(sam_predictor, local_image_path, xywh_boxes, dice_cal=False, segmentation_poly="", label_list=label_list,crop=crop)

                image_sources.append(image_source)
                main_images.append(main_image)
                gt_images.append(gt_image)
                dice_lists.append(dice_list)
                box_lists.append(box_list)
                roi_mask_lists.append(roi_mask_list)
            else:
                image = cv2.imread(local_image_path)
                image_source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_sources.append(image_source)
                main_images.append(image_source)
        
    return image_sources, main_images, gt_images, dice_lists, box_lists, roi_mask_lists, local_image_paths


def SAM_segmentation_with_json(sam_predictor, json_path):
    file_json = open(json_path)
    json_obj = json.load(file_json)

    # Extract basic information: task, model, and dataset
    try:
        info_json = json_obj['info']
        result_task = info_json['task']
        result_model = info_json['model']
        result_dataset = info_json['dataset']
    except:
        # Fallback if the structure does not include 'info'
        result_task = json_obj['task']
        result_model = json_obj['model']
        result_dataset = json_obj['dataset']

    # Initialize variables for processing the data
    data_json = json_obj['data']

    data_index = 0
    image_file = data_json[data_index]["image"][0]

    try:
        local_image_path = os.path.join("/home/oss/Datasets/", image_file)
        image = Image.open(local_image_path)
    except:
        new_root_path = '/home/datadisk2/Datasets2/processed_datasets/'
        try:
            local_image_path = os.path.join(new_root_path, image_file)
            image = Image.open(local_image_path)
        except:
            local_image_path = os.path.join("/home/oss/Datasets/", image_file)
            image = Image.open(local_image_path)

    bounding_boxes = data_json[data_index]["answer"]

    xywh_boxes = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert strings to integers
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        xywh_boxes.append([x, y, w, h])
    
    image_source, main_image, gt_image, dice_list, box_list, roi_mask_list = perform_segmentation(sam_predictor, local_image_path, xywh_boxes, dice_cal=False, segmentation_poly="", label_list="")
    return image_source, main_image, gt_image, dice_list, box_list, roi_mask_list


def sementation_json(json_path, sam_checkpoint='/home/datadisk/evaluation/models/SAM/sam_vit_h_4b8939.pth'):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam_checkpoint = sam_checkpoint
    sam = build_sam(checkpoint=sam_checkpoint).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    image_source, main_image, gt_image, dice_list, box_list, roi_mask_list = SAM_segmentation_with_json(sam_predictor, json_path)

    return image_source, main_image, gt_image, dice_list, box_list, roi_mask_list


# Main function to execute segmentation based on user inputs
def sementation_call(data_list, data_path, sam_checkpoint='/home/datadisk/evaluation/models/SAM/sam_vit_h_4b8939.pth'):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam_checkpoint = sam_checkpoint
    sam = build_sam(checkpoint=sam_checkpoint).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    return SAM_segmentatio_json_detect(sam_predictor, data_list, data_path)


# Main function to execute segmentation based on user inputs
def sementation(args):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam_checkpoint = '/path/to/sam_checkpoint.pth'
    sam = build_sam(checkpoint=sam_checkpoint).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    if args.json_path:
        dice_lists = SAM_segmentatio_json(sam_predictor, args.json_path, args.data_path)
    else:
        image_path = args.image_path
        perform_segmentation(sam_predictor, image_path, args.bounding_box)

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--json-path", type=str, default="")
#    parser.add_argument("--image-path", type=str, default="")
#    parser.add_argument("--bounding-box", type=str, default="")
#    parser.add_argument("--data-path", type=str, default="/path/to/data")
#    args = parser.parse_args()
#    sementation(args)
