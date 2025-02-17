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

# Converts a polygon to a binary mask
def polygon_to_mask(polygon, image_height, image_width):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    rle = maskUtils.frPyObjects(polygon, image_height, image_width)  # Generate RLE mask
    mask = maskUtils.decode(rle)  # Decode RLE to binary mask
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
                         label_list="", visualization=True, show_box=True, show_text=True):
    # Load the input image
    image = cv2.imread(image_path)
    image_source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image_source.shape

    # Set up the segmentation predictor
    sam_predictor = predictor
    sam_predictor.set_image(image_source)

    # Prepare for visualization
    main_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi_list = box_list
    dice_list = []
    roi_mask_list = []

    # Iterate through each ROI
    for indexing, (x, y, w, h) in enumerate((roi_list)):
        isError = False
        # Adjust ROI expansion rate
        expanded = int(min(w, h) * expanded_rate)
        if w > W * 0.75 or h > H * 0.75:
            expanded = int(min(w, h) * 0.05)

        # Generate ground truth mask if polygons are provided
        if segmentation_poly != "":
            polygon = segmentation_poly[indexing]
            binary_mask = polygon_to_mask(polygon, H, W)

        # Crop the ROI region from the image
        cropped_image = image_source[int(y) - expanded:int(y + h) + expanded, int(x) - expanded:int(x + w) + expanded]
        if segmentation_poly != "":
            cropped_gt = binary_mask[int(y) - expanded:int(y + h) + expanded, int(x) - expanded:int(x + w) + expanded]

        # Try setting the cropped image to the predictor
        try:
            sam_predictor.set_image(cropped_image)
        except:
            expanded = 0  # Fallback without expansion
            cropped_image = image_source[int(y) - expanded:int(y + h) + expanded, int(x) - expanded:int(x + w) + expanded]
            sam_predictor.set_image(cropped_image)
            if segmentation_poly != "":
                cropped_gt = binary_mask[int(y) - expanded:int(y + h) + expanded, int(x) - expanded:int(x + w) + expanded]

        # Define the bounding box and input point for prediction
        bbox = [expanded, expanded, w + expanded, h + expanded]
        input_box = np.array(bbox)

        # Perform segmentation prediction
        masks, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        roi_mask = masks[0]
        roi_mask_list.append(roi_mask)

        # Calculate Dice coefficient if requested
        if dice_cal and segmentation_poly != "":
            dice = dice_coefficient(roi_mask, cropped_gt)
            dice_list.append(dice)

        # Visualization overlay (optional)
        if visualization:
            # Code for overlaying the ROI mask and GT mask onto the image
            pass  # (Omitted here for brevity)

    return image_source, main_image, gt_image, dice_list, box_list, roi_mask_list

# Process multiple images and annotations from a JSON file
def SAM_segmentatio_json(sam_predictor, json_path, dataset_path):
    # Load JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    data_list = json_data['data']
    dice_lists = []
    for data in data_list:
        image_path = data['image'][0]
        annotations = data['annotations']
        box_list = [anno['bbox'] for anno in annotations]
        label_list = [anno['label'] for anno in annotations]
        segmentation_poly = [anno['segmentation'] for anno in annotations]

        # Perform segmentation
        local_image_path = os.path.join(dataset_path, image_path)
        _, _, _, dice_list, _, _ = perform_segmentation(sam_predictor, local_image_path, box_list, dice_cal=True,
                                                        segmentation_poly=segmentation_poly, label_list=label_list)
        dice_lists += dice_list
    return dice_lists

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default="")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--bounding-box", type=str, default="")
    parser.add_argument("--data-path", type=str, default="/path/to/data")
    args = parser.parse_args()
    sementation(args)
