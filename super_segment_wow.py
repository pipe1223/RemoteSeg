import sys
sys.path.append('/home/kt/python/data_engineer/sam_road')
sys.path.append('/home/kt/python/data_engineer/road_extract')
print (sys.path)
import json
import argparse
import sam_road.SAM_ROAD as sam_road
import sam_box.SAM_box as sam

import road_extract.road_extraction as road_ex

import matplotlib.pyplot as plt
import cv2

from skimage import measure
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
#data clearning

def resize_to_square_and_pad(image_source, scale):
    """
    Resize the image to a square with dimensions divisible by 256,
    then resize it by dividing dimensions by the given scale and add padding.

    Parameters:
        image_source (numpy.ndarray): The input image.
        scale (int): The scaling divisor (e.g., 1, 2, 4, 8, ...).

    Returns:
        numpy.ndarray: The resized, squared, and padded image.
    """
    # Original dimensions
    original_height, original_width = image_source.shape[:2]
    
    # Calculate the next multiple of 256 for the larger dimension
    max_dim = max(original_height, original_width)
    square_dim = ((max_dim + 255) // 256) * 256  # Round up to the nearest multiple of 256

    # Resize the image to square dimensions
    resized_to_square = cv2.resize(image_source, (square_dim, square_dim), interpolation=cv2.INTER_AREA)
    
    # Compute scaled dimensions
    new_dim = max(1, square_dim // scale)

    # Resize the square image by the given scale
    resized_image = cv2.resize(resized_to_square, (new_dim, new_dim), interpolation=cv2.INTER_AREA)

    # Calculate padding to restore the image to the original square dimensions
    delta_width = square_dim - new_dim
    delta_height = square_dim - new_dim

    top = delta_height // 2
    bottom = delta_height - top
    left = delta_width // 2
    right = delta_width - left

    # Add padding
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def remove_padding_and_resize(image_padded, original_size):
    """
    Remove padding from an image and resize it back to its original size.

    Parameters:
        image_padded (numpy.ndarray): The padded image.
        original_size (tuple): The original size of the image (height, width).

    Returns:
        numpy.ndarray: The image resized back to its original size.
    """
    # Identify the non-zero region (content without padding)
    gray_image = cv2.cvtColor(image_padded, cv2.COLOR_BGR2GRAY) if len(image_padded.shape) == 3 else image_padded
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(binary_mask)

    # Crop the region of interest (ROI) without padding
    cropped_image = image_padded[y:y+h, x:x+w]

    # Resize back to the original size
    resized_to_original = cv2.resize(cropped_image, (original_size[1], original_size[0]), interpolation=cv2.INTER_AREA)

    return resized_to_original

def combine_multiscale_results(scale_results, original_size, method='average'):
    """
    Combine multi-scale results into a single result.

    Parameters:
        scale_results (list): A list of results (masks or probability maps) at different scales.
        original_size (tuple): The original size of the image (height, width).
        method (str): Combination method - 'average' or 'max'.

    Returns:
        numpy.ndarray: The combined result resized to the original image size.
    """
    combined_result = np.zeros(original_size, dtype=np.float32)

    for result in scale_results:
        # Resize each result back to the original size
        resized_result = cv2.resize(result, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Combine the results based on the selected method
        if method == 'average':
            combined_result += resized_result
        elif method == 'max':
            combined_result = np.maximum(combined_result, resized_result)
        else:
            raise ValueError("Invalid method. Use 'average' or 'max'.")

    # For averaging, divide by the number of scales
    if method == 'average':
        combined_result /= len(scale_results)

    # Normalize the result if necessary
    combined_result = np.clip(combined_result, 0, 255).astype(np.uint8)
    return combined_result

def combine_multiscale_results(scale_results, scales, original_size, method='average'):
    """
    Combine multi-scale results into a single result while ensuring alignment.

    Parameters:
        scale_results (list): A list of results (masks or probability maps) at different scales.
        scales (list): A list of scales corresponding to each result.
        original_size (tuple): The original size of the image (height, width).
        method (str): Combination method - 'average' or 'max'.

    Returns:
        numpy.ndarray: The combined result resized to the original image size.
    """
    combined_result = np.zeros(original_size, dtype=np.float32)

    for idx, (result, scale) in enumerate(zip(scale_results, scales)):
        print(f"Processing scale {scale} - result shape: {result.shape}")

        # Step 1: Reverse padding to restore to square resolution (if padding was added)
        target_size = (int(original_size[1] * scale), int(original_size[0] * scale))
        if result.shape[:2] != target_size:
            print(f"Resizing result {idx} to {target_size}")
            result = cv2.resize(result, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Step 2: Resize result back to the original image size
        resized_result = cv2.resize(result, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

        # Step 3: Combine results
        if method == 'average':
            combined_result += resized_result
        elif method == 'max':
            combined_result = np.maximum(combined_result, resized_result)
        else:
            raise ValueError("Invalid method. Use 'average' or 'max'.")

    # Normalize results
    if method == 'average':
        combined_result /= len(scale_results)

    # Clip the results to ensure valid range (e.g., for masks)
    combined_result = np.clip(combined_result, 0, 255).astype(np.uint8)
    return combined_result

def generate_segmentation_MAID():
    m_aid_dir = ""
    return

def read_detection_json(args):
    file_json = open(args.json_path)
    json_obj = json.load(file_json)
    data_json = json_obj#[0]
    return data_json

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
                # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new("1", (width + 2, height + 2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    number = 0
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # if number == 94:
        #     test = 1
        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        if (poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        if isinstance(poly, MultiPolygon):
            # Handle multiple Polygons
            # print("Result is a MultiPolygon:")
            for i, sub_poly in enumerate(poly.geoms):
                segmentation = np.array(sub_poly.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)
        elif isinstance(poly, Polygon):
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        # number = number + 1
        # print(poly)
        # print(number)

    return polygons, segmentations


def generate_segmentation(args):
    index = 30
    data_json = read_detection_json(args)[index:index+1]
    #['data'][520:570]
    # Extract basic information: task, model, and dataset
#     try:
#         info_json = json_obj['info']
#         result_task = info_json['task']
#         result_model = info_json['model']
#         result_dataset = info_json['dataset']
#     except:
#         # Fallback if the structure does not include 'info'
#         result_task = json_obj['task']
#         result_model = json_obj['model']
#         result_dataset = json_obj['dataset']

#     # Initialize variables for processing the data
#     data_json = json_obj['data']

#     data_index = 0
#     raw_image = data_json[data_index]["image"][0]
#     bounding_boxes = data_json[data_index]["answer"]

    
#     print (bounding_boxes)
    image_sources, main_images, gt_images, dice_lists, box_lists, roi_mask_lists, local_image_paths = sam.sementation_call(data_json, args.data_path)
    
    config_dir = "/home/kt/python/sam_road/config/"
    sam_road_checkpoint = "/home/datadisk/evaluation/models/SAM/SAM-road/"
    net_cityscale, config_cityscale = sam_road.load_sam_road(f"{config_dir}toponet_vitb_512_cityscale.yaml", f"{sam_road_checkpoint}cityscale_vitb_512_e10.ckpt", "cuda")
    net_spacenet, config_spacenet = sam_road.load_sam_road(f"{config_dir}toponet_vitb_256_spacenet.yaml", f"{sam_road_checkpoint}spacenet_vitb_256_e10.ckpt", "cuda")
    
    
    print(roi_mask_lists)
    for image_source, main_image, local_image_path in zip(image_sources,main_images, local_image_paths):
        
        
        original_height, original_width = image_source.shape[:2]
        if original_height <256 or original_width <256:
            new_height = 256
            new_width = 256

            # Resize the image
            resized_image = cv2.resize(image_source, (new_width, new_height), interpolation=cv2.INTER_AREA)
            image_source = resized_image
        print ("image shape: ",image_source.shape)
        
        
        
        plt.imshow(main_image)
        plt.show()
        #road #1
        multi_scales = [1,2,3,4]
        scale_results = []
        scale_results_small = []
        for scale in multi_scales:
            print ("scale:", scale)
            padded_image = resize_to_square_and_pad(image_source, scale)
            plt.imshow(padded_image)
            plt.show()
            pred_nodes_spacenet, pred_edges_spacenet, itsc_mask_spacenet, road_mask_spacenet = sam_road.infer_one_img(net_spacenet, padded_image, config_spacenet)
            
            
            overlay_image = sam_road.overlay_mask_on_image(padded_image, road_mask_spacenet, color=(0, 255, 0), alpha=0.2,th=15)
            
            final_over = remove_padding_and_resize(overlay_image, image_source.shape)
            final_scale_result = remove_padding_and_resize(road_mask_spacenet, image_source.shape)
            scale_results.append(final_scale_result)
            
#             plt.imshow(final_over)
#             plt.show()
            
            padded_image
            img = cv2.resize(image_source,(1024,1024))
            padded_image = resize_to_square_and_pad(img, scale)
            mask, blended_image = road_ex.road_extraction_image(padded_image)
            
            final_scale_result = remove_padding_and_resize(blended_image, img.shape)
            scale_results_small.append(final_scale_result)
#             plt.imshow(final_scale_result)
#             plt.show()
        
        #combined_mask = combine_multiscale_results(scale_results, main_image.shape[:2], method='average')
        combined_mask = combine_multiscale_results(scale_results,multi_scales, main_image.shape[:2], method='average')
        combined_mask_small = combine_multiscale_results(scale_results_small,multi_scales, (1024,1024,3), method='average')
        overlay_image = sam_road.overlay_mask_on_image(main_image, combined_mask, color=(0, 255, 255), alpha=0.2,th=15)
            
        
        threshold = 15
        
        binary_mask = (combined_mask > threshold).astype(int)
        
#         sub_mask = create_sub_masks(binary_mask,binary_mask.shape[1],binary_mask.shape[0])
#         print (sub_mask)
        polygons, segmentations = create_sub_mask_annotation(binary_mask)

    
        
        #road #2
        mask, blended_image = road_ex.road_extraction(local_image_path)

        #do something with segmentations (save)
        print (segmentations)
    