import sys
sys.path.append('sam_road')
sys.path.append('road_extract')
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
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime
import os
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

def read_detection_json(json_path):
    file_json = open(json_path)
    json_obj = json.load(file_json)
    data_json = json_obj['data']
    
    
    ##check and handle json format 
    try:
        data_json[0]['object_annotations'][0]
        return data_json
    except:
        ##neeed handle format of double array
        flattened_data = [item for sublist in data_json for item in sublist]
         
        return flattened_data

    return data_json

# def create_sub_masks(mask_image, width, height):
#     # Initialize a dictionary of sub-masks indexed by RGB colors
#     sub_masks = {}
#     for x in range(width):
#         for y in range(height):
#             # Get the RGB values of the pixel
#             pixel = mask_image.getpixel((x, y))[:3]

#             # Check to see if we have created a sub-mask...
#             pixel_str = str(pixel)
#             sub_mask = sub_masks.get(pixel_str)
#             if sub_mask is None:
#                 # Create a sub-mask (one bit per pixel) and add to the dictionary
#                 # Note: we add 1 pixel of padding in each direction
#                 # because the contours module doesn"t handle cases
#                 # where pixels bleed to the edge of the image
#                 sub_masks[pixel_str] = Image.new("1", (width + 2, height + 2))

#             # Set the pixel value to 1 (default is 0), accounting for padding
#             sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

#     return sub_masks

# def create_sub_masks(mask_array, width, height):
#     """
#     Create sub-masks from a numpy array representation of a mask.
    
#     Args:
#         mask_array (numpy.ndarray): The input mask as a NumPy array. Can be grayscale (2D) or RGB (3D).
#         width (int): The width of the mask.
#         height (int): The height of the mask.

#     Returns:
#         dict: A dictionary where keys are unique values (or RGB tuples for RGB images),
#               and values are binary NumPy arrays (sub-masks).
#     """
#     # Initialize a dictionary of sub-masks
#     sub_masks = {}
    
#     # Check if the mask is grayscale (2D) or RGB (3D)
#     if mask_array.ndim == 2:
#         # Grayscale: Extract unique values
#         unique_values = np.unique(mask_array)
#         for value in unique_values:
#             # Create a binary mask for the current value
#             color_mask = (mask_array == value)
            
#             # Convert the binary mask to a padded array
#             padded_mask = np.pad(color_mask, pad_width=1, mode='constant', constant_values=0)
            
#             # Add the sub-mask to the dictionary
#             sub_masks[str(value)] = padded_mask.astype(np.uint8)
#     elif mask_array.ndim == 3:
#         # RGB: Extract unique RGB values
#         unique_colors = np.unique(mask_array.reshape(-1, mask_array.shape[2]), axis=0)
#         for color in unique_colors:
#             # Create a binary mask for the current color
#             color_mask = np.all(mask_array == color, axis=-1)
            
#             # Convert the binary mask to a padded array
#             padded_mask = np.pad(color_mask, pad_width=1, mode='constant', constant_values=0)
            
#             # Add the sub-mask to the dictionary
#             sub_masks[str(tuple(color))] = padded_mask.astype(np.uint8)
#     else:
#         raise ValueError("mask_array must be either 2D (grayscale) or 3D (RGB).")

#     return sub_masks

# def create_sub_mask_annotation(sub_mask):
#     # Find contours (boundary lines) around each sub-mask
#     # Note: there could be multiple contours if the object
#     # is partially occluded. (E.g. an elephant behind a tree)
#     contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

#     polygons = []
#     segmentations = []
#     number = 0
#     for contour in contours:
#         # Flip from (row, col) representation to (x, y)
#         # and subtract the padding pixel
#         for i in range(len(contour)):
#             row, col = contour[i]
#             contour[i] = (col - 1, row - 1)

#         # if number == 94:
#         #     test = 1
#         # Make a polygon and simplify it
#         poly = Polygon(contour)
#         poly = poly.simplify(1.0, preserve_topology=False)

#         if (poly.is_empty):
#             # Go to next iteration, dont save empty values in list
#             continue

#         polygons.append(poly)

#         if isinstance(poly, MultiPolygon):
#             # Handle multiple Polygons
#             # print("Result is a MultiPolygon:")
#             for i, sub_poly in enumerate(poly.geoms):
#                 segmentation = np.array(sub_poly.exterior.coords).ravel().tolist()
#                 segmentations.append(segmentation)
#         elif isinstance(poly, Polygon):
#             segmentation = np.array(poly.exterior.coords).ravel().tolist()
#             segmentations.append(segmentation)
#         # number = number + 1
#         # print(poly)
#         # print(number)

#     return polygons, segmentations

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

    #create df
    # Define the column names
    columns = ["image_path", "class", "roi", "dice","path"]
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)
    
    #load models
    config_dir = "sam_road/config/"
    sam_road_checkpoint = "/home/datadisk/evaluation/models/SAM/SAM-road/"
    net_cityscale, config_cityscale = sam_road.load_sam_road(f"{config_dir}toponet_vitb_512_cityscale.yaml", f"{sam_road_checkpoint}cityscale_vitb_512_e10.ckpt", "cuda")
    net_spacenet, config_spacenet = sam_road.load_sam_road(f"{config_dir}toponet_vitb_256_spacenet.yaml", f"{sam_road_checkpoint}spacenet_vitb_256_e10.ckpt", "cuda")
    
    index = 23
    data_jsons = read_detection_json(args.json_path)#[index:index+1]
    dataset_name = "toy_set"
    file_testing_name = args.json_path.split("/")[-1].split(".")[0]
    
    sam_box_dir = "/home/datadisk/pipe/results/sam_box/"+dataset_name+"/"+file_testing_name+"/"
    
    visualiztion = False
    
    dice_small_list = []
    dice_sam_list = []
    dice_sam_combine_list = []
    
    image_path_list = []
    mask_gt_list = []
    mask_small_list = []
    mask_sam_list = []
    mask_sam_combine_list = []
    
        
    columns = ["image_path", "class", "dice_small", "dice_sam","dice_sam_com", "small_path", "sam_path", "sam_combine_path", "gt_path"]
    # Create an empty DataFrame with the specified columns
    save_df = pd.DataFrame(columns=columns)
    
    dataset_name = 'tester_road'
    file_testing_name = 'tester_road'
    sam_road_dir = "/home/datadisk/pipe/results/sam_road/"+dataset_name+"/"+file_testing_name+"/"
    
    for data_json in data_jsons:
        ##NEED remove
        
        image_path = open_image_with_nothing(data_json['image'][0])
        image_source = cv2.imread(image_path)
        H, W, _ = image_source.shape
        
        
        mask_segment_list = []
        for raod_obj in data_json["object_annotations"]:
            if visualiztion:
                print ("------------DATA-----------")
                print (raod_obj)
            
            segmentation_poly = raod_obj["segmentation"]
            if segmentation_poly!="":
                polygon = segmentation_poly
                if isinstance(polygon, list):
#                     print ('list poly')
                    binary_mask = polygon_to_mask(polygon, H, W)
                    
                elif type(polygon) == str:
#                     print ('str poly')
                    binary_mask = str_poly_to_mask(polygon, H, W)

                else:
                    print('print error nothing')
            mask_segment_list.append(binary_mask)
        
        #merge mask
        
        binary_mask = mask_segment_list[0]
        for segment_mask in mask_segment_list:
            binary_mask = np.logical_or(binary_mask, segment_mask)
        
        
        if visualiztion:
            print ("-------MASK----------")
            plt.imshow(binary_mask)
            plt.show()
        try:
            if (True):
                #road #1
                multi_scales = [1,2,3,4]
                scale_results = []
                scale_results_small = []
                scale_result_mask_small = []
                for scale in multi_scales:
                    #print ("scale:", scale)
                    padded_image = resize_to_square_and_pad(image_source, scale)
    #                 plt.imshow(padded_image)
    #                 plt.show()
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
                    #small_scale_result = remove_padding_and_resize(mask, image_source.shape)
                    scale_results_small.append(final_scale_result)
                    #scale_result_mask_small.append(small_scale_result)
        #             plt.imshow(final_scale_result)
        #             plt.show()

                #combined_mask = combine_multiscale_results(scale_results, main_image.shape[:2], method='average')

                image_sourceCopy = image_source.copy()
                combined_mask = combine_multiscale_results(scale_results,multi_scales, image_sourceCopy.shape[:2], method='average')
                combined_real_small = combine_multiscale_results(scale_results_small,multi_scales, (1024,1024,3), method='average')
                #combined_mask_small = combine_multiscale_results(scale_result_mask_small,multi_scales, (1024,1024,3), method='average')
                overlay_image = sam_road.overlay_mask_on_image(image_sourceCopy, combined_mask, color=(0, 255, 255), alpha=0.2,th=25)

                if visualiztion:

                    print ("-------Original----------")
                    plt.imshow(image_source)
                    plt.show()

                    plt.imshow(overlay_image)
                    plt.show()

                threshold = 25
                sam_combine_mask = (combined_mask > threshold).astype(int)
                if visualiztion:
                    print ('Sam(combine) TH:',threshold)
                
                ### SAVE Sam(combine) result mask


                
                ### END
                
                dice_sam_com = dice_coefficient(sam_combine_mask, binary_mask)
                dice_sam_combine_list.append(dice_sam_com)
                if visualiztion:
                    print ('DICE:',dice_sam_com)
                    plt.imshow(sam_combine_mask)
                    plt.show()      

                image_sourceCopy = image_source.copy()
                if visualiztion:
                    print ('SAM(single) TH: 15')
                try:

                    h, w = image_sourceCopy.shape[:2]
                    if w != h:
                        new_size = max(w, h)  # Choose the larger dimension to maintain aspect ratio
                        image_sourceCopy = cv2.resize(image_sourceCopy, (new_size, new_size))

                    pred_nodes_spacenet, pred_edges_spacenet, itsc_mask_spacenet, road_mask_spacenet = sam_road.infer_one_img(net_spacenet, image_sourceCopy, config_spacenet)
                    overlay_image = sam_road.overlay_mask_on_image(image_sourceCopy, road_mask_spacenet, color=(0, 255, 0), alpha=0.5,th=15)

                    road_mask_spacenet= np.resize(road_mask_spacenet, binary_mask.shape)
                    
                    ### SAVE SAM result mask
                    
                    
                    
                    ### END
                    
                    
                    dice_SAM = dice_coefficient(road_mask_spacenet, binary_mask)
                    if visualiztion:
                        print ('DICE:',dice_SAM)

                        plt.imshow(overlay_image)
                        plt.show()
                except:



                    print ('Error size need resize')

    #             print ('small model (combine)')
    #             dice = dice_coefficient(combined_mask_small, binary_mask)
    #             print ('DICE:',dice)
    #             plt.imshow(combined_mask_small)
    #             plt.show()


                #road #2

                print ('small model (single)')
                mask, blended_image = road_ex.road_extraction(image_path)
                mask= np.resize(mask, binary_mask.shape)

                print ('mask', mask.shape)
                print ('binary', binary_mask.shape)
                
                ### SAVE small model result mask

                

                ### END

                dice_small = dice_coefficient(mask, binary_mask)
                small_model_mask = mask
                if visualiztion:
                    print ('DICE:',dice_small)
                    plt.imshow(blended_image)
                    plt.show()
        except:
            print ("error:",image_path)
        
        
        #####
        

        
        save_folder = data_json['image'][0].split("/")[-1].split(".")[0]
        if not os.path.exists(sam_road_dir+save_folder):
            os.makedirs(sam_road_dir+save_folder)
        
        
        gt_final_path = sam_road_dir+save_folder+"/"+datetime.now().strftime('%Y%m%d%H%M%S%f')+"_gt.jpg"
        
        mask_small_final_path = sam_road_dir+save_folder+"/small_"+datetime.now().strftime('%Y%m%d%H%M%S%f')+".jpg"
        mask_sam_final_path = sam_road_dir+save_folder+"/sam_"+datetime.now().strftime('%Y%m%d%H%M%S%f')+".jpg"
        mask_sam_combine_final_path = sam_road_dir+save_folder+"/sam_com_"+datetime.now().strftime('%Y%m%d%H%M%S%f')+".jpg"

        try:
            cv2.imwrite(gt_final_path, binary_mask)
            cv2.imwrite(mask_small_final_path, small_model_mask)
            cv2.imwrite(mask_sam_final_path, road_mask_spacenet)
            cv2.imwrite(mask_sam_combine_final_path, sam_combine_mask)


        except:
            # Convert boolean mask to uint8 (0s and 255s)
            binary_gt_mask = np.uint8(binary_mask) * 255
            binary_sam_mask = np.uint8(road_mask_spacenet) * 255
            binary_small__mask = np.uint8(small_model_mask) * 255
            binary_sam_com__mask = np.uint8(sam_combine_mask) * 255

            # Save the binary mask as an image            
            cv2.imwrite(gt_final_path, binary_gt_mask)
            cv2.imwrite(mask_small_final_path, binary_small__mask)
            cv2.imwrite(mask_sam_final_path, binary_sam_mask)
            cv2.imwrite(mask_sam_combine_final_path, binary_sam_com__mask)
        #########
        
        image_path_list.append(data_json['image'][0])
        dice_small_list.append(dice_small)
        dice_sam_list.append(dice_SAM)
        dice_sam_combine_list.append(dice_sam_com)
        
        mask_gt_list.append(gt_final_path)
        mask_small_list.append(mask_small_final_path)
        mask_sam_list.append(mask_sam_final_path)
        mask_sam_combine_list.append(mask_sam_combine_final_path)
        
        new_row = {"image_path": data_json['image'][0], 
               "class": 'ROAD',
               "dice_small": dice_small, 
               "dice_sam": dice_SAM, 
               "dice_sam_com": dice_sam_com, 
               "small_path":mask_small_final_path, 
               "sam_path":mask_sam_final_path, 
               "sam_combine_path":mask_sam_combine_final_path, 
               "gt_path":gt_final_path}
        save_df = save_df.append(new_row, ignore_index=True)
    
    save_df.to_csv(sam_road_dir+'output.csv', index=False)
    return sam_road_dir+'output.csv', save_df


                

if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default="")
    parser.add_argument("--data-path", type=str, default="/goss/Datasets/")
    #sys.argv = ['test','--json-path', "/home/datadisk2/JSON/new_format/dataset2_JSON/VHRShips/Annotations_test.json"]
    #sys.argv = ['test','--json-path', "/home/datadisk2/JSON/new_format/dataset1_JSON/DOTA2.0/Annotations_train_hbb.json"]
    #sys.argv = ['test','--json-path', "/home/datadisk2/JSON/new_format/dataset1_JSON/million-AID/train_80_percent.json"]
    #sys.argv = ['test','--json-path', "/datadisk2/JSON/new_format/dataset2_JSON/iSAID/Annotations_val.json"]




    #sys.argv = ['test','--json-path', 'hbb_detection_toyset.json']
    #sys.argv = ['test','--json-path', "[PIX_SEG]BH-Pools_Watertanks_Datasets_Annotations_test.json"] 


    args = parser.parse_args()

    # Use the parsed arguments


    df = generate_segmentation(args)


    #read_detection_json(args)
    print(f"json_path: {args.json_path}")