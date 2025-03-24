import numpy as np
import os
import imageio
import torch
import cv2

from utils_road import load_config, create_output_dir_and_save_config
from dataset import cityscale_data_partition, read_rgb_img, get_patch_info_one_img
from dataset import spacenet_data_partition
from model import SAMRoad
import graph_extraction
import graph_utils
import triage
import pickle
import scipy
import rtree
from collections import defaultdict
import time
import PIL

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import json

def get_img_paths(root_dir, image_indices):
    img_paths = []

    for ind in image_indices:
        img_paths.append(os.path.join(root_dir, f"region_{ind}_sat.png"))

    return img_paths

def overlay_mask_on_image(img, road_mask, color=(0, 0, 255), alpha=0.5, th =100):
    # Ensure the mask has the same shape as the image
    if img.shape[:2] != road_mask.shape:
        road_mask = cv2.resize(road_mask, (img.shape[1], img.shape[0]))
    
    # Create a color version of the mask with the desired color
    # Mask values should be 0 or 1 (binary)
    color_mask = np.zeros_like(img)
    color_mask[:, :] = color  # Fill with the desired color
    
    # Convert the grayscale mask to a binary mask where non-zero values are 1
    binary_mask = road_mask > th
    
    # Create the final colored mask using the binary mask
    color_mask = np.where(binary_mask[:, :, None], color_mask, 0)

    # Overlay the colored mask on the original image using alpha blending
    overlayed_image = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
    
    return overlayed_image


# def crop_img_patch(img, x0, y0, x1, y1):
#      return img[y0:y1, x0:x1, :]

def crop_img_patch(img, x0, y0, x1, y1, target_size=256):
    """
    Crop the patch from the image and pad it to the target size.
    
    Args:
        img (numpy array): The input image (height, width, channels).
        x0, y0 (int): Top-left corner of the patch.
        x1, y1 (int): Bottom-right corner of the patch.
        target_size (int): The target size for height and width (default is 256).
    
    Returns:
        numpy array: The cropped and padded patch of size (target_size, target_size, channels).
    """
    # Ensure the patch is within the image bounds
    h, w, c = img.shape
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)

    # Crop the patch
    patch = img[y0:y1, x0:x1]

    # Calculate padding
    pad_h = target_size - patch.shape[0]
    pad_w = target_size - patch.shape[1]

    # If the patch is smaller than the target size, pad it
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        patch = cv2.copyMakeBorder(
            patch,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black padding
        )

    return patch


def remove_padding(patch, original_shape):
    """
    Remove padding from a patch to restore its original shape.

    Args:
        patch (tensor): The padded patch (H, W, C).
        original_shape (tuple): The original shape of the patch (height, width).

    Returns:
        tensor: The patch with padding removed.
    """
    original_h, original_w = original_shape
    padded_h, padded_w = patch.shape[:2]

    # Calculate how much padding to remove
    pad_top = (padded_h - original_h) // 2
    pad_left = (padded_w - original_w) // 2

    # Crop the padding
    return patch[pad_top : pad_top + original_h, pad_left : pad_left + original_w]


def get_batch_img_patches(img, batch_patch_info):
    patches = []
    print (img.shape)
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def infer_one_img(net, img, config, device='cuda'):
    # TODO(congrui): centralize these configs
    image_size = img.shape[0]

    batch_size = config.INFER_BATCH_SIZE
    # list of (i, (x_begin, y_begin), (x_end, y_end))
    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE)
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )

    

    # [IMG_H, IMG_W]
    fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(device, non_blocking=False)
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(device, non_blocking=False)
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(device, non_blocking=False)

    # stores img embeddings for toponet
    # list of [B, D, h, w], len=batch_num
    img_features = list()

    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(device, non_blocking=False)
            # [B, H, W, 2]
            mask_scores, patch_img_features = net.infer_masks_and_img_features(batch_img_patches)
            img_features.append(patch_img_features)
        # Aggregate masks
#         for patch_index, patch_info in enumerate(batch_patch_info):
#             _, (x0, y0), (x1, y1) = patch_info
#             keypoint_patch, road_patch = mask_scores[patch_index, :, :, 0], mask_scores[patch_index, :, :, 1]
#             fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
#             fused_road_mask[y0:y1, x0:x1] += road_patch
#             pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=device)
        
        #pipe modify
        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            
            # Extract the processed patch from `mask_scores`
            keypoint_patch, road_patch = (
                mask_scores[patch_index, :, :, 0],
                mask_scores[patch_index, :, :, 1],
            )

            # Remove padding from the patch
            original_h = y1 - y0
            original_w = x1 - x0
#             print("---------------------------")
#             print(f"Original patch dimensions: ({original_h}, {original_w})")
#             print(f"Keypoint patch dimensions before cropping: {keypoint_patch.shape}")
#             print(f"Road patch dimensions before cropping: {road_patch.shape}")
            
            keypoint_patch = remove_padding(keypoint_patch, (original_h, original_w))
            road_patch = remove_padding(road_patch, (original_h, original_w))
                        
#             print(f"Cropped keypoint patch dimensions: {keypoint_patch.shape}")
#             print(f"Cropped road patch dimensions: {road_patch.shape}")

            # Merge the cropped patch back into the fused masks
            fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            
#             print(f"Target fusion region dimensions: ({y1-y0}, {x1-x0})")
            pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=device)
            
            
    fused_keypoint_mask /= pixel_counter
    fused_road_mask /= pixel_counter
    # range 0-1 -> 0-255
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()

    # ## Astar graph extraction
    # pred_graph = graph_extraction.extract_graph_astar(fused_keypoint_mask, fused_road_mask, config)
    # # Doing this conversion to reuse copied code
    # pred_nodes, pred_edges = graph_utils.convert_from_nx(pred_graph)
    # return pred_nodes, pred_edges, fused_keypoint_mask, fused_road_mask
    # ## Astar graph extraction
    
    
    ## Extract sample points from masks
    graph_points = graph_extraction.extract_graph_points(fused_keypoint_mask, fused_road_mask, config)
    if graph_points.shape[0] == 0:
        return graph_points, np.zeros((0, 2), dtype=np.int32), fused_keypoint_mask, fused_road_mask

    # for box query
    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = v
        # hack to insert single points
        graph_rtree.insert(i, (x, y, x, y))
    
    ## Pass 2: infer toponet to predict topology of points from stored img features
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]

        topo_data = {
            'points': [],
            'pairs': [],
            'valid': [],
        }
        idx_maps = []


        # prepares pairs queries
        for patch_info in batch_patch_info:
            _, (x0, y0), (x1, y1) = patch_info
            patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {patch_idx : all_idx for patch_idx, all_idx in enumerate(patch_point_indices)}
            patch_point_num = len(patch_point_indices)
            # normalize into patch
            patch_points = graph_points[patch_point_indices, :] - np.array([[x0, y0]], dtype=graph_points.dtype)
            # for knn and circle query
            patch_kdtree = scipy.spatial.KDTree(patch_points)

            # k+1 because the nearest one is always self
            # idx is to the patch subgraph
            knn_d, knn_idx = patch_kdtree.query(patch_points, k=config.MAX_NEIGHBOR_QUERIES + 1, distance_upper_bound=config.NEIGHBOR_RADIUS)
            # [patch_point_num, n_nbr]
            knn_idx = knn_idx[:, 1:]  # removes self
            # [patch_point_num, n_nbr] idx is to the patch subgraph
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, config.MAX_NEIGHBOR_QUERIES)
            )
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            # [patch_point_num, n_nbr, 2]
            pairs = np.stack([src_idx, tgt_idx], axis=-1)

            topo_data['points'].append(patch_points)
            topo_data['pairs'].append(pairs)
            topo_data['valid'].append(valid)
            idx_maps.append(idx_patch2all)
        
        # collate
        collated = {}
        for key, x_list in topo_data.items():
            length = max([x.shape[0] for x in x_list])
            collated[key] = np.stack([
                np.pad(x, [(0, length - x.shape[0])] + [(0, 0)] * (len(x.shape) - 1))
                for x in x_list
            ], axis=0)

        # skips this batch if there's no points
        if collated['points'].shape[1] == 0:
            continue
        
        # infer toponet
        # [B, D, h, w]
        batch_features = img_features[batch_index]
        # [B, N_sample, N_pair, 2]
        batch_points = torch.tensor(collated['points'], device=device)
        batch_pairs = torch.tensor(collated['pairs'], device=device)
        batch_valid = torch.tensor(collated['valid'], device=device)


        with torch.no_grad():
            # [B, N_samples, N_pairs, 1]
            topo_scores = net.infer_toponet(batch_features, batch_points, batch_pairs, batch_valid)
                
        # all-invalid (padded, no neighbors) queries returns nan scores
        # [B, N_samples, N_pairs]
        topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()

        # aggregate edge scores
        batch_size, n_samples, n_pairs = topo_scores.shape
        for bi in range(batch_size):
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not collated['valid'][bi, si, pi]:
                        continue
                    # idx to the full graph
                    src_idx_patch, tgt_idx_patch = collated['pairs'][bi, si, pi, :]
                    src_idx_all, tgt_idx_all = idx_maps[bi][src_idx_patch], idx_maps[bi][tgt_idx_patch]
                    edge_score = topo_scores[bi, si, pi]
                    assert 0.0 <= edge_score <= 1.0
                    edge_scores[(src_idx_all, tgt_idx_all)] += edge_score
                    edge_counts[(src_idx_all, tgt_idx_all)] += 1.0

    # avg edge scores and filter
    pred_edges = []
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge] 
        if score > config.TOPO_THRESHOLD:
            pred_edges.append(edge)
    pred_edges = np.array(pred_edges).reshape(-1, 2)
    pred_nodes = graph_points[:, ::-1]  # to rc
    
    

    return pred_nodes, pred_edges, fused_keypoint_mask, fused_road_mask

def load_sam_road(config_file, check_point, device):
    print (config_file)
    config = load_config(config_file)
    
    # Builds eval model    
    device = torch.device(device)#
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    net = SAMRoad(config)

    checkpoint = torch.load(check_point, map_location=device)
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)
    
    return net, config

