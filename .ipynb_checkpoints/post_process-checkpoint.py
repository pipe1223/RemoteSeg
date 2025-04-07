import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_opening, binary_closing, disk, ball
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray, gray2rgb

#-------------- post processing --------------#
def post_process(pred_mask,box_area,obj_class,box_area_thred=1500):
    '''
    pred_mask: binary mask, with 3 channels(rgb)
    box_area: box size, should be height*width
    obj_class: object category
    box_area_thred: the threshold of box size for post processing
    return post_mask: binary mask with 3 channels(rgb)
    '''
    post_mask = rgb2gray(pred_mask)   # to gray
    
    if box_area > box_area_thred:
        if obj_class == 'airport':
            post_mask = remove_small_regions(post_mask,min_size=1000)  # remove small regions
        else:
            post_mask = remove_small_regions(post_mask,min_size=100)   # remove small regions

        post_mask = binary_fill_holes(post_mask)  # fill holes

        if obj_class not in ['helicopter','airplane','harbor','windmill']:
            post_mask = binary_opening(post_mask, disk(3))   # opening, remove noise
            
        post_mask = binary_closing(post_mask, disk(3))  # closing, fill small holes
        
        post_mask = binary_fill_holes(post_mask)  # fill holes
        post_mask = gaussian_filter(post_mask.astype(float), sigma=1) > 0.5   # smoothing boundaries

    else:
        
        post_mask = binary_fill_holes(post_mask)  # fill holes
        post_mask = gaussian_filter(post_mask.astype(float), sigma=1) > 0.5   # smoothing boundaries

    post_mask = np.where(post_mask, 255, 0).astype(np.uint8)
    post_mask = gray2rgb(post_mask)

    return post_mask


def remove_small_regions(mask, min_size):
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    for region in regions:
        if region.area < min_size:
            labeled_mask[labeled_mask == region.label] = 0
    return labeled_mask > 0