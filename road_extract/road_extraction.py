import cv2
import os
import numpy as np
import torch
import warnings
from torch.autograd import Variable as V
from framework import MyFrame
import matplotlib.pyplot as plt
from loss import dice_bce_loss
from networks.dlinknet import DinkNet34
from networks.nllinknet import NL34_LinkNet

DIR_PATH='/home/kt/python/TTTK/road_extract/'


warnings.filterwarnings("ignore")

def plt_and_show(img):
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
def blend_image_with_mask(image, mask, alpha=0.5):
    """
    Blends a segmentation mask onto an image with transparency and visualizes the result.
    
    :param image_path: Path to the original image.
    :param mask_path: Path to the binary segmentation mask (values 0 and 1).
    :param alpha: Transparency of the mask overlay (0.0 to 1.0).
    """
    # Load the original image
#     image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for visualization
    
    # Load the mask and ensure it's binary (0 or 1)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.clip(mask, 0, 1)
    
    # Create a red overlay for the mask
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 0] = 255  # Set red channel to 255
    
    # Blend the original image with the red overlay based on the mask
    blended_image = np.where(mask[:, :, None] == 1,
                             cv2.addWeighted(image, 1 - alpha, red_overlay, alpha, 0),
                             image)
    
    return mask, blended_image

# evaluate-folder

def road_extraction(img_path, model_weight = "weights/nllinknet.pt"):
    #img_path = "/home/datadisk/xyy/Projects/RoadExtract/samples/[PIX_SEG]_098_GID15_130854_GF2_PMS2__L1A0001389317-MSS2_ori.png"

    # nllnknet
    solver = MyFrame(NL34_LinkNet, dice_bce_loss, 2e-4)
    solver.load(DIR_PATH+model_weight)
    model_name = 'nllinknet'


    # 读取图片，分割
    img = cv2.imread(img_path)

    img = cv2.resize(img,(1024,1024))

    img_input = img[None, ...].transpose(0, 3, 1, 2)
    img_input = V(torch.Tensor(np.array(img_input, np.float32) / 255.0 * 3.2 - 1.6).cuda())
    predict = solver.test_one_img(img_input)
    predict = np.array(predict, np.int64)


    mask, blended_image = blend_image_with_mask(img, predict, alpha=0.5)
    
    return mask, blended_image

def road_extraction_image(img, model_weight = "weights/nllinknet.pt"):
    #img_path = "/home/datadisk/xyy/Projects/RoadExtract/samples/[PIX_SEG]_098_GID15_130854_GF2_PMS2__L1A0001389317-MSS2_ori.png"

    # nllnknet
    solver = MyFrame(NL34_LinkNet, dice_bce_loss, 2e-4)
    solver.load(DIR_PATH+model_weight)
    model_name = 'nllinknet'


    img_input = img[None, ...].transpose(0, 3, 1, 2)
    img_input = V(torch.Tensor(np.array(img_input, np.float32) / 255.0 * 3.2 - 1.6).cuda())
    predict = solver.test_one_img(img_input)
    predict = np.array(predict, np.int64)


    mask, blended_image = blend_image_with_mask(img, predict, alpha=0.5)
    
    return mask, blended_image
  
    