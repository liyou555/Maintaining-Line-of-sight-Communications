import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import json
import cv2
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
import pandas as pd
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from matting_api import Matting

import deocclusion.inference as infer
from deocclusion.demos.demo_utils import *
import time
import argparse

def evaluate_occlusion(image_path, json_path, dilate_kernel):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    height, width = image.shape[:2]
    deeplab = DeeplabV3(mix_type=1)
    modal, category, ori_bboxes, seg = deeplab.detect_image(image, count=False,
                                                            name_classes=[])

    x_min, y_min, W, H = map(int, ori_bboxes[0])
    amodalGT = np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    with open(json_path, 'r') as f:
        annotation_data = json.load(f)
    for shape in annotation_data['shapes']:
        label = shape['label']
        points = shape['points']
        polygon = [(int(point[0]), int(point[1])) for point in points]
        pil_mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(pil_mask)
        draw.polygon(polygon, fill=1)
        mask = np.array(pil_mask)
        if label == '4':
            amodalGT = mask
        elif label == '0':
            modal[0] = mask
        elif label == '5':
            modal[1] = mask
        mask = np.zeros((height, width), dtype=np.uint8)

    num_model = np.sum(modal[0] == 1)
    num_amodel_GT = np.sum(amodalGT == 1)
    bboxes = expand_bbox(ori_bboxes, enlarge_ratio=3., single_ratio=1.5)
    order_matrix = np.array([[0, -1], [1, 0]])
    amodal_patches_pred = infer.infer_amodal(
        pcnetm.model, image, modal, category, bboxes, order_matrix,
        use_rgb=pcnetm.use_rgb, th=0.5, dilate_kernel=dilate_kernel,
        input_size=256, min_input_size=16, interp='linear', debug_info=False)
    amodal_pred_ours = infer.patch_to_fullimage(
        amodal_patches_pred, bboxes, image.shape[0], image.shape[1], interp='linear')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    light_yellow = (0.6, 1, 0.6, 0.8)  # R, G, B, alpha
    masked_amodal = np.zeros((*amodalGT.shape, 4))
    nonzero_mask = amodalGT != 0  # 非零掩码
    masked_amodal[nonzero_mask] = light_yellow

    light_yellow = (0.6, 1, 0.6, 0.8)  # R, G, B, alpha
    masked_amodal = np.zeros((*modal[0].shape, 4))
    nonzero_mask = modal[0] != 0  # 非零掩码
    masked_amodal[nonzero_mask] = light_yellow

    light_yellow = (0.6, 1, 0.6, 0.8)  # R, G, B, alpha
    masked_amodal = np.zeros((*amodal_pred_ours[0].shape, 4))
    nonzero_mask = amodal_pred_ours[0] != 0
    masked_amodal[nonzero_mask] = light_yellow

    # 计算遮挡比例
    num_amodel = np.sum(amodal_pred_ours[0] == 1)
    image_name = image_path.split("/")[-1]
    occlusion_ratio = (num_amodel - num_model) / num_amodel * 100
    occlusion_ratio_GT = (num_amodel_GT - num_model) / num_amodel_GT * 100

    print(
        f"Image: {image_name}, segBeforeGT: {num_model}, segAfterGT: {num_amodel_GT}, Occlusion Ratio GT: {occlusion_ratio_GT:.2f} %")
    print(
        f"Image: {image_name}, segBefore: {num_model}, segAfter: {num_amodel}, Occlusion Ratio: {occlusion_ratio:.2f} %")

    # 在原图上绘制矩形框
    output_path = ""  # 保存路径
    cv2.rectangle(image_rgb, (x_min, y_min), (x_min+W, y_min+H), (0, 255, 0), 2)
    text = f"Occlusion: {occlusion_ratio:.2f}%"
    cv2.putText(image_rgb, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    plt.figure(figsize=(16, 16. / 480 * 640))
    plt.imshow(image_rgb)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

# PCNet-M
exp = './experiments/COCOA/pcnet_m'

config_file = exp + '/config.yaml'
load_model = exp + '/checkpoints/ckpt_iter_56000_all.pth.tar'

pcnetm = DemoPCNetM(config_file, load_model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

from deeplab import DeeplabV3
import matplotlib.pyplot as plt

# 图片路径
image_path = ''
# 标注路径
json_path = ''
evaluate_occlusion(image_path, json_path, 25)
