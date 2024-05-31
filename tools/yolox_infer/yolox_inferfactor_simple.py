from mmdet.apis import init_detector, inference_detector
import cv2
import time
import numpy as np
import os
import threading
from tools.drawUtils import *
from util import prefactor_fence
CLASSES = ['face', 'facewithmask', 'person', 'lianglunche', 'sanlunche', 'car', 'truck', 'dog', 'cat']
zitai_classes = ['微右', '微左', '正脸', '下左', '下右', '微下', '重上', '重下', '重右', '重左', '遮挡或半脸']
bodydetector_categoriesName = ['short_sleeves', 'long_sleeves', 'skirt', 'long_trousers', 'short_trousers', 'backbag',
                               'glasses', 'handbag', 'hat', 'haversack', 'trunk']
clousestyle_categoriesName = ['medium_long_style', 'medium_style', 'long_style']
clousecolor_categoriesName = ['light_blue', 'light_red', 'khaki', 'gray', 'blue', 'red', 'green', 'brown', 'yellow',
                              'purple', 'white', 'orange', 'deep_blue', 'deep_green', 'deep_red', 'black', 'stripe',
                              'lattice', 'mess', 'decor', 'blue_green']

config_file = '/home/chase/shy/dataset/spjgh/models/yolox_l_8x8_300e_coco_spjgh.py'
checkpoint_file = '/home/chase/shy/dataset/spjgh/models/epoch_300.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:1')

if __name__ == '__main__':
    # timeList = []
    # for i in range(20):
    #     # for imgName in os.listdir(imgPath):
    #     #     img = imgPath+'/'+imgName
    #     img = '/home/chase/shy/dataset/xinao/test/6132.jpg'
    #     # print(img)
    #     # test a single image and show the results
    #     img = cv2.imread(img)  # or img = mmcv.imread(img), which will only load it once
    #     start = time.time()
    #     resultList = inference_detector(model, img)
    #     timeList.append(time.time() - start)
    # print(min(timeList), max(timeList), np.mean(timeList))

    imgPath = '/home/chase/shy/138/images/'
    drawPath = '/home/chase/shy/138/draw2'
    for imgName in os.listdir(imgPath):
        print(imgName)
        start = time.time()
        img = cv2.imread(imgPath+'/'+imgName)
        # img = cv2.imread('/home/chase/Desktop/微信图片_20240408181351.png')
        result = inference_detector(model, img)
        print(time.time() - start)
        bboxes, labels, face_bboxes, face_labels, face_kps, face_zitais, face_mohus, body_bboxes, body_labels, upclouse_bboxes, upclouse_styles, clouse_bboxes, clouse_labels, clouse_colors = result
        # bboxes, labels, face_bboxes, face_kps, face_zitais, face_mohus, body_bboxes, body_labels, upclouse_bboxes, upclouse_styles, clouse_bboxes, clouse_labels, clouse_colors = result
        # prefactor_fence(bboxes, labels, face_bboxes, face_kps, face_zitais, face_mohus, body_bboxes, body_labels, upclouse_bboxes, upclouse_styles, clouse_bboxes, clouse_labels, clouse_colors, None)
        drawBboxes(img, bboxes, labels, CLASSES)
        drawBboxes(img, body_bboxes, body_labels, bodydetector_categoriesName)
        # drawFacekps(img, face_kps)
        # rectLabels(img, face_bboxes, face_zitais, zitai_classes, drawPath, imgName)
        # rectLabels(img, face_bboxes, face_mohus, zitai_classes, drawPath, imgName, type='mohu')
        # rectLabels(img, body_bboxes, body_labels, bodydetector_categoriesName, drawPath, imgName)
        # rectLabels(img, upclouse_bboxes, upclouse_styles, clousestyle_categoriesName, drawPath, imgName)
        # rectLabels(img, clouse_bboxes, clouse_colors, clousecolor_categoriesName, drawPath, imgName)

        cv2.imwrite(drawPath+'/'+imgName, img)
