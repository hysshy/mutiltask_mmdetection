from mmdet.apis import init_detector, inference_detector
import cv2
import time
import numpy as np
import os
import threading
from Log import logger
from drawUtils import *
CLASSES = ['face', 'facewithmask', 'person', 'lianglunche', 'sanlunche', 'car', 'truck', 'dog', 'cat']
zitai_classes = ['微右', '微左', '正脸', '下左', '下右', '微下', '重上', '重下', '重右', '重左', '遮挡或半脸']

config_file = '/home/chase/shy/mmdetection/yolox_m_8x8_300e_coco_spjgh.py'
checkpoint_file = '/home/chase/shy/mmdetection/work_dirs/yolox_m_8x8_300e_coco2_kp/epoch_300.pth'
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

    imgPath = '/home/chase/Desktop/138/images'
    drawPath = '/home/chase/shy/dataset/xinao/draw5'
    for imgName in os.listdir(imgPath):
        img = cv2.imread(imgPath+'/'+imgName)
        start = time.time()
        result = inference_detector(model, img)
        print(time.time() - start)
        bboxes, labels, face_bboxes, face_kps, face_zitais, face_mohus = result
        drawBboxes(img, bboxes, labels, CLASSES)
        drawFacekps(img, face_kps)
        rectLabels(img, face_bboxes, face_zitais, zitai_classes, drawPath, imgName)
        rectLabels(img, face_bboxes, face_mohus, zitai_classes, drawPath, imgName, type='mohu')
        cv2.imwrite(drawPath+'/'+imgName, img)
