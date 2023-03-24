import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
import mmcv
import time
import cv2

config_file = '/home/chase/shy/mmdetection/faster_rcnn_r50_fpn_2x_spjgh.py'
checkpoint_file = '/home/chase/shy/epoch_12.pth'
face_classes = ['face', 'facewithmask']
person_classes = ['person', 'lianglunche']
pet_classes = ['dog', 'cat']
car_classes = ['sanlunche', 'car', 'truck']
zitai_classes = ['微右', '微左', '正脸', '下左', '下右', '微下', '重上', '重下', '重右', '重左', '遮挡或半脸']
upclouse_classes = ["short_sleeves", "long_sleeves"]
downclouse_classes = [ "skirt", "short_trousers", "long_trousers"]
otherfactor_classes = ["glasses", "hat", "handbag", "haversack", "backbag", 'trunk']
color_classes = ["black", "white", "gray", "blue", "red", "yellow", 'orange', "green", "brown","purple", "light_blue", "light_red", "deep_blue", "deep_green", "deep_red", "blue_green"]
upclouse_style_classes = ['medium_style', 'long_style']

colorList = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:1')

def drawBboxes(img, bboxes, labels, classes, extraLabels=None, extraClasses=None, extraLabels2=None, extraClasses2=None):
    for i in range(len(bboxes)):
        bbox = bboxes[i].astype(int)
        if len(bbox) > 0:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            label = labels[i]
            name = classes[label]
            if extraLabels is not None:
                name = name + '_'  + extraClasses[extraLabels[i]]
            if extraLabels2 is not None:
                name = name + '_' + extraClasses2[extraLabels2[i]]
            cv2.putText(img, name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)

def drawFacekps(img, face_kps):
    for i in range(len(face_kps)):
        faceKp = face_kps[i].astype(int)
        for k in range(5):
            point = (faceKp[k][0], faceKp[k][1])
            cv2.circle(img, point, 3, (255, 0, 0), 0)

def rectLabels(img, bboxes, labels, classes, type='zitai'):
    for i in range(len(bboxes)):
        bbox = bboxes[i].astype(int)
        label = labels[i]
        if type == 'mohu':
            mohu_label = labels[i]
            name = str(round(mohu_label[0], 2))
        else:
            name = classes[label]
        if not os.path.exists(savePath + '/' + name):
            os.makedirs(savePath + '/' + name)
        cv2.imwrite(savePath + '/' + name + '/' + imgName.replace('.jpg', '_'+str(i) + '.jpg'),
                    img[bbox[1]:bbox[3], bbox[0]:bbox[2]])

imgPath = '/home/chase/shy/dataset/xinao/test'
savePath = '/home/chase/shy/dataset/xinao/draw2'
timeList = []
for i in range(20):
# for imgName in os.listdir(imgPath):
#     img = imgPath+'/'+imgName
    img = '/home/chase/shy/dataset/xinao/test/6132.jpg'
    # print(img)
    # test a single image and show the results
    img = cv2.imread(img)  # or img = mmcv.imread(img), which will only load it once
    start = time.time()
    person_bboxes, person_labels, car_bboxes, car_labels, pets_bboxes, pets_labels, face_bboxes, face_labels, \
    zitai_labels, face_kps, faceMohus, clearPerson_bboxes, clearPerson_labels, upclouse_factor_bboxes, upclouse_factor_labels, \
    upclouse_color_labels, upclouse_style_labels, downclouse_factor_bboxes, downclouse_factor_labels, downclouse_color_labels, other_factor_bboxes, other_factor_labels\
        = inference_detector(model, img)
    timeList.append(time.time()-start)
print(min(timeList), max(timeList), np.mean(timeList))
    # drawBboxes(img, clearPerson_bboxes, clearPerson_labels, person_classes)
    # drawBboxes(img, car_bboxes, car_labels, car_classes)
    # drawBboxes(img, pets_bboxes, pets_labels, pet_classes)
    # drawBboxes(img, face_bboxes, face_labels, face_classes)
    # drawFacekps(img, face_kps)
    # rectLabels(img, face_bboxes, zitai_labels, zitai_classes)
    # rectLabels(img, face_bboxes, faceMohus, zitai_classes, type='mohu')
    # for i in range(len(upclouse_factor_bboxes)):
    #     drawBboxes(img, upclouse_factor_bboxes[i], upclouse_factor_labels[i], upclouse_classes, upclouse_color_labels[i], color_classes, upclouse_style_labels[i], upclouse_style_classes)
    # for i in range(len(downclouse_factor_bboxes)):
    #     drawBboxes(img, downclouse_factor_bboxes[i], downclouse_factor_labels[i], downclouse_classes, downclouse_color_labels[i], color_classes)
    # for i in range(len(other_factor_bboxes)):
    #     drawBboxes(img, other_factor_bboxes[i], other_factor_labels[i], otherfactor_classes)
    # cv2.imwrite(savePath+'/'+imgName, img)