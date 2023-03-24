from mmdet.apis import init_detector, inference_detector
import cv2
import time
import os
import threading
from Log import logger

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

config_file = '/home/chase/shy/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py'
checkpoint_file = '/home/chase/shy/mmdetection/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
classes = ('electric','bicycle', 'lunyi')
rtsp = "rtsp://admin:windaka123@10.10.51.208"
cap = cv2.VideoCapture(rtsp)
cap.open(rtsp)
id = 0
camera_img = None
cv2.namedWindow('Test', 0)

def readFrame():
    while 1:
        global camera_img
        ret, camera_img = cap.read()
        # camera_img = cv2.resize(camera_img, (1280, 720))

if __name__ == '__main__':
    t = threading.Thread(target=readFrame)
    t.start()
    time.sleep(2)
    # global camera_img
    while 1:
        img = camera_img
        if img is not None:
            t = time.time()
            result = inference_detector(model, img)
            bboxes, labels = result
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            for i in range(len(bboxes)):
                bbox = bboxes[i].astype(int)
                logger.info('find ' + CLASSES[labels[i]])
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 5)
                cv2.putText(img, CLASSES[labels[i]], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 9)
            cv2.imshow('Test', img)
            cv2.waitKey(1)
