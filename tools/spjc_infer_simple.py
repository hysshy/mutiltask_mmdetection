import os

from mmdet.apis import init_detector, inference_detector
import mmcv
import time
import cv2

config_file = '/home/chase/shy/mmdetection/faster_rcnn_r50_fpn_2x_spjgh.py'
checkpoint_file = '/home/chase/shy/mmdetection/tools/work_dirs/faster_rcnn_r50_fpn_2x_coco/epoch_12.pth'
face_classes = ['face', 'facewithmask']
person_classes = ['person', 'lianglunche']
pet_classes = ['dog', 'cat']
car_classes = ['sanlunche', 'car', 'truck']
zitai_classes = ['微右', '微左', '正脸', '下左', '下右', '微下', '重上', '重下', '重右', '重左', '遮挡或半脸']
colorList = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:1')

imgPath = '/home/chase/shy/dataset/xinao/img'
savePath = '/home/chase/shy/dataset/xinao/draw'
for imgName in os.listdir(imgPath):
    img = imgPath+'/'+imgName
    # test a single image and show the results
    img = cv2.imread(img)  # or img = mmcv.imread(img), which will only load it once
    start = time.time()
    person_bboxes, person_labels, car_bboxes, car_labels, pets_bboxes, pets_labels, face_bboxes, face_labels, zitai_labels, face_kps, faceMohus = \
        inference_detector(model, img)
    # print(time.time()-start)
    for i in range(len(face_bboxes)):
        bbox = face_bboxes[i].astype(int)
        # if bbox[3]-bbox[1]<40 or bbox[2]-bbox[0]< 40:
        #     continue
        faceKp = face_kps[i].astype(int)
        for k in range(5):
            point = (faceKp[k][0], faceKp[k][1])
            cv2.circle(img, point, 3, (255, 0, 0), 0)
        label = face_labels[i]
        if label == 0:
            zitai_label = zitai_labels[i]
            print(zitai_label)
            mohu_label = faceMohus[i]
            if not os.path.exists(savePath+'/'+zitai_classes[zitai_label]):
                os.makedirs(savePath+'/'+zitai_classes[zitai_label])
            cv2.imwrite(savePath+'/'+zitai_classes[zitai_label]+'/'+imgName.replace('.jpg', str(i)+'.jpg'), img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            mohudu = round(mohu_label[0], 2)
            if not os.path.exists(savePath+'/'+str(mohudu)):
                os.makedirs(savePath+'/'+str(mohudu))
            cv2.imwrite(savePath+'/'+str(mohudu)+'/'+imgName.replace('.jpg', str(i)+'.jpg'), img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        #     if mohudu <= 0.5:
        #         if not os.path.exists(savePath + '/' + '模糊'):
        #             os.makedirs(savePath + '/' + '模糊')
        #         cv2.imwrite(savePath + '/' + '模糊' + '/' + imgName.replace('.jpg', str(i) + '.jpg'),
        #                     img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        #     else:
        #         if zitai_classes[zitai_label] in ['微右', '微左', '正脸']:
        #             if not os.path.exists(savePath + '/' + '可识别'):
        #                 os.makedirs(savePath + '/' + '可识别')
        #             cv2.imwrite(savePath + '/' + '可识别' + '/' + imgName.replace('.jpg', str(i) + '.jpg'),
        #                         img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        #         else:
        #             if not os.path.exists(savePath + '/' + '姿态不可识别'):
        #                 os.makedirs(savePath + '/' + '姿态不可识别')
        #             cv2.imwrite(savePath + '/' + '姿态不可识别' + '/' + imgName.replace('.jpg', str(i) + '.jpg'),
        #                         img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 1)
        cv2.putText(img, face_classes[label], (bbox[0], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    for i in range(len(person_bboxes)):
        bbox = person_bboxes[i].astype(int)
        if len(bbox) > 0:
            label = person_labels[i]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            cv2.putText(img, person_classes[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)

    for i in range(len(car_bboxes)):
        bbox = car_bboxes[i].astype(int)
        if len(bbox) > 0:
            label = car_labels[i]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            cv2.putText(img, car_classes[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)

    for i in range(len(pets_bboxes)):
        bbox = pets_bboxes[i].astype(int)
        if len(bbox) > 0:
            label = pets_labels[i]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            cv2.putText(img, pet_classes[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)
    cv2.imwrite(savePath+'/'+imgName, img)