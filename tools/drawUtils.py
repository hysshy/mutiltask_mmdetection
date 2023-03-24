import cv2
import os
colorList = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

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

def rectLabels(img, bboxes, labels, classes, savePath, imgName, type='zitai'):
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
        print(bbox)
        cv2.imwrite(savePath + '/' + name + '/' + imgName.replace('.jpg', '_'+str(i) + '.jpg'),
                    img[bbox[1]:bbox[3], bbox[0]:bbox[2]])