import time

import torch
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps #增加
import torch.nn.functional as F
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, targetName, startLabel):
    findImgsIndex = []
    find_img_metas = []
    find_gt_bboxes = []
    find_gt_labels = []
    find_gt_keypoints = []
    for i in range(len(img_metas)):
        fileName = img_metas[i]["filename"]
        if fileName.split("/")[-2] == targetName:
            findImgsIndex.append(True)
            find_img_metas.append(img_metas[i])
            find_gt_bboxes.append(gt_bboxes[i])
            if fileName.split("/")[-2] == targetName == 'faceMohuImgs':
                mohu_label = gt_visibles[i]
                mohu_label_list = []
                for j in range(mohu_label.size(0)):
                    mohu_label_list.append(mohu_label[j][0].view(1))
                find_gt_label = torch.cat(mohu_label_list)
            else:
                find_gt_label = gt_labels[i] - startLabel
            find_gt_labels.append(find_gt_label)
            if fileName.split("/")[-2] == targetName == 'faceKpImgs':
                find_gt_keypoints.append(gt_keypoints[i])
        else:
            findImgsIndex.append(False)
    findImgsIndex = np.array(findImgsIndex)
    findImgsIndex = torch.from_numpy(findImgsIndex)

    #检测训练数据筛选
    find_x = []
    if len(find_img_metas) > 0:
        for i in range(5):
            find_x.append(x[i][findImgsIndex])

    if fileName.split("/")[-2] == targetName == 'faceKpImgs':
        # 人脸关键点训练数据筛选
        facekpfeature_depth = 5  # 特征图的层数
        facekp_x = []
        facekp_gt_bboxes_value = []
        facekp_gt_labels_value = []
        facekp_gt_keypoints_value = []
        if len(find_img_metas) > 0:

            for facefeature_i in range(facekpfeature_depth):
                facekp_x.append([])

            for i in range(len(find_gt_bboxes)):
                face_gt_keypoints_item = find_gt_keypoints[i]
                fisrtKp = face_gt_keypoints_item[:, 0, 1]
                validFaceKpIndex = torch.gt(fisrtKp, 0)
                facekp_gt_bboxes_item = find_gt_bboxes[i][validFaceKpIndex].clone()
                facekp_gt_labels_item = find_gt_labels[i][validFaceKpIndex].clone()
                if len(facekp_gt_bboxes_item > 0):
                    facekp_gt_bboxes_value.append(facekp_gt_bboxes_item)
                    facekp_gt_labels_value.append(facekp_gt_labels_item)
                    facekp_gt_keypoints_value.append(face_gt_keypoints_item[validFaceKpIndex].clone())
                    for facefeature_i in range(facekpfeature_depth):
                        facekp_x[facefeature_i].append(find_x[facefeature_i][i:i + 1].contiguous())

            for facekp_x_i in range(facekpfeature_depth):
                catx = facekp_x[facekp_x_i]
                facekp_x[facekp_x_i] = torch.cat(catx, 0)

        return facekp_x, find_img_metas, facekp_gt_bboxes_value, facekp_gt_keypoints_value
    return find_x, find_img_metas, find_gt_bboxes, find_gt_labels

def asserSameFile(img_metas):
    fileName = img_metas[0]["filename"].split("/")[-2]
    for i in range(1, len(img_metas)):
        assert fileName == img_metas[1]["filename"].split("/")[-2]

def rectBodyFactorLabels(x, img_metas, gt_bboxes, gt_labels, adult_label, start_label):
    bodyfeature_depth = 3  # 特征图的层数
    bodyfactor_x = []
    for bodyfeature_i in range(bodyfeature_depth):
        bodyfactor_x.append([])

    bodyfactor_max_width = 0
    bodyfactor_max_height = 0
    bodyfactor_img_metas = []
    bodyfactor_gt_bboxes = []
    bodyfactor_gt_labels = []
    bodyfactor_index = []

    for img_id in range(len(gt_bboxes)):
        gt_bboxes_perimg = gt_bboxes[img_id].clone()
        gt_labels_perimg = gt_labels[img_id].clone()

        # person目标真实值
        person_gt_indexs_perimg = torch.where(gt_labels_perimg == adult_label)
        person_gt_bboxes_perimg = gt_bboxes_perimg[person_gt_indexs_perimg]
        person_gt_labels_perimg = gt_labels_perimg[person_gt_indexs_perimg]

        # bodyfactor目标真实值
        bodyfactor_gt_indexs_perimg = torch.where(gt_labels_perimg > adult_label)
        bodyfactor_gt_bboxes_perimg = gt_bboxes_perimg[bodyfactor_gt_indexs_perimg]
        bodyfactor_gt_labels_perimg = gt_labels_perimg[bodyfactor_gt_indexs_perimg] - start_label  # 类别从1开始


        # 计算body_factor与person的iou,并判断body_factor所属的person，一个person作为一张图
        if bodyfactor_gt_bboxes_perimg.size(0) > 0:  # 判断图中是否有bodyfactor
            personwithbodyfactor_indexs = []  # person所对应的body_factor的下标
            for person_gt_i in range(len(person_gt_labels_perimg)):
                personwithbodyfactor_indexs.append([])
            bodyfactorandperson_gt_overlaps = bbox_overlaps(bodyfactor_gt_bboxes_perimg, person_gt_bboxes_perimg,
                                                            mode="iof")  # body_factor与person的iou交集
            bodyfactorandperson_gt_overlaps_argmax = torch.max(bodyfactorandperson_gt_overlaps, 1)[
                1]  # bodyfacor所对应的person的下标
            # 确定bodyfactor在person里面
            bodyfactorandperson_gt_overlaps_maxscore = torch.max(bodyfactorandperson_gt_overlaps, 1)[0]
            for i in range(bodyfactorandperson_gt_overlaps_maxscore.size(0)):
                assert bodyfactorandperson_gt_overlaps_maxscore[i] > 0.8
            # bodyfacor所对应的person的下标转化为person所对应的body_factor的下标
            for bodyfactorandperson_gt_overlaps_argmax_i in range(bodyfactorandperson_gt_overlaps_argmax.size(0)):
                personwithbodyfactor_indexs[
                    bodyfactorandperson_gt_overlaps_argmax[bodyfactorandperson_gt_overlaps_argmax_i].item()].append(
                    bodyfactorandperson_gt_overlaps_argmax_i)

            # 计算person的特征图范围，计算bodyfactorbbox（-person_bbox）
            for person_gt_i in range(len(person_gt_labels_perimg)):
                personwithbodyfactor_index = personwithbodyfactor_indexs[person_gt_i]
                personwithbodyfactor_person_bboxes = person_gt_bboxes_perimg[person_gt_i]
                personwithbodyfactor_bodyfactor_bboxes = bodyfactor_gt_bboxes_perimg[personwithbodyfactor_index]
                if personwithbodyfactor_bodyfactor_bboxes.size(0) == 0:
                    bodyfactor_index.append(False)
                    continue
                bodyfactor_index.append(True)
                personwithbodyfactor_bodyfactor_labels = bodyfactor_gt_labels_perimg[personwithbodyfactor_index]
                personx1, persony1, personx2, persony2 = personwithbodyfactor_person_bboxes
                # 计算person的特征图范围
                for x_i in range(bodyfeature_depth):
                    x_x1 = (personx1 / (2 ** (x_i + 2))).int()
                    x_y1 = (persony1 / (2 ** (x_i + 2))).int()
                    x_x2 = (personx2 / (2 ** (x_i + 2))).int()
                    x_y2 = (persony2 / (2 ** (x_i + 2))).int()
                    bodyfactor_x[x_i].append(x[x_i][img_id:img_id + 1, :, x_y1:x_y2,
                                             x_x1:x_x2].contiguous())
                # 计算bodyfactor的bbox
                for bodyfactor_bbox in personwithbodyfactor_bodyfactor_bboxes:
                    bodyfactor_bbox[0] = torch.max(bodyfactor_bbox[0], personx1)
                    bodyfactor_bbox[1] = torch.max(bodyfactor_bbox[1], persony1)
                    bodyfactor_bbox[2] = torch.min(bodyfactor_bbox[2], personx2)
                    bodyfactor_bbox[3] = torch.min(bodyfactor_bbox[3], persony2)
                    bodyfactor_bbox[0] -= personx1
                    bodyfactor_bbox[1] -= persony1
                    bodyfactor_bbox[2] -= personx1
                    bodyfactor_bbox[3] -= persony1
                bodyfactor_gt_bboxes.append(personwithbodyfactor_bodyfactor_bboxes)
                bodyfactor_gt_labels.append(personwithbodyfactor_bodyfactor_labels)

                bodyfactor_img_meta_perperson = img_metas[img_id].copy()
                # 计算对齐特征图的宽高和pad
                person_width = int(personx2 - personx1)
                person_height = int(persony2 - persony1)
                if person_width > bodyfactor_max_width:
                    bodyfactor_max_width = person_width
                if person_height > bodyfactor_max_height:
                    bodyfactor_max_height = person_height

                body_scale = bodyfactor_img_meta_perperson["scale_factor"]
                bodyfactor_img_meta_perperson["ori_shape"] = (
                person_height / body_scale[1], person_width / body_scale[0], 3)
                bodyfactor_img_meta_perperson["img_shape"] = (person_height, person_width, 3)
                bodyfactor_img_metas.append(bodyfactor_img_meta_perperson)

    pad_person_width = int(((bodyfactor_max_width + 16) / 16)) * 16
    pad_person_height = int(((bodyfactor_max_height + 16) / 16)) * 16
    # pad_person_width = person_max_width + 16
    # pad_person_height = person_max_height + 16
    for bodyfactor_img_meta_i in range(len(bodyfactor_img_metas)):
        bodyfactor_img_metas[bodyfactor_img_meta_i]["pad_shape"] = (pad_person_height, pad_person_width, 3)
    # assert len(bodyfactor_gt_labels) >= 1
    if len(bodyfactor_gt_labels) >= 1:
        for bodyfactor_gt_labels_i in range(len(bodyfactor_gt_labels)):
            for personsx_i in range(bodyfeature_depth):
                person_x = bodyfactor_x[personsx_i][bodyfactor_gt_labels_i]
                pad_width = int(pad_person_width / (2 ** (personsx_i + 2))) - person_x.size(3)
                pad_height = int(pad_person_height / (2 ** (personsx_i + 2))) - person_x.size(2)
                assert pad_width >= 0
                assert pad_height >= 0
                person_x = F.pad(person_x, (0, pad_width, 0, pad_height), "constant", value=0)
                bodyfactor_x[personsx_i][bodyfactor_gt_labels_i] = person_x
        for personsx_i in range(bodyfeature_depth):
            catx = bodyfactor_x[personsx_i]
            bodyfactor_x[personsx_i] = torch.cat(catx, 0)

    assert len(bodyfactor_gt_bboxes) > 0
    return bodyfactor_x, bodyfactor_gt_bboxes, bodyfactor_gt_labels, bodyfactor_img_metas ,bodyfactor_index

def simple_test_findTarget(det_bboxes, det_labels, points, startlabel, endLabel, minWidth=0, minHeight=0):
    index = torch.where(
        (det_labels >= startlabel) & (det_labels <= endLabel))
    target_labels = det_labels[index] - startlabel
    target_bboxes = det_bboxes[index]
    # 筛选目标大小
    if minWidth > 0 or minHeight > 0:
        bboxes_width = target_bboxes[:, 2] - target_bboxes[:, 0]
        bboxes_height = target_bboxes[:, 3] - target_bboxes[:, 1]
        valued_index = torch.where((bboxes_width > minWidth) & (bboxes_height > minHeight))
        target_bboxes = target_bboxes[valued_index]
        target_labels = target_labels[valued_index]
    # 电子围栏
    if points is not None:
        infence_index = []
        polygon = Polygon(points)
        for det_i in range(len(target_labels)):
            box = target_bboxes[det_i]
            if polygon.contains(Point([box[0], box[1]])) and polygon.contains(Point([box[2], box[3]])):
                infence_index.append(True)
            else:
                infence_index.append(False)
        target_labels = target_labels[infence_index]
        target_bboxes = target_bboxes[infence_index]
    return target_bboxes, target_labels, index


def simple_test_splitTarget(det_bboxes, det_labels, points, startFacelabel, endFaceLabel, startpersonLabel, endpersonLabel,
                            startCarLabel, endCarLabel,startPetLabel, endPetLabel, minWidth=0, minHeight=0):
    car_index = torch.where(
        (det_labels >= startCarLabel) & (det_labels <= endCarLabel))
    pets_index = torch.where(
        (det_labels >= startPetLabel) & (det_labels <= endPetLabel))
    person_index = torch.where(
        (det_labels >= startpersonLabel) & (det_labels <= endpersonLabel))
    face_index = torch.where(
        (det_labels >= startFacelabel) & (det_labels <= endFaceLabel))
    car_labels = det_labels[car_index] - startCarLabel
    car_bboxes = det_bboxes[car_index]
    pets_labels = det_labels[pets_index] - startPetLabel
    pets_bboxes = det_bboxes[pets_index]
    person_labels = det_labels[person_index] - startpersonLabel  # 去掉车辆，宠物，从头开始排序
    person_bboxes = det_bboxes[person_index]
    face_labels = det_labels[face_index] - startFacelabel
    face_bboxes = det_bboxes[face_index]

    bboxes_width = person_bboxes[:, 2] - person_bboxes[:, 0]
    bboxes_height = person_bboxes[:, 3] - person_bboxes[:, 1]
    factorValued_index = torch.where((bboxes_width > minWidth) & (bboxes_height > minHeight))
    person_bboxes = person_bboxes[factorValued_index]
    person_labels = person_labels[factorValued_index]

    if points is not None:
        # 电子围栏
        infence_index = []
        polygon = Polygon(points)
        for det_i in range(len(person_labels)):
            box = person_bboxes[det_i]
            if polygon.contains(Point([box[0], box[1]])) and polygon.contains(Point([box[2], box[3]])):
                infence_index.append(True)
            else:
                infence_index.append(False)
        person_labels = person_labels[infence_index]
        person_bboxes = person_bboxes[infence_index]

        infence_index = []
        polygon = Polygon(points)
        for det_i in range(len(car_labels)):
            box = car_bboxes[det_i]
            if polygon.contains(Point([(box[0]+box[2])/2, (box[1]+box[3])/2])):
                infence_index.append(True)
            else:
                infence_index.append(False)
        car_labels = car_labels[infence_index]
        car_bboxes = car_bboxes[infence_index]

        infence_index = []
        polygon = Polygon(points)
        for det_i in range(len(pets_labels)):
            box = pets_bboxes[det_i]
            if polygon.contains(Point([(box[0]+box[2])/2, (box[1]+box[3])/2])):
                infence_index.append(True)
            else:
                infence_index.append(False)
        pets_labels = pets_labels[infence_index]
        pets_bboxes = pets_bboxes[infence_index]

        infence_index = []
        polygon = Polygon(points)
        for det_i in range(len(face_labels)):
            box = face_bboxes[det_i]
            if polygon.contains(Point([(box[0]+box[2])/2, (box[1]+box[3])/2])):
                infence_index.append(True)
            else:
                infence_index.append(False)
        face_labels = face_labels[infence_index]
        face_bboxes = face_bboxes[infence_index]

    return person_bboxes, person_labels, face_bboxes, face_labels, car_bboxes, car_labels, pets_bboxes,  pets_labels

def simple_test_findPersonFeats(x, person_bboxes, img_metas):
    bodyfeature_depth = 3  # 特征图的层数
    persons_x_list = []
    person_img_metas = []
    scale_factor = img_metas[0]["scale_factor"]

    for i in range(len(person_bboxes)):
        personx1, persony1, personx2, persony2 = person_bboxes[i][:4].cpu().numpy() * scale_factor
        persons_x = []
        for x_i in range(bodyfeature_depth):
            x_x1 = int(personx1 / (2 ** (x_i + 2)))
            x_y1 = int(persony1 / (2 ** (x_i + 2)))
            x_x2 = int(personx2 / (2 ** (x_i + 2)))
            x_y2 = int(persony2 / (2 ** (x_i + 2)))
            persons_x.append(x[x_i][:, :, x_y1:x_y2, x_x1:x_x2].contiguous())
        person_img_meta = img_metas[0].copy()
        # 计算特征图的宽高和pad
        person_width = int(personx2 - personx1)
        person_height = int(persony2 - persony1)
        person_img_meta["ori_shape"] = (person_height/scale_factor[1], person_width/scale_factor[0], 3)
        person_img_meta["img_shape"] = (person_height, person_width, 3)
        person_img_meta["pad_shape"] = (person_height, person_width, 3)
        person_img_metas.append(person_img_meta)
        persons_x_list.append(persons_x)
    return persons_x_list, person_img_metas

def simple_test_findClearPerson(person_bboxes, person_labels, minWidth=100, minHeight=300):

    vaguePerson_labels, clearPerson_labels = [torch.full([0],0).type_as(person_labels) for i in range(2)]
    vaguePerson_bboxes, clearPerson_bboxes = [torch.full([0,5],0).type_as(person_bboxes) for i in range(2)]

    if person_bboxes.size(0) > 0:
        bboxes_width = person_bboxes[:,2] - person_bboxes[:,0]
        bboxes_height = person_bboxes[:,3] - person_bboxes[:,1]
        clear_index = torch.where((bboxes_width > minWidth) & (bboxes_height > minHeight))
        vague_index = torch.where((bboxes_width < minWidth) | (bboxes_height < minHeight))
        clearPerson_bboxes = person_bboxes[clear_index]
        clearPerson_labels = person_labels[clear_index]
        vaguePerson_bboxes = person_bboxes[vague_index]
        vaguePerson_labels = person_labels[vague_index]

    return clearPerson_bboxes, clearPerson_labels, vaguePerson_bboxes, vaguePerson_labels

def simple_test_BboxesConvert(bboxes, sub_bboxes):
    if bboxes.size(0) > 0:
        xmin, ymin = bboxes[0:2]
        for i in range(len(sub_bboxes)):
            sub_bboxes[i][0:4:2] += xmin
            sub_bboxes[i][1:4:2] += ymin
    return sub_bboxes

def simple_test_findDstLabels(bodyfactor_bboxes, bodyfactor_labels, start_label, end_label):
    dst_labels = torch.full([0],0).type_as(bodyfactor_bboxes)
    dst_bboxes = torch.full([0,5],0).type_as(bodyfactor_bboxes)
    if bodyfactor_labels.size(0) > 0:
        dst_index = torch.where((bodyfactor_labels >= start_label) & (bodyfactor_labels <= end_label))
        dst_labels = bodyfactor_labels[dst_index] - start_label
        dst_bboxes = bodyfactor_bboxes[dst_index]
    return dst_bboxes, dst_labels
