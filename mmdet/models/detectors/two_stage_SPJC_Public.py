import os.path
import math
import torch
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from . import labelstransform
import warnings
import numpy as np
from mmdet.models.attention.generalizedAttention import GeneralizedAttention


@DETECTORS.register_module()
class TwoStageDetector_SPJC_Public(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector_SPJC_Public, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        self.neck_names = neck.pop('neck_names')
        self.attentionType = neck.pop('attention')
        self.convtype = neck.pop('convtype')
        if neck is not None:
            if 'backbone_neck' in self.neck_names:
                self.backbone_neck = build_neck(neck)
                if self.attentionType == 'GA':
                    self.attention_backbone = GeneralizedAttention(in_channels=256, num_heads=8, attention_type='0010',
                                                          convtype=self.convtype)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            self.rpn_head_Dict = {}
            for i in range(len(rpn_head)):
                rpn_head_type = rpn_head[i].pop('rpn_head_type')
                rpn_head[i].update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
                if rpn_head_type == 'faceDetect':
                    self.rpn_head_faceDetect = build_head(rpn_head[i])
                    self.rpn_head_Dict.setdefault(rpn_head_type, self.rpn_head_faceDetect)
                elif rpn_head_type == 'carplateDetect':
                    self.rpn_head_carplateDetect = build_head(rpn_head[i])
                    self.rpn_head_Dict.setdefault(rpn_head_type, self.rpn_head_carplateDetect)
                elif rpn_head_type == 'carDetect':
                    self.rpn_head_carDetect = build_head(rpn_head[i])
                    self.rpn_head_Dict.setdefault(rpn_head_type, self.rpn_head_carDetect)
                elif rpn_head_type == 'cocoDetect':
                    self.rpn_head_cocoDetect = build_head(rpn_head[i])
                    self.rpn_head_Dict.setdefault(rpn_head_type, self.rpn_head_cocoDetect)
                elif rpn_head_type == 'cocoMask':
                    self.rpn_head_cocoMask = build_head(rpn_head[i])
                    self.rpn_head_Dict.setdefault(rpn_head_type, self.rpn_head_cocoMask)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            self.roi_head_Dict = {}
            self.attentionDict = {}
            self.neckDict = {}
            for i in range(len(roi_head)):
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg.rcnn)
                roi_head_type = roi_head[i].pop('roi_head_type')
                if roi_head_type == 'faceDetect':
                    self.roi_head_faceDetect = build_head(roi_head[i])
                    self.roi_head_Dict.setdefault(roi_head_type, self.roi_head_faceDetect)
                    if 'task_neck' in self.neck_names:
                        self.neck_faceDetect = build_neck(neck)
                        self.neckDict.setdefault(roi_head_type, self.neck_faceDetect)
                    if self.attentionType == 'GA2':
                        self.attention_faceDetect = GeneralizedAttention(in_channels=256, num_heads=8,
                                                                         attention_type='0010',
                                                                         convtype=self.convtype)
                        self.attentionDict.setdefault(roi_head_type, self.attention_faceDetect)
                elif roi_head_type == 'faceGender':
                    self.roi_head_faceGender = build_head(roi_head[i])
                    self.roi_head_Dict.setdefault(roi_head_type, self.roi_head_faceGender)
                    if 'task_neck' in self.neck_names:
                        self.neck_faceGender = build_neck(neck)
                        self.neckDict.setdefault(roi_head_type, self.neck_faceGender)
                    if self.attentionType == 'GA2':
                        self.attention_faceGender = GeneralizedAttention(in_channels=256, num_heads=8,
                                                                         attention_type='0010',
                                                                         convtype=self.convtype)
                        self.attentionDict.setdefault(roi_head_type, self.attention_faceGender)

                elif roi_head_type == 'faceKp':
                    self.roi_head_faceKp = build_head(roi_head[i])
                    self.roi_head_Dict.setdefault(roi_head_type, self.roi_head_faceKp)
                    if 'task_neck' in self.neck_names:
                        self.neck_faceKp = build_neck(neck)
                        self.neckDict.setdefault(roi_head_type, self.neck_faceKp)
                    if self.attentionType == 'GA2':
                        self.attention_faceKp = GeneralizedAttention(in_channels=256, num_heads=8, attention_type='0010',
                                                              convtype=self.convtype)
                        self.attentionDict.setdefault(roi_head_type, self.attention_faceKp)

                elif roi_head_type == 'carDetect':
                    self.roi_head_carDetect = build_head(roi_head[i])
                    self.roi_head_Dict.setdefault(roi_head_type, self.roi_head_carDetect)
                    if 'task_neck' in self.neck_names:
                        self.neck_carDetect = build_neck(neck)
                        self.neckDict.setdefault(roi_head_type, self.neck_carDetect)
                    if self.attentionType == 'GA2':
                        self.attention_carDetect = GeneralizedAttention(in_channels=256, num_heads=8, attention_type='0010',
                                                              convtype=self.convtype)
                        self.attentionDict.setdefault(roi_head_type, self.attention_carDetect)

                elif roi_head_type == 'carplateDetect':
                    self.roi_head_carplateDetect = build_head(roi_head[i])
                    self.roi_head_Dict.setdefault(roi_head_type, self.roi_head_carplateDetect)
                    if 'task_neck' in self.neck_names:
                        self.neck_carplateDetect = build_neck(neck)
                        self.neckDict.setdefault(roi_head_type, self.neck_carplateDetect)
                    if self.attentionType == 'GA2':
                        self.attention_carplateDetect = GeneralizedAttention(in_channels=256, num_heads=8, attention_type='0010',
                                                              convtype=self.convtype)
                        self.attentionDict.setdefault(roi_head_type, self.attention_carplateDetect)

                elif roi_head_type == 'cocoDetect':
                    self.roi_head_cocoDetect = build_head(roi_head[i])
                    self.roi_head_Dict.setdefault(roi_head_type, self.roi_head_cocoDetect)
                    if 'task_neck' in self.neck_names:
                        self.neck_cocoDetect = build_neck(neck)
                        self.neckDict.setdefault(roi_head_type, self.neck_cocoDetect)
                    if self.attentionType == 'GA2':
                        self.attention_cocoDetect = GeneralizedAttention(in_channels=256, num_heads=8, attention_type='0010',
                                                              convtype=self.convtype)
                        self.attentionDict.setdefault(roi_head_type, self.attention_cocoDetect)

                elif roi_head_type == 'cocoMask':
                    self.roi_head_cocoMask = build_head(roi_head[i])
                    self.roi_head_Dict.setdefault(roi_head_type, self.roi_head_cocoMask)
                    if 'task_neck' in self.neck_names:
                        self.neck_cocoMask = build_neck(neck)
                        self.neckDict.setdefault(roi_head_type, self.neck_cocoMask)
                    if self.attentionType == 'GA2':
                        self.attention_cocoMask = GeneralizedAttention(in_channels=256, num_heads=8, attention_type='0010',
                                                              convtype=self.convtype)
                        self.attentionDict.setdefault(roi_head_type, self.attention_cocoMask)



        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fedbl = 1
        self.lastloss = 0
        self.lastfedbl = 1

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img, targetName, adaptive_w_dict):
        """Directly extract features from the backbone+neck
        """
        if self.attentionType == 'GA':
            x = self.backbone(img)
            x1 = self.backbone_neck(x)
            x2 = self.neckDict[targetName](x)
            x = self.attention_backbone(x1, x2)
        elif self.attentionType == 'GA2':
            x = self.backbone(img)
            x1 = self.backbone_neck(x)
            x2 = self.neckDict[targetName](x)
            x = self.attentionDict[targetName](x1, x2)
        elif self.attentionType == 'None':
            x = self.backbone(img)
            x = self.backbone_neck(x)
        else:
            assert False
        # x = self.backbone(img)
        # x_n = []
        # if 'backbone_neck' in self.neck_names:
        #     x1 = self.backbone_neck(x)
        #     if hasattr(self, 'attention_backbone'):
        #         x1 = self.attention_backbone(x1)
        #     x_n.append(x1)
        # if 'task_neck' in self.neck_names:
        #     x2 = self.neckDict[targetName](x)
        #     if hasattr(self, 'attentionDict'):
        #         x2 = self.attentionDict[targetName](x2)
        #     x_n.append(x2)
        # x = [[] for i in range (len(x_n[0]))]
        # for i in range(len(x_n[0])):
        #     for j in range(len(x_n)):
        #         if x[i] == []:
        #             x[i] = x_n[j][i] * (1-adaptive_w_dict[targetName])
        #         else:
        #             x[i] = x[i] + x_n[j][i] * adaptive_w_dict[targetName]
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_keypoints=None,
                      gt_visibles=None,
                      work_dir=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if work_dir is not None and os.path.exists(work_dir + '/fedbl.txt'):
            with open(work_dir + '/fedbl.txt', mode='r') as f:
                lines = f.readlines()
                fedbls = lines[-1].strip('\n').split(':')
                lastlosslines = lines[-2].strip('\n').split(':')
                assert 'bl_w' == fedbls[0]
                assert 'loss' == lastlosslines[0]
                self.lastloss = float(lastlosslines[1]) / self.lastfedbl
                if self.fedbl != float(fedbls[1]):
                    self.lastfedbl = self.fedbl
                    self.fedbl= float(fedbls[1])


        targetName = img_metas[0]["filename"].split("/")[-2].split('Imgs')[0]
        x = self.extract_feat(img, targetName, None)
        losses = dict()
        # # 区分人脸标签和姿态标签
        # detect_x, detect_img_metas, detect_gt_bboxes, detect_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'detectImgs', 0)
        # facekp_x, facekp_img_metas, facekp_gt_bboxes, facekp_gt_keypoints = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceKpImgs', 0)
        # faceDetect_x, faceDetect_img_metas, faceDetect_gt_bboxes, faceDetect_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceDetectImgs', 0)
        # faceGender_x, faceGender_img_metas, faceGender_gt_bboxes, faceGender_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceGenderImgs', 0)
        # carplateDetect_x, carplateDetect_img_metas, carplateDetect_gt_bboxes, carplateDetect_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'carplateDetectImgs', 0)
        # x_dict = {'detect':detect_x, 'faceDetect':faceDetect_x, 'faceGender':faceGender_x, 'faceKp':facekp_x, 'carplateDetect':carplateDetect_x}
        # img_metas_dict = {'detect':detect_img_metas, 'faceDetect':faceDetect_img_metas, 'faceGender':faceGender_img_metas, 'faceKp':facekp_img_metas, 'carplateDetect':carplateDetect_img_metas}
        # gt_bboxes_dict = {'detect':detect_gt_bboxes, 'faceDetect':faceDetect_gt_bboxes, 'faceGender':faceGender_gt_bboxes, 'faceKp':facekp_gt_bboxes, 'carplateDetect':carplateDetect_gt_bboxes}
        # gt_labels_dict = {'detect':detect_gt_labels, 'faceDetect':faceDetect_gt_labels, 'faceGender':faceGender_gt_labels, 'faceKp':facekp_gt_keypoints, 'carplateDetect':carplateDetect_gt_labels}

        #检测类任务：社区目标、人脸、车牌
        if targetName == 'cocoDetect' and gt_masks is not None:
            targetName = 'cocoMask'
        if targetName in ['detect', 'faceDetect', 'carplateDetect', 'carDetect', 'carplateDetect', 'cocoDetect', 'cocoMask']:
            # RPN forward and loss
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head_Dict[targetName].forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            for name, value in rpn_losses.items():
                losses['{}_{}'.format(targetName, name)] = (
                    value)
            # ROI forward and loss
            roi_losses = self.roi_head_Dict[targetName].forward_train(x, img_metas, proposal_list,
                                                                   gt_bboxes, gt_labels,
                                                                   gt_bboxes_ignore, gt_masks,
                                                                   **kwargs)
            for name, value in roi_losses.items():
                losses['{}_{}'.format(targetName, name)] = (
                    value)

        # 人脸关键点检测
        elif targetName == 'faceKp':
            facekp_roi_losses = self.roi_head_faceKp.kp_forward_train(x, img_metas, gt_keypoints,
                                                           gt_bboxes)
            for name, value in facekp_roi_losses.items():
                losses['{}_{}'.format(targetName, name)] = (
                    value)

        # 人脸性别识别
        elif targetName == 'faceGender':
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head_faceDetect.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            gender_roi_losses = self.roi_head_faceGender.cls_forward_train(x, img_metas, proposal_list,
                                                                       gt_bboxes, gt_labels,
                                                                       gt_bboxes_ignore)
            for name, value in gender_roi_losses.items():
                losses['{}_{}'.format(targetName, name)] = (
                    value)
        if self.lastloss > 0:
            curloss = 0
            for name, value in losses.items():
                if 'loss' in name:
                    if isinstance(value, list):
                        for i in range(len(value)):
                            curloss = curloss + value[i].data.cpu().numpy()
                    else:
                        curloss = curloss + value.data.cpu().numpy()
            curloss = max(0.00001, curloss)
            self.preloss = (curloss/self.lastloss)**0.3 * self.lastloss
            w = self.preloss / curloss * self.fedbl
            # w = min(math.exp(min((1/curloss/20), 10)), 15) / min(math.exp(min((1/self.lastloss/20), 10)), 15) * self.fedbl
            curloss2 = 0
            for name, value in losses.items():
                if 'loss' in name:
                    if isinstance(value, list):
                        for i in range(len(value)):
                            losses[name][i] *= w
                            curloss2 += losses[name][i]
                    else:
                        losses[name] *= w
                        curloss2 += losses[name]
            # print(targetName+' : '+str(w)+' : '+str(curloss)+' : '+str(self.lastloss)+' : '+str(curloss2) + ':' + str(self.preloss))
            # self.lastloss = self.preloss
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def IoF(self, bbox1, bbox2):
        inter_rect_xmin = max(bbox1[0], bbox2[0])
        inter_rect_ymin = max(bbox1[1], bbox2[1])
        inter_rect_xmax = min(bbox1[2], bbox2[2])
        inter_rect_ymax = min(bbox1[3], bbox2[3])
        inter_area = max(0, (inter_rect_xmax - inter_rect_xmin)) * max(0, (inter_rect_ymax - inter_rect_ymin))
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        iou = inter_area / area1
        return iou

    def mathFaceHat(self, bbox1, bbox2):
        faceHeight = bbox2[3] - bbox2[1]
        iouLine = 0
        if bbox1[3] - bbox2[1] > -faceHeight and bbox1[3] - bbox2[1] < faceHeight:
            iouLine = (min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))/(max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0]))
        return iouLine


    def simple_test(self, img, img_metas, proposals=None, rescale=False, points=None, pre_bboxes=None, adaptive_w_dict=None):
        """Test without augmentation."""
        targetName = img_metas[0]["filename"].split("/")[-2].split('Imgs')[0]
        x = self.extract_feat(img, targetName, adaptive_w_dict)
        # 根据pre_bboxes进行检测
        if pre_bboxes is not None:
            pre_bboxes = torch.from_numpy(np.array(pre_bboxes))
            pre_bboxes = pre_bboxes.to(img.device)
            proposal_list = pre_bboxes.type(torch.float)
        elif pre_bboxes is None:
            proposal_list = self.rpn_head_Dict[targetName].simple_test_rpn(x, img_metas)

        if targetName == 'faceKp':
            face_kps = self.roi_head_faceKp.simple_hy_kp_test(
                x, proposal_list, img_metas, rescale=rescale)
            face_kps = face_kps.cpu().numpy()
            return face_kps
        elif targetName == 'faceGender':
            # 人脸姿态分类
            faceGender_labels = self.roi_head_faceGender.simple_cls_test(x, proposal_list, img_metas)
            faceGender_labels = faceGender_labels.cpu().numpy()
            return faceGender_labels
        else:
            return self.roi_head_Dict[targetName].simple_test(
                x, proposal_list, img_metas, rescale=rescale)
            face_kps = []
            det_bboxes, det_labels = self.roi_head_Dict[targetName].simple_test(
                x, proposal_list, img_metas, rescale=rescale, ifdet=True)
            if len(det_bboxes) > 0:
                face_kps = self.roi_head_faceKp.simple_hy_kp_test(
                    x, det_bboxes.clone(), img_metas, rescale=rescale)
                face_kps = face_kps.cpu().numpy()
            det_bboxes = det_bboxes.cpu().numpy()
            det_labels = det_labels.cpu().numpy()
            return det_bboxes, det_labels, face_kps

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
