# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from . import labelstransform

@DETECTORS.register_module()
class SingleStageDetector_SPJGH(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector_SPJGH, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints,
                      gt_visibles,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector_SPJGH, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        labelstransform.asserSameFile(img_metas)
        # 目标检测
        if img_metas[0]['filename'].split("/")[-2] == 'detectImgs':
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)
        # 人脸关键点检测
        if img_metas[0]['filename'].split("/")[-2] == 'faceKpImgs':
            assert len(gt_bboxes) == len(gt_keypoints)
            losses = self.bbox_head.kp_forward_train(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_bboxes_ignore)

        # 人脸姿态分类
        if img_metas[0]['filename'].split("/")[-2] == 'faceZitaiImgs':
            # for i in range(len(gt_labels)):
            #     gt_labels[i] = gt_labels[i] - 9
            losses = self.bbox_head.facezitai_forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)

        # 人脸模糊度评估
        if img_metas[0]['filename'].split("/")[-2] == 'faceMohuImgs':
            gt_mohus = []
            for i in range(len(img_metas)):
                gt_mohu = []
                for gt_visible in gt_visibles[i]:
                    gt_mohu.append(gt_visible[0].view(-1))
                gt_mohu = torch.cat(gt_mohu)
                gt_mohus.append(gt_mohu)
            losses = self.bbox_head.facemohu_forward_train(x, img_metas, gt_bboxes, gt_labels, gt_mohus, gt_bboxes_ignore)

        # 人体部位检测
        if img_metas[0]['filename'].split("/")[-2] == 'bodyDetect':
            for i in range(len(gt_labels)):
                gt_labels[i] = gt_labels[i] - 20
            losses = self.bbox_head.facezitai_forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False, points=None):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        det_bboxes, det_labels, face_bboxes, face_kps, face_zitais, face_mohus = [],[],[],[],[],[]
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale, getKeep=False)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
        det_bboxes, det_labels, valid_mask, keep = results_list[0]
        face_bboxes, face_labels, face_index = labelstransform.simple_test_findTarget(det_bboxes, det_labels, points, 0, 1)
        #人脸属性识别
        if len(face_bboxes) > 0:
            #人脸关键点识别
            face_kps = self.bbox_head.simple_test_kps(
                feat, face_bboxes.clone(), valid_mask, keep, face_index, img_metas, rescale=rescale)
            face_kps = face_kps[0].cpu().numpy()
            # 人脸姿态分类
            face_zitais = self.bbox_head.simple_test_facezitai(feat, valid_mask, keep, face_index, img_metas)[0].cpu().numpy()
            # 人脸模糊度评估
            face_mohus =  self.bbox_head.simple_test_facemohu(feat, valid_mask, keep, face_index, img_metas)[0].cpu().numpy()
            face_bboxes = face_bboxes.cpu().numpy()
        if len(det_bboxes) > 0:
            det_bboxes = det_bboxes.cpu().numpy()
            det_labels = det_labels.cpu().numpy()
        # return results_list[0]
        return det_bboxes, det_labels, face_bboxes, face_kps, face_zitais, face_mohus

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
