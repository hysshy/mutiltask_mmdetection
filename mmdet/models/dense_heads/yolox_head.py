# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, build_bbox_coder)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class YOLOXHead(BaseDenseHead, BBoxTestMixin):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 body_classes=0,
                 clouseStyle_classes=0,
                 clouseColor_classes=0,
                 feat_channels=256,
                 with_faceKp = False,
                 with_facemohu = False,
                 facezita_num_classes = 0,
                 stacked_convs=2,
                 strides=[8, 16, 32],
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):

        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.body_classes = body_classes
        self.clouseStyle_classes = clouseStyle_classes
        self.clouseColor_classes = clouseColor_classes
        self.with_faceKp = with_faceKp
        self.with_facemohu = with_facemohu
        self.facezita_num_classes = facezita_num_classes
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)
        if self.with_faceKp:
            self.loss_facekp = build_loss(dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
            bbox_coder = dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1])
            self.bbox_coder = build_bbox_coder(bbox_coder)
        if self.with_facemohu:
            self.loss_facemohu = build_loss(dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        self.use_l1 = False  # This flag will be modified by hooks.
        self.loss_l1 = build_loss(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        if self.with_faceKp:
            #关键点预测
            self.multi_level_facekp_convs = nn.ModuleList()
            self.multi_level_conv_facekp = nn.ModuleList()
        if self.facezita_num_classes:
            #人脸姿态分类
            self.multi_level_facezitai_convs = nn.ModuleList()
            self.multi_level_conv_facezitai = nn.ModuleList()
        if self.with_facemohu:
            #人脸模糊度评估
            self.multi_level_facemohu_convs = nn.ModuleList()
            self.multi_level_conv_facemohu = nn.ModuleList()
        if self.body_classes > 0:
            #人体部位属性检测
            self.multi_level_bodydetect_cls_convs = nn.ModuleList()
            self.multi_level_bodydetect_reg_convs = nn.ModuleList()
            self.multi_level_bodydetect_conv_cls = nn.ModuleList()
            self.multi_level_bodydetect_conv_reg = nn.ModuleList()
            self.multi_level_bodydetect_conv_obj = nn.ModuleList()
        if self.clouseStyle_classes > 0:
            #衣服风格检测
            self.multi_level_clouseStyle_convs = nn.ModuleList()
            self.multi_level_conv_clouseStyle = nn.ModuleList()
        if self.clouseColor_classes > 0:
            #衣服颜色检测
            self.multi_level_clouseColor_convs = nn.ModuleList()
            self.multi_level_conv_clouseColor = nn.ModuleList()

        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

            if self.with_faceKp:
                self.multi_level_facekp_convs.append(self._build_stacked_convs())
                conv_facekp = nn.Conv2d(self.feat_channels, 10, 1)
                self.multi_level_conv_facekp.append(conv_facekp)
            if self.facezita_num_classes:
                self.multi_level_facezitai_convs.append(self._build_stacked_convs())
                conv_facezitai = nn.Conv2d(self.feat_channels, self.facezita_num_classes, 1)
                self.multi_level_conv_facezitai.append(conv_facezitai)
            if self.with_facemohu:
                self.multi_level_facemohu_convs.append(self._build_stacked_convs())
                conv_facemohu = nn.Conv2d(self.feat_channels, 1, 1)
                self.multi_level_conv_facemohu.append(conv_facemohu)
            if self.body_classes > 0:
                self.multi_level_bodydetect_cls_convs.append(self._build_stacked_convs())
                self.multi_level_bodydetect_reg_convs.append(self._build_stacked_convs())
                conv_cls = nn.Conv2d(self.feat_channels, self.body_classes, 1)
                conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
                conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
                self.multi_level_bodydetect_conv_cls.append(conv_cls)
                self.multi_level_bodydetect_conv_reg.append(conv_reg)
                self.multi_level_bodydetect_conv_obj.append(conv_obj)
            if self.clouseStyle_classes > 0:
                self.multi_level_clouseStyle_convs.append(self._build_stacked_convs())
                conv_clouseStyle = nn.Conv2d(self.feat_channels, self.clouseStyle_classes, 1)
                self.multi_level_conv_clouseStyle.append(conv_clouseStyle)
            if self.clouseColor_classes > 0:
                self.multi_level_clouseColor_convs.append(self._build_stacked_convs())
                conv_clouseColor = nn.Conv2d(self.feat_channels, self.clouseColor_classes, 1)
                self.multi_level_conv_clouseColor.append(conv_clouseColor)

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def init_weights(self):
        super(YOLOXHead, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)
        if self.with_faceKp:
            for conv_facekp, in zip(self.multi_level_conv_facekp,):
                conv_facekp.bias.data.fill_(bias_init)
        if self.facezita_num_classes:
            for conv_facezitai, in zip(self.multi_level_conv_facezitai,):
                conv_facezitai.bias.data.fill_(bias_init)
        if self.with_facemohu:
            for conv_facemohu, in zip(self.multi_level_conv_facemohu,):
                conv_facemohu.bias.data.fill_(bias_init)
        if self.body_classes > 0:
            for conv_cls, conv_obj in zip(self.multi_level_bodydetect_conv_cls,
                                          self.multi_level_bodydetect_conv_obj):
                conv_cls.bias.data.fill_(bias_init)
                conv_obj.bias.data.fill_(bias_init)
        if self.clouseStyle_classes > 0:
            for conv_clouseStyle, in zip(self.multi_level_conv_clouseStyle,):
                conv_clouseStyle.bias.data.fill_(bias_init)
        if self.clouseColor_classes > 0:
            for conv_clouseColor, in zip(self.multi_level_conv_clouseColor,):
                conv_clouseColor.bias.data.fill_(bias_init)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj):
        """Forward feature of a single scale level."""

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, objectness

    def forward_addbranch_single(self, x, branch_convs, convs_branch):
        """Forward feature of a single scale level."""
        feat = branch_convs(x)
        pred = convs_branch(feat)
        return pred,

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        return multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj)

    def forward_facekp(self, feats):
        return multi_apply(self.forward_addbranch_single, feats,
                           self.multi_level_facekp_convs,
                           self.multi_level_conv_facekp
                           )

    def forward_facemohu(self, feats):
        return multi_apply(self.forward_addbranch_single, feats,
                           self.multi_level_facemohu_convs,
                           self.multi_level_conv_facemohu
                           )

    def forward_facezitai(self, feats):
        return multi_apply(self.forward_single, feats,
                           self.multi_level_facezitai_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_facezitai,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj
                           )

    def forward_facezitai_intest(self, feats):
        return multi_apply(self.forward_addbranch_single, feats,
                           self.multi_level_facezitai_convs,
                           self.multi_level_conv_facezitai,
                           )

    def forward_clouseStyle_intest(self, feats):
        return multi_apply(self.forward_addbranch_single, feats,
                           self.multi_level_clouseStyle_convs,
                           self.multi_level_conv_clouseStyle,
                           )

    def forward_clouseColor_intest(self, feats):
        return multi_apply(self.forward_addbranch_single, feats,
                           self.multi_level_clouseColor_convs,
                           self.multi_level_conv_clouseColor,
                           )

    def forward_bodydetect(self, feats):
        return multi_apply(self.forward_single, feats,
                           self.multi_level_bodydetect_cls_convs,
                           self.multi_level_bodydetect_reg_convs,
                           self.multi_level_bodydetect_conv_cls,
                           self.multi_level_bodydetect_conv_reg,
                           self.multi_level_bodydetect_conv_obj)

    def forward_clouseStyle(self, feats):
        return multi_apply(self.forward_single, feats,
                           self.multi_level_clouseStyle_convs,
                           self.multi_level_bodydetect_reg_convs,
                           self.multi_level_conv_clouseStyle,
                           self.multi_level_bodydetect_conv_reg,
                           self.multi_level_bodydetect_conv_obj
                           )

    def forward_clouseColor(self, feats):
        return multi_apply(self.forward_single, feats,
                           self.multi_level_clouseColor_convs,
                           self.multi_level_bodydetect_reg_convs,
                           self.multi_level_conv_clouseColor,
                           self.multi_level_bodydetect_conv_reg,
                           self.multi_level_bodydetect_conv_obj
                           )


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def get_kps(self, facekp_preds,
                kp_rois,
                valid_mask,
                keep,
                face_index,
                img_metas=None,
                rescale=None
                ):
        num_imgs = len(img_metas)
        # flatten cls_scores, bbox_preds and objectness
        flatten_facekp_preds = [
            facekp_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 10)
            for facekp_pred in facekp_preds
        ]
        flatten_facekp_preds = torch.cat(flatten_facekp_preds, dim=1)
        result_list = []
        for img_id in range(len(img_metas)):
            facekp_preds = flatten_facekp_preds[img_id]
            facekp_preds = facekp_preds[valid_mask][keep][face_index]
            kp = self.bbox_coder.kp_decode(
                kp_rois[:, 1:], facekp_preds, max_shape=img_metas[0]['img_shape'])
            kp = kp.view(-1, 5, 2)
            if rescale:
                scale_factor = img_metas[0]["scale_factor"]
                scale_factor = torch.tensor(scale_factor).type_as(kp)
                kp /= scale_factor[0:2]
            result_list.append(kp)
        return result_list

    @force_fp32(apply_to=('cls_scores'))
    def get_cls(self, scores,
                      valid_mask,
                      keep,
                      face_index,
                      img_metas=None,
                      ):
        num_imgs = len(img_metas)
        # flatten cls_scores, bbox_preds and objectness
        flatten_scores = [
            score.permute(0, 2, 3, 1).reshape(num_imgs, -1, score.shape[1])
            for score in scores
        ]
        flatten_scores = torch.cat(flatten_scores, dim=1)
        result_list = []
        for img_id in range(len(img_metas)):
            scores = flatten_scores[img_id]
            scores = scores[valid_mask][keep][face_index]
            max_scores, labels = torch.max(scores, 1)
            result_list.append(labels)
        return result_list

    @force_fp32(apply_to=('facemohu_scores'))
    def get_facemohu(self, facemohu_scores,
                      valid_mask,
                      keep,
                      face_index,
                      img_metas=None,
                      ):
        num_imgs = len(img_metas)
        flatten_facemohu_scores = [
            facemohu_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, facemohu_score.shape[1])
            for facemohu_score in facemohu_scores
        ]
        flatten_facemohu_scores = torch.cat(flatten_facemohu_scores, dim=1)
        result_list = []
        for img_id in range(len(img_metas)):
            facemohu_scores = flatten_facemohu_scores[img_id]
            facemohu_scores = facemohu_scores[valid_mask][keep][face_index]
            result_list.append(facemohu_scores)
        return result_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   getKeep=False):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array(
            [img_meta['scale_factor'] for img_meta in img_metas])

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  cls_score.shape[1])
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        if rescale:
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]

            result_list.append(
                self._bboxes_nms(cls_scores, bboxes, score_factor, cfg, getKeep))

        return result_list

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def _kps_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg, getkeep=False):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if getkeep:
            if labels.numel() == 0:
                return bboxes, labels, None, None
            else:
                dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
                return dets, labels[keep], valid_mask, keep
        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             lossname=None
             ):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, cls_pred.shape[1])
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, flatten_cls_preds.shape[-1])[pos_masks],
            cls_targets) / num_total_samples

        if lossname is None:
            loss_dict = dict(
                loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)
        else:
            loss_dict = {lossname+'_loss_cls':loss_cls, lossname+'_loss_bbox':loss_bbox, lossname+'_loss_obj':loss_obj}

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                l1_targets) / num_total_samples
            if lossname is None:
                loss_dict.update(loss_l1=loss_l1)
            else:
                loss_dict.update({lossname+'_loss_l1':loss_l1})
        return loss_dict


    @force_fp32(apply_to=('cls_scores'))
    def cls_loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             loss_name,
             gt_bboxes_ignore=None):

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, cls_pred.shape[1])
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)

        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, flatten_cls_preds.shape[-1])[pos_masks],
            cls_targets) / num_total_samples

        loss_dict = {loss_name:loss_cls}
        return loss_dict

    #关键点损失计算
    @force_fp32(apply_to=('kp_pred'))
    def kp_loss(self,
                cls_scores,
                bbox_preds,
                objectnesses,
                facekp_preds,
                gt_bboxes,
                gt_labels,
                gt_keypoints,
                img_metas,
                gt_bboxes_ignore=None):
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        flatten_facekp_preds = [
            facekp_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 10)
            for facekp_pred in facekp_preds
        ]
        flatten_facekp_preds = torch.cat(flatten_facekp_preds, dim=1)
        flatten_priors = [flatten_priors[pos_masks[i]] for i in range(num_imgs)]
        kp_targets, kp_weights, pos_bboxes_targets_index= self.get_kp_targets(flatten_priors, bbox_targets, gt_keypoints, img_metas)

        pos_masks = torch.cat(pos_masks, 0)
        facekp_preds = flatten_facekp_preds.view(-1, 10)[pos_masks][pos_bboxes_targets_index]
        loss_kp = self.loss_facekp(  # 计算bbox损失
            facekp_preds,
            kp_targets,
            kp_weights,
            avg_factor=kp_targets.size(0),
            reduction_override=None) * 0.005
        loss_dict = dict(loss_kp=loss_kp)
        return loss_dict

    #人脸模糊度损失计算
    @force_fp32(apply_to=('kp_pred'))
    def facemohu_loss(self,
                cls_scores,
                bbox_preds,
                objectnesses,
                facemohu_preds,
                gt_bboxes,
                gt_labels,
                gt_mohus,
                img_metas,
                gt_bboxes_ignore=None):
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        flatten_facemohu_preds = [
            facemohu_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for facemohu_pred in facemohu_preds
        ]
        flatten_facemohu_preds = torch.cat(flatten_facemohu_preds, dim=1)
        facemohu_targets, facemohu_weights = self.get_facemohu_targets(bbox_targets, gt_bboxes, gt_mohus)
        pos_masks = torch.cat(pos_masks, 0)
        facemohu_preds = flatten_facemohu_preds.view(-1, 1)[pos_masks]
        loss_facemohu = self.loss_facemohu(  # 计算bbox损失
            facemohu_preds,
            facemohu_targets,
            facemohu_weights,
            avg_factor=facemohu_preds.size(0),
            reduction_override=None) * 0.5
        loss_dict = dict(loss_facemohu=loss_facemohu)
        return loss_dict

    #获得人脸关键点的训练目标
    def get_kp_targets(self, flatten_priors, bboxes_targets,
                   gt_keypoints = None,
                   img_metas = None):
        pos_gt_keypoints =[]
        pos_bboxes_targets = []
        pos_bboxes_targets_index = []
        for img_i in range(len(bboxes_targets)):
            bboxes = bboxes_targets[img_i]
            keypoints = gt_keypoints[img_i]
            pos_gt_keypoint = []
            pos_bboxes_target_index = []
            for bbox_i in range(len(bboxes)):
                bbox = bboxes[bbox_i]
                bbox_keypoint = []
                for kp_i in range(len(keypoints)):
                    keypoint = keypoints[kp_i]
                    for i in range(len(keypoint)):
                        if keypoint[i][0] >= (bbox[0]-1) and keypoint[i][0] <= (bbox[2]+1) and keypoint[i][1] >= (bbox[1]-1) and keypoint[i][1] <= (bbox[3]+1):
                            kp_in_bbox = True
                        else:
                            kp_in_bbox = False
                            break
                    if kp_in_bbox:
                        bbox_keypoint.append(keypoint)
                assert len(bbox_keypoint) > 0
                if len(bbox_keypoint) == 1:
                    pos_gt_keypoint.append(bbox_keypoint[0].view(1, 10))
                    pos_bboxes_target_index.append(True)
                else:
                    pos_bboxes_target_index.append(False)

            pos_bboxes_target_index = torch.from_numpy(np.array(pos_bboxes_target_index)).to(bboxes.device)
            bboxes = bboxes[pos_bboxes_target_index]
            if len(pos_gt_keypoint) > 0:
                pos_gt_keypoint = torch.cat(pos_gt_keypoint)
            assert len(bboxes) == len(pos_gt_keypoint)
            pos_gt_keypoints.append(pos_gt_keypoint)
            pos_bboxes_targets.append(bboxes)
            pos_bboxes_targets_index.append(pos_bboxes_target_index)
        for i in range(len(pos_bboxes_targets)):
            for j in range(len(pos_bboxes_targets[i])):
                for k in range(5):
                    assert pos_gt_keypoints[i][j][k*2] >= pos_bboxes_targets[i][j][0] and pos_gt_keypoints[i][j][k*2] <= pos_bboxes_targets[i][j][2]
                    assert pos_gt_keypoints[i][j][k*2+1] >= pos_bboxes_targets[i][j][1] and pos_gt_keypoints[i][j][k*2+1] <= pos_bboxes_targets[i][j][3]

        kp_targets, kp_weights = multi_apply(
            self._get_kp_target_single,
            pos_bboxes_targets,
            pos_gt_keypoints)
        kp_targets = torch.cat(kp_targets, 0)
        kp_weights = torch.cat(kp_weights, 0)
        pos_bboxes_targets_index = torch.cat(pos_bboxes_targets_index, 0)
        return kp_targets, kp_weights, pos_bboxes_targets_index

    #获得人脸模糊度的训练目标
    def get_facemohu_targets(self, bboxes_targets, gt_bboxes, gt_mohus):
        pos_gt_mohus =[]
        for img_i in range(len(bboxes_targets)):
            bboxes_target = bboxes_targets[img_i]
            gt_bbox = gt_bboxes[img_i]
            gt_mohu = gt_mohus[img_i]
            pos_gt_mohu = []
            for target_i in range(len(bboxes_target)):
                for bbox_i in range(len(gt_bbox)):
                    if gt_bbox[bbox_i].equal(bboxes_target[target_i]):
                        pos_gt_mohu.append(gt_mohu[bbox_i].view(-1, 1))
                        break
            assert len(pos_gt_mohu) == len(bboxes_target)
            pos_gt_mohu = torch.cat(pos_gt_mohu)
            pos_gt_mohus.append(pos_gt_mohu)
        facemohu_targets = torch.cat(pos_gt_mohus, 0)
        facemohu_weights = facemohu_targets.new_ones(facemohu_targets.size(0), 1)
        return facemohu_targets, facemohu_weights

    # 关键点训练目标框
    def _get_kp_target_single(
            self,
            gt_bboxes,
            pos_gt_keypoints):
        num_samples = gt_bboxes.size(0)
        kp_targets = gt_bboxes.new_zeros(num_samples, 10)
        kp_weights = gt_bboxes.new_zeros(num_samples, 10)
        if num_samples > 0:
            kpos_inds = torch.ne(pos_gt_keypoints[:, 0], -1)
            kp_pos_bboxes = gt_bboxes[kpos_inds]
            kpos_gt_keypoints = pos_gt_keypoints[kpos_inds]
            pos_kp_targets = self.bbox_coder.kp_encode(kp_pos_bboxes, kpos_gt_keypoints)
            kp_targets[:pos_kp_targets.size(0), :] = pos_kp_targets
            kp_weights[:pos_kp_targets.size(0), :] = 1
        return kp_targets, kp_weights


    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, cls_preds.shape[1]))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               cls_preds.shape[1]) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, num_pos_per_img)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
