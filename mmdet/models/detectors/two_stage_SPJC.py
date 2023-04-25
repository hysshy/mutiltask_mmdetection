import torch
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from . import labelstransform
import warnings



@DETECTORS.register_module()
class TwoStageDetector_SPJC(BaseDetector):
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
        super(TwoStageDetector_SPJC, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck_detect = build_neck(neck)
            self.neck_faceDetect = build_neck(neck)
            self.neck_faceGender = build_neck(neck)
            self.neck_faceKp = build_neck(neck)
            self.neck_carplateDetect = build_neck(neck)
            self.neckDict = {'detectImgs':self.neck_detect, 'faceDetectImgs':self.neck_faceDetect, 'faceGenderImgs':self.neck_faceGender, 'faceKpimages':self.neck_faceKp, 'carplateImgs':self.neck_carplateDetect}

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            for i in range(len(rpn_head)):
                rpn_head[i].update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head_detect = build_head(rpn_head[0])
            self.rpn_head_faceDetect = build_head(rpn_head[1])
            self.rpn_head_carplateDetect = build_head(rpn_head[2])

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            for i in range(len(roi_head)):
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg.rcnn)
            self.roi_head_detect = build_head(roi_head[0])
            self.roi_head_faceDetect = build_head(roi_head[1])
            self.roi_head_faceKp = build_head(roi_head[2])
            self.roi_head_faceGender = build_head(roi_head[3])
            self.roi_head_carplateDetect = build_head(roi_head[4])


        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        fileName = img_metas[0]["filename"]
        targetName = fileName.split("/")[-2]
        x = self.neckDict[targetName](x)
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


        x = self.extract_feat(img, img_metas)
        losses = dict()
        # 区分人脸标签和姿态标签
        detect_x, detect_img_metas, detect_gt_bboxes, detect_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'detectImgs', 0)
        facekp_x, facekp_img_metas, facekp_gt_bboxes, facekp_gt_keypoints = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceKpimages', 0)
        faceDetect_x, faceDetect_img_metas, faceDetect_gt_bboxes, faceDetect_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceDetectImgs', 7)
        faceGender_x, faceGender_img_metas, faceGender_gt_bboxes, faceGender_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceGenderImgs', 9)
        carplate_x, carplate_img_metas, carplate_gt_bboxes, carplate_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'carplateImgs', 11)

        # 社区目标检测
        if len(detect_img_metas) >0:
            assert len(detect_img_metas) == 2

            # RPN forward and loss
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head_detect.forward_train(
                detect_x,
                detect_img_metas,
                detect_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            detect_roi_losses = self.roi_head_detect.forward_train(detect_x, detect_img_metas, proposal_list,
                                                     detect_gt_bboxes, detect_gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            for name, value in detect_roi_losses.items():
                losses['{}'.format(name)] = (
                        value)

        # 人脸检测
        if len(faceDetect_img_metas) >0:
            assert len(faceDetect_img_metas) == 2

            # RPN forward and loss
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head_faceDetect.forward_train(
                faceDetect_x,
                faceDetect_img_metas,
                faceDetect_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            faceDetect_roi_losses = self.roi_head_faceDetect.forward_train(faceDetect_x, faceDetect_img_metas, proposal_list,
                                                     faceDetect_gt_bboxes, faceDetect_gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            for name, value in faceDetect_roi_losses.items():
                losses['faceDetect_{}'.format(name)] = (
                        value)

        # 人脸关键点检测
        if len(facekp_gt_bboxes) > 0:
            assert len(facekp_gt_bboxes) == 2
            facekp_roi_losses = self.roi_head_faceKp.kp_forward_train(facekp_x, facekp_img_metas, facekp_gt_keypoints,
                                                           facekp_gt_bboxes)
            for name, value in facekp_roi_losses.items():
                losses['facekp_{}'.format(name)] = (
                    value)

        # 性别识别
        if len(faceGender_img_metas) > 0:
            assert len(faceGender_img_metas) == 2
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head_faceDetect.forward_train(
                faceGender_x,
                faceGender_img_metas,
                faceGender_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            gender_roi_losses = self.roi_head_faceGender.cls_forward_train(faceGender_x, faceGender_img_metas, proposal_list,
                                                                       faceGender_gt_bboxes, faceGender_gt_labels,
                                                                       gt_bboxes_ignore)
            for name, value in gender_roi_losses.items():
                losses['gender_{}'.format(name)] = (
                    value)

        if len(carplate_img_metas) > 0:
            assert len(carplate_img_metas) == 2

            # RPN forward and loss
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head_carplateDetect.forward_train(
                carplate_x,
                carplate_img_metas,
                carplate_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            carplate_roi_losses = self.roi_head_carplateDetect.forward_train(carplate_x, carplate_img_metas, proposal_list,
                                                     carplate_gt_bboxes, carplate_gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            for name, value in carplate_roi_losses.items():
                losses['carplate_{}'.format(name)] = (
                        value)

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


    def simple_test(self, img, img_metas, proposals=None, rescale=False, points=None):
        """Test without augmentation."""
        person_bboxes, person_labels, car_bboxes, car_labels, pets_bboxes, pets_labels, face_bboxes, face_labels, zitai_labels, face_kps, carplates = \
            [],[],[],[],[],[],[],[],[],[],[]
        x = self.extract_feat(img, img_metas)
        if proposals is None:
            proposal_list = self.rpn_head_detect.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        if proposal_list[0].size(0) == 0:
            return [],[],[]
        return self.roi_head_detect.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        det_bboxes, det_labels = self.roi_head_detect.simple_test(
            x, proposal_list, img_metas, rescale=rescale, ifdet=True)
        # det_bboxes, det_labels = labelstransform.simple_test_unRepeated(det_bboxes, det_labels)
        person_bboxes, person_labels, face_bboxes, face_labels, car_bboxes, car_labels, pets_bboxes, pets_labels = labelstransform.simple_test_splitTarget(det_bboxes, det_labels, points, 0, 1, 2, 3, 4, 6, 7, 8)
        if len(face_bboxes) > 0:
            #人脸姿态分类
            zitai_labels = self.faceGender_roi_head.simple_cls_test(x, face_bboxes.clone(), img_metas)
            print(zitai_labels)
            face_kps = self.faceKp_roi_head.simple_hy_kp_test(
                x, face_bboxes.clone(), img_metas, rescale=rescale)
            face_kps = face_kps.cpu().numpy()
            zitai_labels = zitai_labels.cpu().numpy()
            # 人脸关键点检测
            carplates = self.carplate_roi_head.simple_hy_qt_test(
                x, face_bboxes.clone(), img_metas, rescale=rescale)
            carplates = carplates.cpu().numpy()

        face_bboxes = face_bboxes.cpu().numpy()
        face_labels = face_labels.cpu().numpy()
        person_bboxes =person_bboxes.cpu().numpy()
        person_labels = person_labels.cpu().numpy()
        car_bboxes = car_bboxes.cpu().numpy()
        car_labels = car_labels.cpu().numpy()
        pets_bboxes = pets_bboxes.cpu().numpy()
        pets_labels = pets_labels.cpu().numpy()

        return person_bboxes, person_labels, car_bboxes, car_labels, pets_bboxes, pets_labels, face_bboxes, face_labels, zitai_labels, face_kps, carplates


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
