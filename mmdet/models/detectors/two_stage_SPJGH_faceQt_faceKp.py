import torch
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from . import labelstransform
import warnings

@DETECTORS.register_module()
class TwoStageDetector_SPJGH_FaceQt_FaceKp(BaseDetector):
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
        super(TwoStageDetector_SPJGH_FaceQt_FaceKp, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            for i in range(len(rpn_head_)):
                rpn_head_[i].update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_[0])
            self.bodydetect_rpn_head = build_head(rpn_head_[1])

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            for i in range(len(roi_head)):
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg.rcnn)
            self.detect_roi_head = build_head(roi_head[0])
            self.facezitai_roi_head = build_head(roi_head[1])
            self.facemohu_roi_head = build_head(roi_head[2])
            self.faceKp_roi_head = build_head(roi_head[3])
            self.bodydetect_roi_head = build_head(roi_head[4])
            self.clouse_color_roi_head = build_head(roi_head[5])
            self.clouse_style_roi_head = build_head(roi_head[6])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
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
        print(img_metas[0]['filename'])
        print(img_metas[1]['filename'])
        if img_metas[0]['filename'] == '/home/chase/shy/dataset/spjgh/clouseStyleImgs/refineData_1_1287572342032019456.jpg':
            print('ddd')
        x = self.extract_feat(img)
        losses = dict()
        # 区分人脸标签和姿态标签
        detect_x, detect_img_metas, detect_gt_bboxes, detect_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'detectImgs', 0)
        facezitai_x, facezitai_img_metas, facezitai_gt_bboxes, facezitai_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceZitaiImgs', 0)
        facemohu_x, facemohu_img_metas, facemohu_gt_bboxes, facemohu_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceMohuImgs', 0)
        facekp_x, facekp_img_metas, facekp_gt_bboxes, facekp_gt_keypoints = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'faceKpImgs', 0)
        bodydetect_x, bodydetect_img_metas, bodydetect_gt_bboxes, bodydetect_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'bodyDetectImgs', 0)
        if len(bodydetect_img_metas) > 0:
            bodydetect_x, bodydetect_gt_bboxes, bodydetect_gt_labels, bodydetect_img_metas, bodydetect_index = labelstransform.rectBodyFactorLabels(bodydetect_x, bodydetect_img_metas, bodydetect_gt_bboxes, bodydetect_gt_labels, 0, 1)
        clouse_style_x, clouse_style_img_metas, clouse_style_gt_bboxes, clouse_style_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'clouseStyleImgs', 0)
        if len(clouse_style_img_metas) > 0:
            clouse_style_x, clouse_style_gt_bboxes, clouse_style_gt_labels, clouse_style_img_metas, clouse_style_index =labelstransform.rectBodyFactorLabels(clouse_style_x, clouse_style_img_metas, clouse_style_gt_bboxes, clouse_style_gt_labels, 0, 1)
        clouse_color_x, clouse_color_img_metas, clouse_color_gt_bboxes, clouse_color_gt_labels = labelstransform.findLabelbyFile(x, img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_visibles, 'clouseColorImgs', 0)
        if len(clouse_color_img_metas) > 0:
            clouse_color_x, clouse_color_gt_bboxes, clouse_color_gt_labels, clouse_color_img_metas, clouse_color_index =labelstransform.rectBodyFactorLabels(clouse_color_x, clouse_color_img_metas, clouse_color_gt_bboxes, clouse_color_gt_labels, 0, 1)
        if len(detect_img_metas) >0:
            assert len(detect_img_metas) == 2
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    detect_x,
                    detect_img_metas,
                    detect_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals
            detect_roi_losses = self.detect_roi_head.forward_train(detect_x, detect_img_metas, proposal_list,
                                                     detect_gt_bboxes, detect_gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            for name, value in detect_roi_losses.items():
                if 'loss' in name:
                    losses['{}'.format(name)] = (
                            value * 0.5)
                else:
                    losses['{}'.format(name)] = (
                            value)


        # 姿态分类
        if len(facezitai_img_metas) > 0:
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    facezitai_x,
                    facezitai_img_metas,
                    facezitai_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    ifloss=False)
            zitai_roi_losses = self.facezitai_roi_head.cls_forward_train(facezitai_x, facezitai_img_metas, proposal_list,
                                                                       facezitai_gt_bboxes, facezitai_gt_labels,
                                                                       gt_bboxes_ignore)
            for name, value in zitai_roi_losses.items():
                losses['zitai_{}'.format(name)] = (
                    value)

        # 人脸质量评估
        if len(facemohu_img_metas) > 0:
            assert len(facemohu_img_metas) == 2
            facemohu_roi_losses = self.facemohu_roi_head.qt_forward_train(facemohu_x, facemohu_img_metas, facemohu_gt_labels,
                                                           facemohu_gt_bboxes)

            for name, value in facemohu_roi_losses.items():
                losses['facemohu_{}'.format(name)] = (
                    value)

        # 人脸关键点检测
        if len(facekp_gt_bboxes) > 0:
            assert len(facekp_gt_bboxes) == 2
            facekp_roi_losses = self.faceKp_roi_head.kp_forward_train(facekp_x, facekp_img_metas, facekp_gt_keypoints,
                                                           facekp_gt_bboxes)
            for name, value in facekp_roi_losses.items():
                losses['facekp_{}'.format(name)] = (
                    value)

        # 人体属性检测
        if len(bodydetect_img_metas) > 0:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            bodydetect_rpn_losses, bodydetect_proposal_list = self.bodydetect_rpn_head.forward_train(
                bodydetect_x,
                bodydetect_img_metas,
                bodydetect_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            for name, value in bodydetect_rpn_losses.items():
                losses['bodydetect_{}'.format(name)] = (
                    value)

            bodydetect_roi_losses = self.bodydetect_roi_head.forward_train(bodydetect_x, bodydetect_img_metas, bodydetect_proposal_list,
                                                     bodydetect_gt_bboxes, bodydetect_gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            for name, value in bodydetect_roi_losses.items():
                losses['bodydetect_{}'.format(name)] = (
                    value)

        # 人体属性，衣服颜色分类
        if len(clouse_color_img_metas) > 0:
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.bodydetect_rpn_head.forward_train(
                    clouse_color_x,
                    clouse_color_img_metas,
                    clouse_color_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    ifloss=False)
            clouse_color_roi_losses = self.clouse_color_roi_head.cls_forward_train(clouse_color_x, clouse_color_img_metas, proposal_list,
                                                                       clouse_color_gt_bboxes, clouse_color_gt_labels,
                                                                       gt_bboxes_ignore)
            for name, value in clouse_color_roi_losses.items():
                losses['clouse_color_{}'.format(name)] = (
                    value)

        # 人体属性，衣服风格分类
        if len(clouse_style_img_metas) > 0:
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.bodydetect_rpn_head.forward_train(
                    clouse_style_x,
                    clouse_style_img_metas,
                    clouse_style_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    ifloss=False)
            clouse_style_roi_losses = self.clouse_style_roi_head.cls_forward_train(clouse_style_x, clouse_style_img_metas, proposal_list,
                                                                       clouse_style_gt_bboxes, clouse_style_gt_labels,
                                                                       gt_bboxes_ignore)
            for name, value in clouse_style_roi_losses.items():
                losses['clouse_style_{}'.format(name)] = (
                    value)
            print(losses)
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
        car_bboxes, car_labels, pets_bboxes, pets_labels, withfactorPerson_bboxes, withfactorPerson_labels, \
        nofactorPerson_bboxes, nofactorPerson_labels, upClouse_bboxes_list, upClouse_labels_list, upClouse_color_labels_list,\
        upClouse_style_labels_list, downClouse_bboxes_list, downClouse_labels_list, downClouse_color_labels_list, \
        otherFactor_bboxes_list, otherFactor_labels_list, face_bboxes_list, face_labels_list, faceKps_list, faceZitais_list, \
        faceMohus_list, face_bboxes_list_all, face_labels_list_all, face_other_bboxes_list_all, face_other_labels_list_all,\
        faceKps_list_all, faceZitais_list_all, faceMohus_list_all = [[] for i in range(29)]
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        if proposal_list[0].size(0) == 0:
            return [],[],[]
        # return self.detect_roi_head.simple_test(
        #     x, proposal_list, img_metas, rescale=rescale, ifdet=False)
        det_bboxes, det_labels = self.detect_roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale, ifdet=True)
        # det_bboxes, det_labels = labelstransform.simple_test_unRepeated(det_bboxes, det_labels)
        person_bboxes, person_labels, face_bboxes, face_labels, car_bboxes, car_labels, pets_bboxes, pets_labels = labelstransform.simple_test_splitTarget(det_bboxes, det_labels, points, 0, 1, 2, 3, 4, 6, 7, 8)
        if len(face_bboxes) > 0:
            #人脸姿态分类
            faceZitais_list_all = self.facezitai_roi_head.simple_cls_test(x, face_bboxes.clone(), img_metas).cpu().numpy()
            # 人脸关键点检测
            faceKps_list_all = self.faceKp_roi_head.simple_hy_kp_test(
                x, face_bboxes.clone(), img_metas, rescale=rescale).cpu().numpy()
            # 人脸模糊度评估
            faceMohus_list_all = self.facemohu_roi_head.simple_hy_qt_test(
                x, face_bboxes.clone(), img_metas, rescale=rescale).cpu().numpy()

        if len(person_bboxes) > 0:
            # 行人特征切割
            withfactorPerson_bboxes, withfactorPerson_labels, nofactorPerson_bboxes, nofactorPerson_labels = \
                labelstransform.simple_test_findClearPerson(person_bboxes, person_labels, minWidth=100, minHeight=300)
            if len(withfactorPerson_bboxes) > 0:
                persons_x_list, person_img_metas = labelstransform.simple_test_findPersonFeats(x, withfactorPerson_bboxes, img_metas)
                for i in range(len(persons_x_list)):
                    # 行人属性目标检测
                    proposal_list = self.bodydetect_rpn_head.simple_test_rpn(persons_x_list[i], [person_img_metas[i]])
                    bodydetect_bboxes_single, bodydetect_labels_single = self.bodydetect_roi_head.simple_test(
                        persons_x_list[i], proposal_list, [person_img_metas[i]], rescale=True, ifdet=True)
                    # bodydetect_bboxes_single = labelstransform.simple_test_BboxesConvert(clearPerson_bboxes[i], bodydetect_bboxes_single)
                    # 筛选上衣
                    upClouse_bboxes_single, upClouse_labels_single = labelstransform.simple_test_findDstLabels(bodydetect_bboxes_single, bodydetect_labels_single, 0, 1)
                    # 筛选下衣
                    downClouse_bboxes_single, downclouse_labels_single = labelstransform.simple_test_findDstLabels(bodydetect_bboxes_single, bodydetect_labels_single, 2, 4)
                    # 其他属性
                    otherFactor_bboxes_single, otherFactor_labels_single = labelstransform.simple_test_findDstLabels(bodydetect_bboxes_single, bodydetect_labels_single, 5, 10)
                    clouse_bboxes_single = torch.cat([upClouse_bboxes_single, downClouse_bboxes_single])
                    # 颜色分类
                    clouse_color_labels_single = self.clouse_color_roi_head.simple_cls_test(persons_x_list[i], clouse_bboxes_single.clone(), [person_img_metas[i]])
                    upClouse_color_labels_single = clouse_color_labels_single[0:upClouse_bboxes_single.size(0)]
                    downClouse_color_labels_single = clouse_color_labels_single[upClouse_bboxes_single.size(0):]
                    # 风格分类
                    upClouse_style_labels_single = self.clouse_style_roi_head.simple_cls_test(persons_x_list[i], upClouse_bboxes_single.clone(), [person_img_metas[i]])

                    upClouse_bboxes_single = labelstransform.simple_test_BboxesConvert(withfactorPerson_bboxes[i], upClouse_bboxes_single)
                    downClouse_bboxes_single = labelstransform.simple_test_BboxesConvert(withfactorPerson_bboxes[i], downClouse_bboxes_single)
                    otherFactor_bboxes_single = labelstransform.simple_test_BboxesConvert(withfactorPerson_bboxes[i], otherFactor_bboxes_single)

                    face_other_bboxes_single, face_other_labels_single = labelstransform.simple_test_findOther_face(otherFactor_bboxes_single, otherFactor_labels_single)

                    upClouse_bboxes_list.append(upClouse_bboxes_single.cpu().numpy())
                    upClouse_labels_list.append(upClouse_labels_single.cpu().numpy())
                    upClouse_style_labels_list.append(upClouse_style_labels_single.cpu().numpy())
                    upClouse_color_labels_list.append(upClouse_color_labels_single.cpu().numpy())
                    downClouse_bboxes_list.append(downClouse_bboxes_single.cpu().numpy())
                    downClouse_labels_list.append(downclouse_labels_single.cpu().numpy())
                    downClouse_color_labels_list.append(downClouse_color_labels_single.cpu().numpy())
                    otherFactor_bboxes_list.append(otherFactor_bboxes_single.cpu().numpy())
                    otherFactor_labels_list.append(otherFactor_labels_single.cpu().numpy())
                    face_other_bboxes_list_all.append(face_other_bboxes_single.cpu().numpy())
                    face_other_labels_list_all.append(face_other_labels_single.cpu().numpy())


        face_bboxes_list_all = face_bboxes.cpu().numpy()
        face_labels_list_all = face_labels.cpu().numpy()
        withfactorPerson_bboxes = withfactorPerson_bboxes.cpu().numpy()
        withfactorPerson_labels = withfactorPerson_labels.cpu().numpy()
        nofactorPerson_bboxes = nofactorPerson_bboxes.cpu().numpy()
        nofactorPerson_labels = nofactorPerson_labels.cpu().numpy()
        car_bboxes = car_bboxes.cpu().numpy()
        car_labels = car_labels.cpu().numpy()
        pets_bboxes = pets_bboxes.cpu().numpy()
        pets_labels = pets_labels.cpu().numpy()
        # 对应人脸属性
        for i in range(len(withfactorPerson_bboxes)):
            face_bbox_item, face_label_item, faceKps_item, faceZitais_item, faceMohus_item = [], [], [],[],[]
            for j in range(len(face_bboxes_list_all)):
                if labelstransform.IoF(face_bboxes_list_all[j], withfactorPerson_bboxes[i]) > 0.5:
                    face_bbox_item = face_bboxes_list_all[j]
                    face_label_item = face_labels_list_all[j]
                    faceKps_item = faceKps_list_all[j]
                    faceZitais_item = faceZitais_list_all[j]
                    faceMohus_item = faceMohus_list_all[j]
                    break
            face_bboxes_list.append(face_bbox_item)
            face_labels_list.append(face_label_item)
            faceKps_list.append(faceKps_item)
            faceZitais_list.append(faceZitais_item)
            faceMohus_list.append(faceMohus_item)

        return car_bboxes, car_labels, pets_bboxes, pets_labels, withfactorPerson_bboxes, withfactorPerson_labels, \
        nofactorPerson_bboxes, nofactorPerson_labels, upClouse_bboxes_list, upClouse_labels_list, upClouse_color_labels_list,\
        upClouse_style_labels_list, downClouse_bboxes_list, downClouse_labels_list, downClouse_color_labels_list, \
        otherFactor_bboxes_list, otherFactor_labels_list, face_bboxes_list, face_labels_list, faceKps_list, faceZitais_list, \
        faceMohus_list, face_bboxes_list_all, face_labels_list_all, face_other_bboxes_list_all, face_other_labels_list_all,\
        faceKps_list_all, faceZitais_list_all, faceMohus_list_all


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
