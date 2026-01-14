# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models import build_detector

from mmtrack.core import outs2results,results2outs
from ..builder import MODELS, build_motion, build_reid, build_tracker
from ..motion import CameraMotionCompensation, LinearMotion
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class AuavReID(BaseMultiObjectTracker):
    """Tracking without bells and whistles.

    Details can be found at `Tracktor<https://arxiv.org/abs/1903.05625>`_.
    """

    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 pretrains=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            # if detector:
            #     detector_pretrain = pretrains.get('detector', None)
            #     if detector_pretrain:
            #         detector.init_cfg = dict(
            #             type='Pretrained', checkpoint=detector_pretrain)
            #     else:
            #         detector.init_cfg = None
            if reid:
                reid_pretrain = pretrains.get('reid', None)
                if reid_pretrain:
                    reid.init_cfg = dict(
                        type='Pretrained', checkpoint=reid_pretrain)
                else:
                    reid.init_cfg = None
        # if detector is not None:
        #     self.detector = build_detector(detector)

        if reid is not None:
            self.reid = build_reid(reid)

        # if motion is not None:
        #     self.motion = build_motion(motion)
        #     if not isinstance(self.motion, list):
        #         self.motion = [self.motion]
        #     for m in self.motion:
        #         if isinstance(m, CameraMotionCompensation):
        #             self.cmc = m
        #         if isinstance(m, LinearMotion):
        #             self.linear_motion = m

        if tracker is not None:
            self.tracker = build_tracker(tracker)

    @property
    def with_cmc(self):
        """bool: whether the framework has a camera model compensation
                model.
        """
        return hasattr(self, 'cmc') and self.cmc is not None

    @property
    def with_linear_motion(self):
        """bool: whether the framework has a linear motion model."""
        return hasattr(self,
                       'linear_motion') and self.linear_motion is not None

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        raise NotImplementedError(
            'Please train `detector` and `reid` models firstly, then \
                inference with Tracktor.')

    def simple_test(self,
                    img,
                    img_metas,
                    rescale=False,
                    public_bboxes=None,
                    **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        # x = self.detector.extract_feat(img)
        # if hasattr(self.detector, 'roi_head'):
        #     # TODO: check whether this is the case
        #     if public_bboxes is not None:
        #         public_bboxes = [_[0] for _ in public_bboxes]
        #         proposals = public_bboxes
        #     else:
        #         proposals = self.detector.rpn_head.simple_test_rpn(
        #             x, img_metas)
        #     det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
        #         x,
        #         img_metas,
        #         proposals,
        #         self.detector.roi_head.test_cfg,
        #         rescale=rescale)
        #     # TODO: support batch inference
        #     det_bboxes = det_bboxes[0]
        #     det_labels = det_labels[0]
        #     num_classes = self.detector.roi_head.bbox_head.num_classes
        # elif hasattr(self.detector, 'bbox_head'):
        #     num_classes = self.detector.bbox_head.num_classes
        #     raise NotImplementedError(
        #         'Tracktor must need "roi_head" to refine proposals.')
        # else:
        #     raise TypeError('detector must has roi_head or bbox_head.')

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            feats=x,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'])
    
    # 暂时先处理追踪框，正确做法是处理检测框，以后再改，切记
    def reid_mdmt(self,reid_model,  result1,result2,**kwargs):
        num_classes = result1.get('num_classes', None)
        track_bboxes1 = result1.get('track_bboxes', None)
        det_bboxes1 = result1.get('det_bboxes', None)
        out1_track_bboxes1 = results2outs(bbox_results=track_bboxes1,)
        out1_track_bboxes1_bboxes = out1_track_bboxes1.get('bboxes',None)
        out1_track_bboxes1_labels = out1_track_bboxes1.get('labels',None)
        out1_det_bboxes1 = results2outs(bbox_results=det_bboxes1,)
        # out1_det_bboxes1_bboxes = out1_det_bboxes1.get('bboxes',None)
        # out1_det_bboxes1_labels = out1_det_bboxes1.get('labels',None)
        img1 = result1.get('img', None)
        img_metas1 = result1.get('img_metas', None)

        out1_det_bboxes1_bboxes = result1.get('det_bboxes_tensor', None)
        out1_det_bboxes1_labels = result1.get('det_labels_tensor', None)

        track_bboxes2 = result2.get('track_bboxes', None)
        det_bboxes2 = result2.get('det_bboxes', None)
        out2_track_bboxes2 = results2outs(bbox_results=track_bboxes2,)
        out2_track_bboxes2_bboxes = out2_track_bboxes2.get('bboxes',None)
        out2_track_bboxes2_labels = out2_track_bboxes2.get('labels',None)
        out2_det_bboxes2 = results2outs(bbox_results=det_bboxes2,)
        # out2_det_bboxes2_bboxes = out2_det_bboxes2.get('bboxes',None)
        # out2_det_bboxes2_labels = out2_det_bboxes2.get('labels',None)
        img2 = result2.get('img', None)
        img_metas2 = result2.get('img_metas', None)

        out2_det_bboxes2_bboxes = result2.get('det_bboxes_tensor', None)
        out2_det_bboxes2_labels = result2.get('det_labels_tensor', None)

        # print("out1_track_bboxes1 = ",out1_track_bboxes1)
        # print("out1_det_bboxes1 = ",out1_det_bboxes1)
        # print("img1 = ",img1)
        # print("img_metas1 = ",img_metas1)
        # print("det_bboxes1 = ",det_bboxes1)
        # print("out1_det_bboxes1_bboxes = ",out1_det_bboxes1_bboxes)
        # print("out1_det_bboxes1_labels = ",out1_det_bboxes1_labels)

        


        track_bboxes, track_labels, track_ids = self.tracker.reid_track(
            img=img1,
            img_metas=img_metas1,
            model=self,
            bboxes1=out1_det_bboxes1_bboxes,
            labels1=out1_det_bboxes1_labels,
            frame_id1=0,
            img2 = img2,
            img_metas2=img_metas2,            
            bboxes2=out2_det_bboxes2_bboxes,
            labels2=out2_det_bboxes2_labels,
            rescale=True,
            **kwargs)
        

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        # det_results = outs2results(
        #     bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            # det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'])

        return out1_track_bboxes1

    # 暂时先处理追踪框，正确做法是处理检测框，以后再改，切记
    def reid_mdmt_com(self,reid_model,  result1,result2,**kwargs):
        num_classes = result1.get('num_classes', None)
        track_bboxes1 = result1.get('track_bboxes', None)
        det_bboxes1 = result1.get('det_bboxes', None)
        out1_track_bboxes1 = results2outs(bbox_results=track_bboxes1,)
        out1_track_bboxes1_bboxes = out1_track_bboxes1.get('bboxes',None)
        out1_track_bboxes1_labels = out1_track_bboxes1.get('labels',None)
        out1_det_bboxes1 = results2outs(bbox_results=det_bboxes1,)
        # out1_det_bboxes1_bboxes = out1_det_bboxes1.get('bboxes',None)
        # out1_det_bboxes1_labels = out1_det_bboxes1.get('labels',None)
        img1 = result1.get('img', None)
        img_metas1 = result1.get('img_metas', None)

        out1_det_bboxes1_bboxes = result1.get('det_bboxes_tensor', None)
        out1_det_bboxes1_labels = result1.get('det_labels_tensor', None)
        out1_det_bboxes1_ids1 = result1.get('track_ids_tensor', None)

        track_bboxes2 = result2.get('track_bboxes', None)
        det_bboxes2 = result2.get('det_bboxes', None)
        out2_track_bboxes2 = results2outs(bbox_results=track_bboxes2,)
        out2_track_bboxes2_bboxes = out2_track_bboxes2.get('bboxes',None)
        out2_track_bboxes2_labels = out2_track_bboxes2.get('labels',None)
        out2_det_bboxes2 = results2outs(bbox_results=det_bboxes2,)
        # out2_det_bboxes2_bboxes = out2_det_bboxes2.get('bboxes',None)
        # out2_det_bboxes2_labels = out2_det_bboxes2.get('labels',None)
        img2 = result2.get('img', None)
        img_metas2 = result2.get('img_metas', None)

        out2_det_bboxes2_bboxes = result2.get('det_bboxes_tensor', None)
        out2_det_bboxes2_labels = result2.get('det_labels_tensor', None)
        out2_det_bboxes2_ids2 = result2.get('track_ids_tensor', None)

        # print("out1_track_bboxes1 = ",out1_track_bboxes1)
        # print("out1_det_bboxes1 = ",out1_det_bboxes1)
        # print("img1 = ",img1)
        # print("img_metas1 = ",img_metas1)
        # print("det_bboxes1 = ",det_bboxes1)
        # print("out1_det_bboxes1_bboxes = ",out1_det_bboxes1_bboxes)
        # print("out1_det_bboxes1_labels = ",out1_det_bboxes1_labels)

        


        track_bboxes, track_labels, track_ids = self.tracker.reid_track_com(
            img=img1,
            img_metas=img_metas1,
            model=self,
            bboxes1=out1_det_bboxes1_bboxes,
            labels1=out1_det_bboxes1_labels,
            ids1 = out1_det_bboxes1_ids1,
            frame_id1=0,
            img2 = img2,
            img_metas2=img_metas2,            
            bboxes2=out2_det_bboxes2_bboxes,
            labels2=out2_det_bboxes2_labels,
            ids2 = out2_det_bboxes2_ids2,
            rescale=True,
            **kwargs)
        

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        # det_results = outs2results(
        #     bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)
        print("track_results = ",track_results)

        return dict(
            # det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'])