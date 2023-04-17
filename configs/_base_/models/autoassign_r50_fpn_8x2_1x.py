
# _base_ = [
#     '../datasets/full_mdmt_detection.py',
#     #  '../default_runtime.py'
# ]
model = dict(
    detector=dict(
        type='AutoAssign',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='caffe',

            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet50_caffe')
                ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs=True,
            num_outs=5,
            relu_before_extra_convs=True,
            init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')),
        bbox_head=dict(
            type='AutoAssignHead',
            num_classes=3,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            strides=[8, 16, 32, 64, 128],
            loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
        train_cfg=None,
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100)
            )

        )


