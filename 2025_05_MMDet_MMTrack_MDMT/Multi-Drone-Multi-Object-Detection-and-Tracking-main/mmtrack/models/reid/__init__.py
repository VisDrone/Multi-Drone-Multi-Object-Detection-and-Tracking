# Copyright (c) OpenMMLab. All rights reserved.
from .base_reid import BaseReID
from .fc_module import FcModule
from .gap import GlobalAveragePooling
from .linear_reid_head import LinearReIDHead
from .mdmt_reid import Mdmt_ReID

__all__ = ['BaseReID', 'GlobalAveragePooling', 'LinearReIDHead', 'FcModule','Mdmt_ReID']
