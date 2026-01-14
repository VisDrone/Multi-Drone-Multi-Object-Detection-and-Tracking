from mmcv.runner import BaseModule

from ..builder import REID

@REID.register_module()
class Mdmt_ReID(BaseModule):

    def __init__(self,
                arg1,
                arg2):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass