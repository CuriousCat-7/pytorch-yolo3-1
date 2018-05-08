import torch
from torch import nn
from torch.nn import functional as F
from region_loss import RegionLoss
from utils import get_region_boxes

class YOLOLayer(RegionLoss):
    def __init__(self, anchor_mask=[], *args, **kwargs):
        super(YOLOLayer, self).__init__(*args, **kwargs)
        self.anchor_mask = anchor_mask
        self.stride = None
    def forward(self, output, target=None):
        if self.training:
            return RegionLoss.forward(output, target)
        else:
            masked_anchors=[]
            for m in self.anchor_mask:
                masked_anchors += self.anchors[m*self.anchor_step: (m+1)*self.anchor_step]
            masked_anchors = [anchor/self.anchor_step for anchor in masked_anchors]
            boxes = get_region_boxes(output.data, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask))
            return boxes

