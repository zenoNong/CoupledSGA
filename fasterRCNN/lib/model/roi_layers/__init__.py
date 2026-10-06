# PyTorch/Torchvision based ROI layers replacement
import torch
import torchvision.ops as ops

nms = ops.nms

class ROIAlign(torch.nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return ops.roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

def roi_align(input, rois, output_size, spatial_scale, sampling_ratio):
    return ops.roi_align(input, rois, output_size, spatial_scale, sampling_ratio)

class ROIPool(torch.nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return ops.roi_pool(input, rois, self.output_size, self.spatial_scale)

def roi_pool(input, rois, output_size, spatial_scale):
    return ops.roi_pool(input, rois, output_size, spatial_scale)

__all__ = ['nms', 'roi_align', 'ROIAlign', 'roi_pool', 'ROIPool']
