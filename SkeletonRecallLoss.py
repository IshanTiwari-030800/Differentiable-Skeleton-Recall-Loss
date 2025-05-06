import torch
import numpy as np
import torch.nn as nn

class SoftSkeletonRecallLoss(nn.Module):


    """
    Compute the soft skeleton recall loss. This loss is the negative of the soft dice score of the skeleton of the 
    prediction and the skeleton of the ground truth. The skeleton is calculated using skimage.morphology.skeletonize.
    The loss is 1 - dice_score(skeletonize(pred), skeletonize(gt)). The loss is 0 if the skeletons are identical.

    Args:
        1. apply_nonlin: Callable, default = None
        2. smooth: float, default = 1.
        3. x: predicted segmentation mask, shape = (b, c, x, y, z) | c - number of classes
        4. y: ground truth segmentation mask, shape = (b, c, x, y, z) | c - number of classes

        Note  - The input x and y should be one-hot encoded, if not, the code will one-hot encode it internally.

    Returns:

        -rec: float, the soft skeleton recall loss
    """

    def __init__(self, apply_nonlin: Callable = None, 
                 smooth: float = 1.):

        super(SoftSkeletonRecallLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:])) # Reshape to have the same number of dimensions as x

            sum_gt = y.sum(axes)

        inter_rec = (x * y).sum(axes)
        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8)) # Recall
        rec = rec.mean() # Mean over the batch

        return -rec