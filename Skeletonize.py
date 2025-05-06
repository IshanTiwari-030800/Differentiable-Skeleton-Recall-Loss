import torch.nn as nn
import torch.nn.functional as F
import torch

class DifferentiableSkeletonize(nn.Module):

    """
    Differentiable approximation of skeletonize morphological operations. Takes binary mask as input, hence c = 1.

    Args
        do_tube: If True, 2px tube around the skeleton will be added, Default is False
        tau: Soft thresholding parameter. High value will lead to softer skeletonization
        iterations: Number of skeletonization iterations
        blend_factor: How much to mix the original mask with the skeletonized output
    """

    def __init__(self, do_tube: bool = False, tau: float = 0.3, iterations: int = 5, blend_factor: float = 0.3):

        """
        Args:
            segmentation - Tensor of shape (b, 1, h, w, d)
        Returns:
            Tensor - The skeletonized tensor of shape (b, 1, h, w, d)
        """

        super(DifferentiableSkeletonize, self).__init__()
        self.do_tube = do_tube
        self.tau = tau
        self.iterations = iterations
        self.blend_factor = blend_factor

    def forward(self, segmentation: torch.tensor) -> torch.tensor:

        b, c, h, w, d = segmentation.shape
        segmentation = segmentation.view(b * c, h, w, d)

        def soft_threshold(x, tau): # Reduces the sharpness of the skeletonization
            return torch.sigmoid((x - 0.5) / tau)

        def differentiable_erosion(x): # Erodes the mask
            return -F.avg_pool3d(-x.unsqueeze(0), kernel_size=3, stride=1, padding=1)
        
        def differentiable_dilation(x): # Dilates the mask
            return F.avg_pool3d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1)
        
        skeleton = torch.zeros_like(segmentation)
        for i in range(segmentation.shape[0]):
            mask = segmentation[i]
            for _ in range(self.iterations):
                eroded = differentiable_erosion(mask)
                dilated = differentiable_dilation(eroded).squeeze() # reduce to (h, w, d)

                skeleton[i] += soft_threshold(mask - dilated, self.tau)
                skeleton[i] += (mask - dilated)
                
        skeleton = skeleton.view(b, c, h, w, d)

        if self.do_tube:
            skeleton = differentiable_dilation(differentiable_dilation(skeleton))

        skeleton = self.blend_factor * segmentation.view(b, c, h, w, d) + (1 - self.blend_factor) * skeleton
        skeleton = torch.clamp(skeleton, 0, 1)  # Ensure values are between 0 and 1
        skeleton = torch.round(skeleton)  # Round to nearest integer (0 or 1)

        return 1 - skeleton