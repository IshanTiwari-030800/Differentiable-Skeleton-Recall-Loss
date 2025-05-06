A custom PyTorch loss function that computes skeleton recall in a fully differentiable manner — by leveraging predefined convolution kernels for skeletonization. Designed to evaluate how well predicted segmentations preserve the topology of thin, elongated structures like vessels, airways, or stones.

🚀 Highlights <br>
>🔍 Skeleton-Aware Loss: Goes beyond boundary or region metrics to focus on topological correctness.

🔁 Fully Differentiable: Unlike traditional skeletonization (e.g., morphological thinning), this loss uses fixed convolutional kernels to simulate skeleton extraction in a differentiable way.

⚙️ Plug-and-Play: Easily integrates into existing PyTorch training pipelines.

🧠 Inspired by classical vision: Combines the intuition of medial axis transforms with modern deep learning.

> Skeletonize.py - Code to skeletonize the given 3D mask
> SkeletonRecall.py - Code to compute soft skeleton recall loss between GT mask and skeletonized mask.

📌 Motivation <br>
>This loss function is inspired by the design philosophy of nnU-Net, which emphasizes strong architectural and loss design priors for medical image segmentation. The Differentiable Skeleton Recall Loss computes skeletons of both ground truth and predicted masks on the fly during training using fixed convolutional kernel operations. By doing so in a fully differentiable manner, it encourages the network to preserve topological structures—such as thin, elongated, or branching regions—more effectively than region-based losses alone. This loss can significantly improve model learning in tasks where accurate object centerlines or connectivity matter, such as vessel segmentation, airway extraction, or stone detection.

Requirements ⚙️ <br>
>Python >= 3.10.0 <br>
>torch >= 1.10.0 <br>
>numpy >= 1.21.0
