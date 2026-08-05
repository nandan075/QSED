"""
Classical image processing modules for Quantum Image Edge Detection (QSED).
Contains image loading, grayscale conversion, 8-direction Sobel operators,
gradient computation, non-maximum suppression, double thresholding, and hysteresis.
"""

from .image_loader import load_image, save_image, generate_synthetic_image
from .grayscale import to_grayscale
from .sobel_masks import get_eight_sobel_masks
from .sobel_operator import apply_sobel_masks
from .gradient import compute_gradient_magnitude
from .nms import non_maximum_suppression
from .threshold import double_threshold
from .hysteresis import edge_tracking_hysteresis

__all__ = [
    "load_image",
    "save_image",
    "generate_synthetic_image",
    "to_grayscale",
    "get_eight_sobel_masks",
    "apply_sobel_masks",
    "compute_gradient_magnitude",
    "non_maximum_suppression",
    "double_threshold",
    "edge_tracking_hysteresis",
]
