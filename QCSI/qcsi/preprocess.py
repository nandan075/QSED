"""
QCSI Image Preprocessing Module.

Handles:
1. EXIF orientation correction.
2. Aspect-ratio preserving downsampling/resizing.
3. Contrast-Limited Adaptive Histogram Equalization (CLAHE).
4. Dual representations: Grayscale/CLAHE for local SIFT keypoints, RGB normalized tensor for DINOv2.
"""

from typing import Tuple, Union, Optional
import io
import numpy as np
import cv2
from PIL import Image, ImageOps


class ImagePreprocessor:
    """Standardized preprocessing pipeline for hybrid quantum-classical vision."""

    def __init__(
        self,
        max_dim: int = 1024,
        dinov2_size: int = 224,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.max_dim = max_dim
        self.dinov2_size = dinov2_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )

    def load_image(self, source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Loads an image from path, raw bytes, PIL Image, or numpy array.
        Applies EXIF transpose and returns an RGB uint8 numpy array (H, W, 3).
        """
        if isinstance(source, str):
            with Image.open(source) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert("RGB")
                return np.array(pil_img, dtype=np.uint8)

        elif isinstance(source, bytes):
            with Image.open(io.BytesIO(source)) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert("RGB")
                return np.array(pil_img, dtype=np.uint8)

        elif isinstance(source, Image.Image):
            pil_img = ImageOps.exif_transpose(source)
            pil_img = pil_img.convert("RGB")
            return np.array(pil_img, dtype=np.uint8)

        elif isinstance(source, np.ndarray):
            img = source.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return img.astype(np.uint8)

        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

    def resize_keep_aspect(self, img_rgb: np.ndarray, max_dim: Optional[int] = None) -> np.ndarray:
        """
        Resizes an image preserving aspect ratio so the longest dimension <= max_dim.
        """
        max_dim = max_dim or self.max_dim
        h, w = img_rgb.shape[:2]
        if max(h, w) <= max_dim:
            return img_rgb

        scale = max_dim / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Applies CLAHE on single-channel grayscale image."""
        if gray.ndim != 2:
            raise ValueError("CLAHE expects a 2D single-channel grayscale array.")
        return self._clahe.apply(gray)

    def prepare_local(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares image for Branch A (local keypoints):
        Returns (resized_gray, clahe_enhanced_gray).
        """
        resized = self.resize_keep_aspect(img_rgb)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        clahe_gray = self.apply_clahe(gray)
        return gray, clahe_gray

    def prepare_global(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Prepares image for Branch B (DINOv2):
        Returns resized RGB array of shape (dinov2_size, dinov2_size, 3).
        """
        # Resize to fixed square for standard ViT patch grid
        resized = cv2.resize(
            img_rgb,
            (self.dinov2_size, self.dinov2_size),
            interpolation=cv2.INTER_AREA,
        )
        return resized

    def preprocess_pair(
        self,
        src_A: Union[str, bytes, np.ndarray, Image.Image],
        src_B: Union[str, bytes, np.ndarray, Image.Image],
    ) -> dict:
        """
        Full dual-branch preprocessing for a pair of images.
        """
        rgb_A = self.load_image(src_A)
        rgb_B = self.load_image(src_B)

        gray_A, clahe_A = self.prepare_local(rgb_A)
        gray_B, clahe_B = self.prepare_local(rgb_B)

        dino_A = self.prepare_global(rgb_A)
        dino_B = self.prepare_global(rgb_B)

        return {
            "rgb_A": rgb_A,
            "rgb_B": rgb_B,
            "gray_A": gray_A,
            "gray_B": gray_B,
            "clahe_A": clahe_A,
            "clahe_B": clahe_B,
            "dino_A": dino_A,
            "dino_B": dino_B,
            "shape_A": rgb_A.shape[:2],
            "shape_B": rgb_B.shape[:2],
        }
