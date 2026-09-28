"""
QCSI Synthetic Transformation and Pair Generator.

Simulates real-world photographic perturbations:
- Geometric: Rotation, scaling, perspective/homography, crop.
- Photometric: Contrast, brightness, Gaussian noise, Gaussian blur.
- Compression: JPEG quality down to 20.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class SyntheticImagePerturber:
    """Applies controlled synthetic image degradations and geometric transforms."""

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def rotate(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotates image around center with border replication."""
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated

    def scale_and_crop(self, img: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scales image and center-crops or pads back to original size."""
        h, w = img.shape[:2]
        new_w = max(10, int(w * scale_factor))
        new_h = max(10, int(h * scale_factor))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros_like(img)
        # Center crop or paste
        if scale_factor >= 1.0:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            canvas = resized[start_y:start_y + h, start_x:start_x + w]
        else:
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized

        return canvas

    def perspective_warp(self, img: np.ndarray, max_jitter: float = 0.15) -> np.ndarray:
        """Applies random projective homography."""
        h, w = img.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        jitter = max_jitter * min(w, h)
        dst_pts = src_pts + self.rng.uniform(-jitter, jitter, size=src_pts.shape).astype(np.float32)

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT)
        return warped

    def add_gaussian_noise(self, img: np.ndarray, sigma: float = 20.0) -> np.ndarray:
        """Adds additive Gaussian sensor noise."""
        noise = self.rng.normal(0, sigma, size=img.shape)
        noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        return noisy

    def add_gaussian_blur(self, img: np.ndarray, ksize: int = 5) -> np.ndarray:
        """Simulates defocus / optical blur."""
        k = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    def adjust_photometry(self, img: np.ndarray, alpha: float = 1.2, beta: float = 15.0) -> np.ndarray:
        """Alters contrast (alpha) and brightness (beta)."""
        adjusted = np.clip(img.astype(np.float64) * alpha + beta, 0, 255).astype(np.uint8)
        return adjusted

    def apply_jpeg_compression(self, img: np.ndarray, quality: int = 30) -> np.ndarray:
        """Encodes and decodes image with lossy JPEG compression."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        _, encimg = cv2.imencode(".jpg", img, encode_param)
        decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return decoded

    def generate_random_transform(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        """Applies a random realistic combination of transformations."""
        out = img.copy()
        ops = []

        # 1. Geometry
        r = self.rng.random()
        if r < 0.3:
            angle = float(self.rng.choice([90, 180, 270]))
            out = self.rotate(out, angle)
            ops.append(f"cardinal_rot_{int(angle)}")
        elif r < 0.6:
            angle = float(self.rng.uniform(-25, 25))
            out = self.rotate(out, angle)
            ops.append(f"rot_{angle:.1f}")
        elif r < 0.85:
            out = self.perspective_warp(out)
            ops.append("perspective")

        # 2. Scale
        if self.rng.random() < 0.5:
            scale = float(self.rng.uniform(0.7, 1.3))
            out = self.scale_and_crop(out, scale)
            ops.append(f"scale_{scale:.2f}")

        # 3. Photometry
        if self.rng.random() < 0.6:
            alpha = float(self.rng.uniform(0.8, 1.3))
            beta = float(self.rng.uniform(-25, 25))
            out = self.adjust_photometry(out, alpha, beta)
            ops.append("photometry")

        # 4. Noise or blur
        if self.rng.random() < 0.4:
            if self.rng.random() < 0.5:
                out = self.add_gaussian_noise(out, sigma=float(self.rng.uniform(10, 25)))
                ops.append("noise")
            else:
                out = self.add_gaussian_blur(out, ksize=int(self.rng.choice([3, 5, 7])))
                ops.append("blur")

        # 5. JPEG compression
        if self.rng.random() < 0.5:
            q = int(self.rng.integers(20, 60))
            out = self.apply_jpeg_compression(out, quality=q)
            ops.append(f"jpeg_{q}")

        transform_desc = "+".join(ops) if ops else "identity"
        return out, transform_desc
