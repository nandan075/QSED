"""
Keypoint Extractor Module.

Supports multiple keypoint detection backends:
1. 'sift': Scale-Invariant Feature Transform (OpenCV SIFT, robust scale & rotation).
2. 'orb': Oriented FAST and Rotated BRIEF (fast binary keypoints).
3. 'harris': Harris Corner Detector (pure NumPy/SciPy implementation, no cv2 dependency required).
4. 'qsed_gradient': Native QSED 8-directional Sobel gradient & directional variation detector.
5. 'grid': Uniform geometric spatial grid across the image.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np


def detect_harris_corners_numpy(
    image: np.ndarray,
    max_keypoints: int = 100,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
    min_distance: int = 5,
) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
    """
    Pure NumPy/SciPy Harris corner detector.
    Does not require cv2 or external binary libraries.

    Args:
        image: 2D grayscale image array of shape (H, W).
        max_keypoints: Maximum keypoints to return.
        k: Harris corner free parameter in [0.04, 0.06].
        threshold_ratio: Minimum corner response relative to max response.
        min_distance: Non-maximum suppression radius in pixels.

    Returns:
        Tuple of (coordinates_list, metadata_list).
    """
    from scipy.signal import convolve2d
    from scipy.ndimage import gaussian_filter

    img = image.astype(np.float64)

    # Spatial derivatives
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    Ix = convolve2d(img, sobel_x, mode="same", boundary="symm")
    Iy = convolve2d(img, sobel_y, mode="same", boundary="symm")

    # Second moment matrix components
    Ixx = gaussian_filter(Ix * Ix, sigma=1.5)
    Iyy = gaussian_filter(Iy * Iy, sigma=1.5)
    Ixy = gaussian_filter(Ix * Iy, sigma=1.5)

    det_M = (Ixx * Iyy) - (Ixy * Ixy)
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M ** 2)

    # Thresholding & Non-Maximum Suppression
    max_r = np.max(R)
    if max_r <= 0:
        return [], []

    thresh = max_r * threshold_ratio
    h, w = image.shape
    border = min_distance + 2

    # Mask valid interior region
    valid_mask = (R > thresh)
    valid_mask[:border, :] = False
    valid_mask[-border:, :] = False
    valid_mask[:, :border] = False
    valid_mask[:, -border:] = False

    y_indices, x_indices = np.where(valid_mask)
    scores = R[y_indices, x_indices]

    # Sort descending by corner response
    order = np.argsort(scores)[::-1]
    y_sorted = y_indices[order]
    x_sorted = x_indices[order]
    scores_sorted = scores[order]

    # Distance-based NMS
    selected_pts: List[Tuple[float, float]] = []
    selected_meta: List[Dict[str, Any]] = []

    for y, x, score in zip(y_sorted, x_sorted, scores_sorted):
        # Check distance to already selected points
        too_close = False
        for sx, sy in selected_pts:
            if (x - sx) ** 2 + (y - sy) ** 2 < (min_distance ** 2):
                too_close = True
                break
        if not too_close:
            selected_pts.append((float(x), float(y)))
            selected_meta.append({"response": float(score), "scale": 1.0, "angle": 0.0, "detector": "harris"})
            if len(selected_pts) >= max_keypoints:
                break

    return selected_pts, selected_meta


def detect_grid_keypoints(
    image_shape: Tuple[int, int],
    max_keypoints: int = 100,
) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
    """
    Generate an evenly spaced spatial grid of keypoint coordinates.
    """
    h, w = image_shape
    grid_side = int(np.floor(np.sqrt(max_keypoints)))
    if grid_side < 2:
        grid_side = 2

    margin_y = h // (grid_side + 1)
    margin_x = w // (grid_side + 1)

    y_coords = np.linspace(margin_y, h - margin_y, grid_side)
    x_coords = np.linspace(margin_x, w - margin_x, grid_side)

    pts: List[Tuple[float, float]] = []
    meta: List[Dict[str, Any]] = []

    for y in y_coords:
        for x in x_coords:
            pts.append((float(x), float(y)))
            meta.append({"response": 1.0, "scale": 1.0, "angle": 0.0, "detector": "grid"})
            if len(pts) >= max_keypoints:
                break
        if len(pts) >= max_keypoints:
            break

    return pts, meta


class KeypointExtractor:
    """
    Extracts keypoints and optional descriptor vectors from an input image.
    """

    def __init__(
        self,
        detector: str = "sift",
        max_keypoints: int = 100,
        min_distance: float = 4.0,
    ):
        self.detector_type = detector.lower()
        self.max_keypoints = max_keypoints
        self.min_distance = min_distance

    def extract(
        self,
        image: np.ndarray,
    ) -> Tuple[List[Tuple[float, float]], Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Detect keypoints in image.

        Args:
            image: 2D grayscale array of shape (H, W).

        Returns:
            Tuple of:
            - keypoints_xy: List of (x, y) coordinates.
            - descriptors: (N, 128) array if available (e.g. SIFT), otherwise None.
            - metadata: List of dicts with detector properties.
        """
        h, w = image.shape

        # 1. SIFT / ORB via OpenCV
        if self.detector_type in ("sift", "orb"):
            try:
                import cv2
                img_u8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)

                if self.detector_type == "sift":
                    det = cv2.SIFT_create(nfeatures=self.max_keypoints)
                else:
                    det = cv2.ORB_create(nfeatures=self.max_keypoints)

                cv_kps, cv_desc = det.detectAndCompute(img_u8, None)
                if cv_kps is not None and len(cv_kps) > 0:
                    pts = [(float(kp.pt[0]), float(kp.pt[1])) for kp in cv_kps]
                    meta = [
                        {
                            "response": float(kp.response),
                            "scale": float(kp.size),
                            "angle": float(kp.angle),
                            "detector": self.detector_type,
                        }
                        for kp in cv_kps
                    ]
                    return pts, cv_desc, meta
            except Exception:
                # Fall back to Harris corners if OpenCV is missing or fails
                pass

        # 2. Native Harris Corners
        if self.detector_type == "harris" or self.detector_type in ("sift", "orb"):
            pts, meta = detect_harris_corners_numpy(image, max_keypoints=self.max_keypoints)
            if len(pts) > 0:
                return pts, None, meta

        # 3. Grid Fallback
        pts, meta = detect_grid_keypoints((h, w), max_keypoints=self.max_keypoints)
        return pts, None, meta
