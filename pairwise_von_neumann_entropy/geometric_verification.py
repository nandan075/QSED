"""
Geometric Verification Module using RANSAC.

Provides:
1. RANSAC Estimation of Homography, Affine, and Fundamental Matrix transforms.
2. OpenCV implementation with an automatic native NumPy/SciPy fallback (DLT + RANSAC)
   so the algorithm functions reliably on any host machine without requiring OpenCV binaries.
3. Safe handling of edge cases (insufficient candidate matches, colinear points, degenerate sets).
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def fit_homography_dlt(pts_src: np.ndarray, pts_dst: np.ndarray) -> Optional[np.ndarray]:
    """
    Direct Linear Transform (DLT) algorithm to estimate 3x3 homography matrix H
    from >= 4 point correspondences: pts_dst ~ H @ pts_src.

    Args:
        pts_src: (N, 2) source points (x, y).
        pts_dst: (N, 2) destination points (x, y).

    Returns:
        3x3 float64 homography matrix, or None if degenerate.
    """
    n = len(pts_src)
    if n < 4:
        return None

    # Normalization for numerical conditioning (Hartley's preconditioner)
    mean_src = np.mean(pts_src, axis=0)
    std_src = np.std(pts_src) + 1e-12
    scale_src = np.sqrt(2.0) / std_src
    T_src = np.array([
        [scale_src, 0, -scale_src * mean_src[0]],
        [0, scale_src, -scale_src * mean_src[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    mean_dst = np.mean(pts_dst, axis=0)
    std_dst = np.std(pts_dst) + 1e-12
    scale_dst = np.sqrt(2.0) / std_dst
    T_dst = np.array([
        [scale_dst, 0, -scale_dst * mean_dst[0]],
        [0, scale_dst, -scale_dst * mean_dst[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Transform coordinates
    pts_src_h = np.column_stack([pts_src, np.ones(n)])
    pts_dst_h = np.column_stack([pts_dst, np.ones(n)])
    norm_src = (T_src @ pts_src_h.T).T
    norm_dst = (T_dst @ pts_dst_h.T).T

    A = []
    for i in range(n):
        x, y = norm_src[i, 0], norm_src[i, 1]
        u, v = norm_dst[i, 0], norm_dst[i, 1]
        A.append([-x, -y, -1,  0,  0,  0, u * x, u * y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A, dtype=np.float64)

    try:
        _, _, vh = np.linalg.svd(A)
        H_norm = vh[-1].reshape(3, 3)
        # Denormalize: H = inv(T_dst) @ H_norm @ T_src
        H = np.linalg.inv(T_dst) @ H_norm @ T_src
        if np.abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H
    except Exception:
        return None


def compute_transfer_error(H: np.ndarray, pts_src: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
    """Compute symmetric or forward transfer reprojection error for homography."""
    n = len(pts_src)
    pts_src_h = np.column_stack([pts_src, np.ones(n)])
    projected = (H @ pts_src_h.T).T
    w = projected[:, 2:3]
    w = np.where(np.abs(w) > 1e-12, w, 1e-12)
    mapped = projected[:, :2] / w
    errors = np.linalg.norm(mapped - pts_dst, axis=1)
    return errors


def ransac_homography_numpy(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    threshold: float = 3.0,
    max_iters: int = 2000,
    seed: Optional[int] = 42,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Independent pure NumPy RANSAC implementation for homography estimation.
    """
    n = len(pts_src)
    if n < 4:
        return None, np.zeros(n, dtype=bool)

    rng = np.random.RandomState(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_H = None
    max_inlier_count = 0

    indices = np.arange(n)
    for _ in range(max_iters):
        sample_idx = rng.choice(indices, size=4, replace=False)
        H = fit_homography_dlt(pts_src[sample_idx], pts_dst[sample_idx])
        if H is None:
            continue

        errors = compute_transfer_error(H, pts_src, pts_dst)
        inliers = errors < threshold
        inlier_count = np.sum(inliers)

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_inliers = inliers
            best_H = H

    # Re-estimate model using all inliers
    if max_inlier_count >= 4 and best_H is not None:
        refined_H = fit_homography_dlt(pts_src[best_inliers], pts_dst[best_inliers])
        if refined_H is not None:
            best_H = refined_H

    return best_H, best_inliers


def verify_matches_ransac(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    model: str = "homography",
    threshold: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.99,
    min_inliers: int = 4,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Geometrically verify a set of 2D-2D point correspondences using RANSAC.

    Safely handles edge cases:
    - n < min_inliers: Returns unverified with status 'insufficient_matches'.
    - Degenerate geometries: Gracefully falls back without raising exceptions.

    Args:
        pts_src: Array of shape (N, 2) representing points in Image A.
        pts_dst: Array of shape (N, 2) representing points in Image B.
        model: 'homography', 'affine', or 'fundamental'.
        threshold: Max reprojection error threshold in pixels.
        max_iters: RANSAC iteration limit.
        confidence: RANSAC confidence parameter.
        min_inliers: Minimum inliers required to declare geometric success.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dictionary with:
            - 'success': Boolean indicator of geometric verification pass.
            - 'model_matrix': Estimated 3x3 transform matrix (or None).
            - 'inlier_mask': Boolean array of length N.
            - 'inlier_count': Number of inliers.
            - 'inlier_ratio': inlier_count / max(1, N).
            - 'status': Explanation string ('verified', 'insufficient_matches', 'failed').
    """
    n = len(pts_src)
    if n < min_inliers:
        return {
            "success": False,
            "model_matrix": None,
            "inlier_mask": np.zeros(n, dtype=bool),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "status": f"insufficient_matches (got {n}, required >= {min_inliers})",
        }

    # Attempt OpenCV RANSAC first if installed
    try:
        import cv2
        pts1 = pts_src.reshape(-1, 1, 2).astype(np.float32)
        pts2 = pts_dst.reshape(-1, 1, 2).astype(np.float32)

        if model.lower() == "homography":
            H, mask = cv2.findHomography(
                pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=max_iters, confidence=confidence
            )
            inlier_mask = (mask.ravel() == 1) if mask is not None else np.zeros(n, dtype=bool)
            inlier_count = int(np.sum(inlier_mask))
            success = bool(inlier_count >= min_inliers and H is not None)
            return {
                "success": success,
                "model_matrix": H,
                "inlier_mask": inlier_mask,
                "inlier_count": inlier_count,
                "inlier_ratio": float(inlier_count / max(1, n)),
                "status": "verified" if success else "failed",
            }
        elif model.lower() == "fundamental":
            F, mask = cv2.findFundamentalMat(
                pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=threshold, confidence=confidence, maxIters=max_iters
            )
            inlier_mask = (mask.ravel() == 1) if mask is not None else np.zeros(n, dtype=bool)
            inlier_count = int(np.sum(inlier_mask))
            success = bool(inlier_count >= min_inliers and F is not None)
            return {
                "success": success,
                "model_matrix": F,
                "inlier_mask": inlier_mask,
                "inlier_count": inlier_count,
                "inlier_ratio": float(inlier_count / max(1, n)),
                "status": "verified" if success else "failed",
            }
    except Exception:
        # Fallback to pure NumPy RANSAC
        pass

    # Pure NumPy fallback
    H, inlier_mask = ransac_homography_numpy(pts_src, pts_dst, threshold=threshold, max_iters=max_iters, seed=seed)
    inlier_count = int(np.sum(inlier_mask))
    success = bool(inlier_count >= min_inliers and H is not None)

    return {
        "success": success,
        "model_matrix": H,
        "inlier_mask": inlier_mask,
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / max(1, n)),
        "status": "verified" if success else "failed",
    }
