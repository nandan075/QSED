"""
Performance Metrics Module for Quantum Image Edge Detection (QSED).

Calculates Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and SSIM.

Paper Reference:
- Section 4.2: Experiment analysis (Equation 16, Table 3).
"""

import numpy as np


def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE) between two 2^n x 2^n images according to Equation (16).

    Equation (16):
        MSE = (1 / 2^{2n}) sum_{Y=0}^{2^n-1} sum_{X=0}^{2^n-1} [Q(Y,X) - R(Y,X)]^2

    Purpose:
        Evaluate quality of detected edge images compared to reference ground truth / lower false-edge rates.
        Per Section 4.2, fewer false edges result in a smaller MSE value.

    Inputs:
        image1 (np.ndarray): First grayscale image matrix Q(Y,X).
        image2 (np.ndarray): Second grayscale image matrix R(Y,X).

    Outputs:
        float: MSE value.

    Time Complexity:
        O(H * W) element-wise difference and sum.

    Space Complexity:
        O(H * W) difference array.

    Reference:
        Paper Section 4.2, Equation (16), Table 3.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes must match, got {image1.shape} and {image2.shape}")

    q = image1.astype(np.float64)
    r = image2.astype(np.float64)

    diff = q - r
    mse_val = np.mean(diff ** 2)
    return float(mse_val)


def calculate_psnr(image1: np.ndarray, image2: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB.

    Inputs:
        image1 (np.ndarray): Original/Reference image.
        image2 (np.ndarray): Reconstructed/Edge image.
        max_pixel (float): Maximum possible pixel value (255 for 8-bit).

    Outputs:
        float: PSNR in decibels (dB).
    """
    mse_val = calculate_mse(image1, image2)
    if mse_val == 0:
        return float('inf')
    return float(20 * np.log10(max_pixel / np.sqrt(mse_val)))


def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) using skimage if available, or fallback approximation.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(image1, image2, data_range=255.0))
    except ImportError:
        # Fallback simple structural similarity metric
        q = image1.astype(np.float64)
        r = image2.astype(np.float64)

        mu_q = np.mean(q)
        mu_r = np.mean(r)
        sigma_q = np.var(q)
        sigma_r = np.var(r)
        sigma_qr = np.mean((q - mu_q) * (r - mu_r))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        ssim_val = ((2 * mu_q * mu_r + c1) * (2 * sigma_qr + c2)) / ((mu_q**2 + mu_r**2 + c1) * (sigma_q + sigma_r + c2))
        return float(ssim_val)
