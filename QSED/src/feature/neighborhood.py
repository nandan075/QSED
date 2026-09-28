import numpy as np
from typing import Dict

def extract_neighborhood(image: np.ndarray, x: int, y: int, size: int = 3) -> np.ndarray:
    """
    Extract size x size patch centered at (x,y) from 2D image.
    
    Args:
        image: 2D input image.
        x: Center x coordinate.
        y: Center y coordinate.
        size: Size of the neighborhood.
        
    Returns:
        np.ndarray: The extracted neighborhood patch.
    """
    h_pad = size // 2
    return image[y - h_pad : y + h_pad + 1, x - h_pad : x + h_pad + 1]

def extract_feature_neighborhood(image: np.ndarray, directional_gradients: Dict[str, np.ndarray], x: int, y: int, size: int = 3) -> np.ndarray:
    """
    Extract multi-channel feature patch.
    
    Args:
        image: 2D grayscale image.
        directional_gradients: Dictionary of 8-direction gradients.
        x: Center x coordinate.
        y: Center y coordinate.
        size: Size of the neighborhood.
        
    Returns:
        np.ndarray: A (size*size, 9) feature array.
    """
    h_pad = size // 2
    
    patch_img = image[y - h_pad : y + h_pad + 1, x - h_pad : x + h_pad + 1].flatten()
    
    keys = ['0', '22.5', '45', '67.5', '90', '112.5', '135', '157.5']
    patches_grad = [directional_gradients[k][y - h_pad : y + h_pad + 1, x - h_pad : x + h_pad + 1].flatten() for k in keys]
    
    feature_vectors = np.column_stack([patch_img] + patches_grad)
    return feature_vectors
