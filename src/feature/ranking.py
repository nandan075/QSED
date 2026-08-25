from typing import List
from .keypoint_detection import Keypoint

def rank_keypoints(keypoints: List[Keypoint], alpha: float = 0.5) -> List[Keypoint]:
    """
    Compute combined score Q_i and rank keypoints.
    
    Q_i = alpha * K_norm_i + (1 - alpha) * S_norm_i
    
    Args:
        keypoints: List of Keypoint objects.
        alpha: Weight for keypoint score vs entropy.
        
    Returns:
        List[Keypoint]: Ranked list of keypoints.
    """
    if not keypoints:
        return []
        
    max_k = max(kp.keypoint_score for kp in keypoints)
    
    for kp in keypoints:
        k_norm = kp.keypoint_score / max_k if max_k > 0 else 0.0
        s_norm = kp.entropy_normalized
        kp.combined_score = alpha * k_norm + (1.0 - alpha) * s_norm
        
    keypoints.sort(key=lambda kp: kp.combined_score, reverse=True)
    return keypoints
    
def select_top_keypoints(keypoints: List[Keypoint], n: int = 50) -> List[Keypoint]:
    """
    Select top n keypoints.
    
    Args:
        keypoints: Ranked list of keypoints.
        n: Number of keypoints to return.
        
    Returns:
        List[Keypoint]: Top n keypoints.
    """
    return keypoints[:n]
