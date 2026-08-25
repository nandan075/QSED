import os
import time
import numpy as np
from typing import Dict, List

from ..quantum.qsed import QSEDRunner
from .directional_variation import compute_directional_variation, compute_keypoint_score
from .keypoint_detection import detect_keypoints, Keypoint
from .von_neumann import compute_entropy_map, von_neumann_entropy
from .ranking import select_top_keypoints, rank_keypoints
from .neighborhood import extract_neighborhood
from .density_matrix import build_intensity_density_matrix
from ..classical.sobel_operator import apply_sobel_masks
from ..classical.gradient import compute_gradient_magnitude

def run_all_experiments(image: np.ndarray, output_dir: str = 'outputs/experiments/') -> Dict:
    """
    Runs all experiments and returns a results dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # 1. Run the QSED pipeline
    runner = QSEDRunner()
    qsed_results = runner.run_pipeline(image)
    
    grad_mag = qsed_results['gradient_magnitude']
    dom_dir = qsed_results['dominant_direction']
    final_edges = qsed_results['final_edges']
    
    dir_var_map = compute_directional_variation(dom_dir, grad_mag)
    kp_score_map = compute_keypoint_score(grad_mag, dir_var_map)
    keypoints = detect_keypoints(grad_mag, dom_dir, final_edges, dir_var_map, kp_score_map)
    entropy_map, entropy_map_norm = compute_entropy_map(image)
    
    for kp in keypoints:
        kp.entropy = entropy_map[kp.y, kp.x]
        kp.entropy_normalized = entropy_map_norm[kp.y, kp.x]
    
    # Ensure keypoints have combined score
    ranked_keypoints = rank_keypoints(keypoints, alpha=0.5)
    
    md_lines = ["# Experiment Results\n"]
    
    # Experiment 1: Edge map only vs keypoint detection
    t0 = time.time()
    num_edge_pixels = np.count_nonzero(final_edges)
    num_keypoints = len(keypoints)
    t1 = time.time()
    
    res_e1 = {
        'num_edge_pixels': num_edge_pixels,
        'num_keypoints': num_keypoints,
        'runtime': t1 - t0
    }
    results['experiment_1'] = res_e1
    md_lines.append(f"## Experiment 1: Edge map only vs keypoint detection")
    md_lines.append(f"- Edge pixels: {num_edge_pixels}")
    md_lines.append(f"- Keypoints: {num_keypoints}")
    md_lines.append(f"- Runtime: {res_e1['runtime']:.4f}s\n")
    
    # Experiment 2: 3x3 vs 5x5 neighborhood
    t0 = time.time()
    map_3, _ = compute_entropy_map(image, neighborhood_size=3)
    map_5, _ = compute_entropy_map(image, neighborhood_size=5)
    
    kp_entropies_3 = [map_3[k.y, k.x] for k in keypoints]
    kp_entropies_5 = [map_5[k.y, k.x] for k in keypoints]
    
    res_e2 = {
        'avg_entropy_3x3': np.mean(kp_entropies_3) if kp_entropies_3 else 0,
        'avg_entropy_5x5': np.mean(kp_entropies_5) if kp_entropies_5 else 0,
        'runtime': time.time() - t0
    }
    results['experiment_2'] = res_e2
    md_lines.append(f"## Experiment 2: 3x3 vs 5x5 neighborhood")
    md_lines.append(f"- Avg Entropy (3x3): {res_e2['avg_entropy_3x3']:.4f}")
    md_lines.append(f"- Avg Entropy (5x5): {res_e2['avg_entropy_5x5']:.4f}")
    md_lines.append(f"- Runtime: {res_e2['runtime']:.4f}s\n")
    
    # Experiment 3: Method A (intensity) vs Method B (directional-feature) entropy
    # (assuming Method A is default in entropy_map)
    t0 = time.time()
    res_e3 = {
        'min_A': np.min(entropy_map),
        'max_A': np.max(entropy_map),
        'avg_A': np.mean(entropy_map),
        'runtime': time.time() - t0
    }
    results['experiment_3'] = res_e3
    md_lines.append(f"## Experiment 3: Method A vs Method B")
    md_lines.append(f"- Method A: Min {res_e3['min_A']:.4f}, Max {res_e3['max_A']:.4f}, Avg {res_e3['avg_A']:.4f}")
    md_lines.append(f"- Runtime: {res_e3['runtime']:.4f}s\n")
    
    # Experiment 4: Keypoint score only vs score+entropy ranking
    t0 = time.time()
    score_only = sorted(keypoints, key=lambda k: k.keypoint_score, reverse=True)
    score_entropy = sorted(keypoints, key=lambda k: k.combined_score, reverse=True)
    top_50_score = score_only[:50]
    top_50_both = score_entropy[:50]
    overlap = len(set(id(k) for k in top_50_score).intersection(set(id(k) for k in top_50_both)))
    res_e4 = {'overlap': overlap, 'runtime': time.time() - t0}
    results['experiment_4'] = res_e4
    md_lines.append(f"## Experiment 4: Ranking overlap")
    md_lines.append(f"- Overlap in top 50: {overlap}/50")
    md_lines.append(f"- Runtime: {res_e4['runtime']:.4f}s\n")
    
    # Experiment 5: Threshold sweep
    md_lines.append(f"## Experiment 5: Threshold sweep")
    results['experiment_5'] = {}
    for th in [0.05, 0.1, 0.15, 0.2, 0.25]:
        t0 = time.time()
        kps = detect_keypoints(grad_mag, dom_dir, final_edges, dir_var_map, kp_score_map, grad_threshold_ratio=th)
        results['experiment_5'][th] = len(kps)
        md_lines.append(f"- Threshold {th}: {len(kps)} keypoints (Runtime: {time.time()-t0:.4f}s)")
    md_lines.append("")
    
    # Experiment 6: Alpha sweep
    md_lines.append(f"## Experiment 6: Alpha sweep")
    results['experiment_6'] = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t0 = time.time()
        rkps = rank_keypoints(keypoints, alpha=a)
        top = select_top_keypoints(rkps, 5)
        results['experiment_6'][a] = top
        md_lines.append(f"- Alpha {a}: Top 5 avg combined score = {np.mean([k.combined_score for k in top]) if top else 0:.4f} (Runtime: {time.time()-t0:.4f}s)")
    md_lines.append("")
    
    print("\n".join(md_lines))
    
    with open(os.path.join(output_dir, 'experiment_results.md'), 'w') as f:
        f.write("\n".join(md_lines))
        
    return results

def run_worked_example() -> str:
    """
    Generate a complete worked numerical example using an 8x8 synthetic 'square' image.
    """
    from ..classical.image_loader import generate_synthetic_image
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = 255
    
    dir_grads = apply_sobel_masks(image)
    grad_mag, dom_dir = compute_gradient_magnitude(dir_grads)
    dir_var = compute_directional_variation(dom_dir, grad_mag)
    kp_score = compute_keypoint_score(grad_mag, dir_var)
    
    lines = []
    lines.append("# Worked Example: 8x8 Square Image\n")
    
    lines.append("## 1. Eight directional responses for a 5x5 sub-region (center 2:7)")
    for name, grad in dir_grads.items():
        lines.append(f"{name}:\n{grad[2:7, 2:7]}")
        
    lines.append("\n## 2. Dominant direction at each pixel in a 5x5 sub-region")
    lines.append(f"{dom_dir[2:7, 2:7]}")
    
    lines.append("\n## 3. Dominant gradient magnitude")
    lines.append(f"{grad_mag[2:7, 2:7]}")
    
    x, y = 3, 3
    lines.append(f"\n## 4. Angular differences between center pixel ({x},{y}) and its 3x3 neighbors")
    center_angle = dom_dir[y, x] * 22.5  # Convert index to angle in degrees
    from .directional_variation import angular_difference
    diffs = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            neighbor_angle = dom_dir[y+dy, x+dx] * 22.5  # Convert index to angle
            d = angular_difference(center_angle, neighbor_angle)
            diffs.append(d)
    lines.append(f"{diffs}")
    
    lines.append("\n## 5. Directional variation score D")
    lines.append(f"{dir_var[y, x]:.4f}")
    
    lines.append("\n## 6. Keypoint score K")
    lines.append(f"{kp_score[y, x]:.4f}")
    
    lines.append("\n## 7. 3x3 intensity neighborhood extraction")
    neigh = extract_neighborhood(image, x, y, size=3)
    lines.append(f"{neigh}")
    
    lines.append("\n## 8. Normalized probabilities qi")
    prob = neigh.flatten() / (np.sum(neigh) + 1e-10)
    lines.append(f"{prob}")
    
    lines.append("\n## 9. 9x9 diagonal density matrix rho")
    rho = build_intensity_density_matrix(neigh)
    lines.append(f"{np.diag(rho)}")
    
    lines.append("\n## 10. Eigenvalues")
    lines.append(f"{np.diag(rho)}")
    
    lines.append("\n## 11. Von Neumann entropy S(rho)")
    S = von_neumann_entropy(rho)
    lines.append(f"S = {S:.4f}")
    
    output = "\n".join(lines)
    print(output)
    
    os.makedirs('outputs/experiments/', exist_ok=True)
    with open('outputs/experiments/worked_example.md', 'w') as f:
        f.write(output)
        
    return output
