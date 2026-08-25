import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature.experiments import run_all_experiments, run_worked_example
from src.quantum.qsed import QSEDRunner
from src.classical.image_loader import load_image
from src.feature.directional_variation import compute_directional_variation, compute_keypoint_score
from src.feature.keypoint_detection import detect_keypoints
from src.feature.von_neumann import compute_entropy_map, save_entropy_outputs, von_neumann_entropy
from src.feature.ranking import rank_keypoints
from src.feature.visualization import plot_keypoint_analysis
from src.feature.neighborhood import extract_neighborhood
from src.feature.density_matrix import build_intensity_density_matrix

def main():
    parser = argparse.ArgumentParser(description="Run Keypoint Entropy detection")
    parser.add_argument('--image', type=str, help="Path to input image")
    parser.add_argument('--size', type=int, default=512, help="Image size")
    parser.add_argument('--output', type=str, default='outputs/keypoints/', help="Output directory")
    parser.add_argument('--demo', action='store_true', help="Run worked numerical example")
    parser.add_argument('--experiments', action='store_true', help="Run full experiment suite")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha for combined score")
    parser.add_argument('--neighborhood', type=int, default=3, help="Neighborhood size")
    parser.add_argument('--top-n', type=int, default=50, help="Number of top keypoints to display")
    
    args = parser.parse_args()
    
    if args.demo:
        run_worked_example()
        sys.exit(0)
        
    if args.experiments:
        if not args.image:
            print("Please provide an image for experiments.")
            sys.exit(1)
        image = load_image(args.image, target_size=(args.size, args.size))
        run_all_experiments(image, args.output)
        sys.exit(0)
        
    if not args.image:
        print("Please provide an image.")
        sys.exit(1)
        
    image = load_image(args.image, target_size=(args.size, args.size))
    
    runner = QSEDRunner()
    qsed_results = runner.run_pipeline(image)
    
    grad_mag = qsed_results['gradient_magnitude']
    dom_dir = qsed_results['dominant_direction']
    final_edges = qsed_results['final_edges']
    
    dir_var = compute_directional_variation(dom_dir, grad_mag, args.neighborhood)
    kp_score = compute_keypoint_score(grad_mag, dir_var)
    
    keypoints = detect_keypoints(grad_mag, dom_dir, final_edges, dir_var, kp_score)
    
    entropy_map, entropy_map_norm = compute_entropy_map(image, args.neighborhood)
    
    for kp in keypoints:
        neigh = extract_neighborhood(image, kp.x, kp.y, args.neighborhood)
        rho = build_intensity_density_matrix(neigh)
        S = von_neumann_entropy(rho)
        kp.entropy = S
        kp.entropy_normalized = entropy_map_norm[kp.y, kp.x]
        
    keypoints = rank_keypoints(keypoints, alpha=args.alpha)
    
    save_entropy_outputs(entropy_map, entropy_map_norm, keypoints, args.output)
    
    plot_keypoint_analysis(image, final_edges, dom_dir, dir_var, kp_score, entropy_map, keypoints, args.output)
    
    print(f"Summary:")
    print(f"Detected {len(keypoints)} keypoints.")
    if len(keypoints) > 0:
        top_k = keypoints[:args.top_n]
        print(f"Top {len(top_k)} average entropy: {sum(k.entropy for k in top_k) / len(top_k):.4f}")

if __name__ == "__main__":
    main()
