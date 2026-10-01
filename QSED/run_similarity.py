"""
Unified Quantum Image Similarity & Matching Runner.

Supports:
1. Pixel-level quantum state overlap / fidelity.
2. SIFT/ORB + Von Neumann Entropy Density Matrix Matching.
3. 4-Directional Cardinal Rotation Search (0°, 90°, 180°, 270°).

Usage:
    python run_similarity.py --img1 path/a.png --img2 path/b.png
    python run_similarity.py --img1 path/a.png --img2 path/b.png --mode sift_entropy --rotate
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Ensure src can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.feature.quantum_sift_entropy import compute_4direction_sift_match, extract_sift_density_matrix, compute_quantum_jsd, compute_mixed_state_fidelity, compute_von_neumann_entropy
from src.feature.entropy_descriptor import (
    compute_image_similarity_entropy_descriptor,
    visualize_matches,
    detect_default_keypoints
)


def run_pixel_similarity(img1_path: str, img2_path: str, resize: int, rotate: bool):
    rotations = [
        ("0 deg", None),
        ("90 deg (CW)", cv2.ROTATE_90_CLOCKWISE),
        ("180 deg", cv2.ROTATE_180),
        ("270 deg (CCW)", cv2.ROTATE_90_COUNTERCLOCKWISE)
    ] if rotate else [("0 deg", None)]

    raw_i1 = cv2.imread(img1_path, 0)
    raw_i2 = cv2.imread(img2_path, 0)

    if raw_i1 is None:
        raise FileNotFoundError(f"Cannot read image: {img1_path}")
    if raw_i2 is None:
        raise FileNotFoundError(f"Cannot read image: {img2_path}")

    i1 = cv2.resize(raw_i1, (resize, resize)) / 255.0 * (np.pi / 2)

    best_fid = -1.0
    best_angle = None
    best_dist = 999.0

    print("\n" + "=" * 60)
    print(f"      QUANTUM PIXEL SIMILARITY RESULTS ({resize}x{resize})")
    print("=" * 60)

    for angle_name, rot_code in rotations:
        r_i2 = raw_i2 if rot_code is None else cv2.rotate(raw_i2, rot_code)
        i2 = cv2.resize(r_i2, (resize, resize)) / 255.0 * (np.pi / 2)
        
        fid = float(np.mean(np.cos(i1 - i2))**2)
        dist = float(np.sqrt(max(0.0, 2 * (1 - np.sqrt(fid)))))

        if rotate:
            print(f"Angle: {angle_name:<15} | Fidelity: {fid * 100:>6.2f}% | Distance: {dist:.4f}")

        if fid > best_fid:
            best_fid = fid
            best_angle = angle_name
            best_dist = dist

    print("-" * 60)
    if rotate:
        print(f"[+] Optimal Orientation     : {best_angle}")
    print(f"[+] Quantum Fidelity (Overlap): {best_fid * 100:.2f}%")
    print(f"[*] Quantum Distance          : {best_dist:.4f}")
    print("=" * 60 + "\n")


def run_sift_entropy_similarity(img1_path: str, img2_path: str, max_kps: int, rotate: bool):
    raw_i1 = cv2.imread(img1_path)
    raw_i2 = cv2.imread(img2_path)

    if raw_i1 is None:
        raise FileNotFoundError(f"Cannot read image: {img1_path}")
    if raw_i2 is None:
        raise FileNotFoundError(f"Cannot read image: {img2_path}")

    print("\n" + "=" * 70)
    print("   QUANTUM SIFT/SURF + VON NEUMANN ENTROPY DENSITY MATRIX MATCHING")
    print("=" * 70)

    if rotate:
        res = compute_4direction_sift_match(raw_i1, raw_i2, max_keypoints=max_kps)
        print(f"[*] Anchor Keypoints: {res['kps_A']} | S(rho_A): {res['entropy_A']:.4f} bits\n")
        print(f"{'Orientation':<16} | {'KPs':<5} | {'S(rho_B)':<9} | {'QJSD':<8} | {'Fidelity':<8} | {'Accuracy':<8}")
        print("-" * 70)
        for r in res["rotations"]:
            print(f"{r['angle']:<16} | {r['kps_B']:<5} | {r['entropy_B']:<9.4f} | {r['qjsd']:<8.4f} | {r['fidelity']*100:<7.2f}% | {r['accuracy']:<6.2f}%")
        
        best = res["best_match"]
        print("=" * 70)
        print(f"[+] Optimal Orientation     : {best['angle_name']}")
        print(f"[+] Quantum State Fidelity  : {best['fidelity'] * 100:.2f}%")
        print(f"[*] Quantum JSD Divergence  : {best['qjsd']:.6f}")
        print(f"[+] Combined Match Accuracy : {best['match_accuracy']:.2f}%")
        print("=" * 70 + "\n")
    else:
        feat_A = extract_sift_density_matrix(raw_i1, max_keypoints=max_kps)
        feat_B = extract_sift_density_matrix(raw_i2, max_keypoints=max_kps)
        
        s_A = compute_von_neumann_entropy(feat_A["density_matrix"])
        s_B = compute_von_neumann_entropy(feat_B["density_matrix"])
        qjsd = compute_quantum_jsd(feat_A["density_matrix"], feat_B["density_matrix"])
        fidelity = compute_mixed_state_fidelity(feat_A["density_matrix"], feat_B["density_matrix"])
        sim_qjsd = max(0.0, (1.0 - np.sqrt(qjsd))) * 100.0
        acc = 0.6 * (fidelity * 100.0) + 0.4 * sim_qjsd

        print(f"[*] Image A Keypoints: {feat_A['num_kps']} | S(rho_A): {s_A:.4f} bits")
        print(f"[*] Image B Keypoints: {feat_B['num_kps']} | S(rho_B): {s_B:.4f} bits")
        print("-" * 70)
        print(f"[+] Quantum State Fidelity  : {fidelity * 100:.2f}%")
        print(f"[*] Quantum JSD Divergence  : {qjsd:.6f}")
        print(f"[+] Combined Match Accuracy : {acc:.2f}%")
        print("=" * 70 + "\n")


def run_local_entropy_24d_similarity(img1_path: str, img2_path: str, max_kps: int = 128, norm: str = "none", ordering: str = "raster", output_dir: str = "outputs/similarity_24d"):
    raw_i1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    raw_i2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if raw_i1 is None:
        raise FileNotFoundError(f"Cannot read image: {img1_path}")
    if raw_i2 is None:
        raise FileNotFoundError(f"Cannot read image: {img2_path}")

    print("\n" + "=" * 70)
    print("   24-D LOCAL VON NEUMANN ENTROPY KEYPOINT SIMILARITY & MATCHING")
    print("=" * 70)

    # Detect keypoints
    kps_1 = detect_default_keypoints(raw_i1, max_keypoints=max_kps)
    kps_2 = detect_default_keypoints(raw_i2, max_keypoints=max_kps)

    res = compute_image_similarity_entropy_descriptor(
        raw_i1, raw_i2, keypoints_a=kps_1, keypoints_b=kps_2, norm=norm, ordering=ordering
    )

    print(f"[*] Image 1 Keypoints detected : {res['num_keypoints_a']}")
    print(f"[*] Image 2 Keypoints detected : {res['num_keypoints_b']}")
    print(f"[*] Corresponding Matches      : {res['num_matches']}")
    print(f"[*] Match Coverage Ratio       : {res['match_ratio'] * 100:.2f}%")
    print(f"[*] Mean Descriptor Distance   : {res['mean_descriptor_distance']:.4f}")
    print("-" * 70)
    print(f"[+] 24-D Entropy Similarity    : {res['similarity_score']:.2f}%")
    print("=" * 70)

    # Expose raw descriptor distances for analysis
    if res['num_matches'] > 0:
        print("\n--- Sample Matched Pairs & Raw Euclidean Distances ---")
        print(f"{'Match #':<8} | {'KP1 (x, y)':<16} | {'KP2 (x, y)':<16} | {'Distance':<10}")
        print("-" * 56)
        for idx, (ia, ib, dist) in enumerate(res['raw_matches'][:10]):
            kpa = res['keypoints_a'][ia]
            kpb = res['keypoints_b'][ib]
            pos_a = f"({kpa.x}, {kpa.y})" if hasattr(kpa, 'x') else f"({kpa[0]}, {kpa[1]})"
            pos_b = f"({kpb.x}, {kpb.y})" if hasattr(kpb, 'x') else f"({kpb[0]}, {kpb[1]})"
            print(f"{idx+1:<8} | {pos_a:<16} | {pos_b:<16} | {dist:<10.4f}")
        if len(res['raw_matches']) > 10:
            print(f"... and {len(res['raw_matches']) - 10} more matches.")
        print("-" * 56 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, "matches_24d_entropy.png")
    visualize_matches(raw_i1, res['keypoints_a'], raw_i2, res['keypoints_b'], res['raw_matches'], save_path=vis_path)
    print(f"[+] Match visualization saved to: {vis_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Quantum Image Similarity & Matching")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image.")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image.")
    parser.add_argument("--mode", type=str, choices=["pixel", "sift_entropy", "local_entropy_24d"], default="sift_entropy",
                        help="Matching mode: 'sift_entropy' (default), 'local_entropy_24d' (24-D local Von Neumann entropy), or 'pixel'.")
    parser.add_argument("--rotate", action="store_true", default=True,
                        help="Perform 4-directional cardinal rotation search (for sift_entropy/pixel modes).")
    parser.add_argument("--resize", type=int, default=1024, help="Resize dimension for pixel mode.")
    parser.add_argument("--kps", type=int, default=128, help="Max keypoints for SIFT / 24-D entropy mode.")
    parser.add_argument("--norm", type=str, choices=["none", "unit_entropy", "l2"], default="none",
                        help="Normalization for 24-D entropy descriptor (default: 'none').")
    parser.add_argument("--ordering", type=str, choices=["raster", "clockwise"], default="raster",
                        help="Position ordering for 24-D descriptor: 'raster' (default) or 'clockwise'.")
    parser.add_argument("--output", type=str, default="outputs/similarity_24d",
                        help="Output directory for visualizations and reports.")

    args = parser.parse_args()

    if args.mode == "pixel":
        run_pixel_similarity(args.img1, args.img2, resize=args.resize, rotate=args.rotate)
    elif args.mode == "local_entropy_24d":
        run_local_entropy_24d_similarity(args.img1, args.img2, max_kps=args.kps, norm=args.norm, ordering=args.ordering, output_dir=args.output)
    else:
        run_sift_entropy_similarity(args.img1, args.img2, max_kps=args.kps, rotate=args.rotate)


if __name__ == "__main__":
    main()
