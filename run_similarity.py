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


def main():
    parser = argparse.ArgumentParser(description="Quantum Image Similarity & Matching")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image.")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image.")
    parser.add_argument("--mode", type=str, choices=["pixel", "sift_entropy"], default="sift_entropy",
                        help="Matching mode: 'sift_entropy' (default) or 'pixel'.")
    parser.add_argument("--rotate", action="store_true", default=True,
                        help="Perform 4-directional cardinal rotation search (default: True).")
    parser.add_argument("--resize", type=int, default=1024, help="Resize dimension for pixel mode.")
    parser.add_argument("--kps", type=int, default=128, help="Max keypoints for SIFT entropy mode.")

    args = parser.parse_args()

    if args.mode == "pixel":
        run_pixel_similarity(args.img1, args.img2, resize=args.resize, rotate=args.rotate)
    else:
        run_sift_entropy_similarity(args.img1, args.img2, max_kps=args.kps, rotate=args.rotate)


if __name__ == "__main__":
    main()
