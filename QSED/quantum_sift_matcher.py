"""
Quantum SIFT/SURF Image Feature Matcher with Von Neumann Entropy & Rotational Search

Integrates:
1. 4-Directional (0°, 90°, 180°, 270°) rotational invariance optimization.
2. SIFT/ORB keypoint extraction & response-weighted pure quantum state encoding (7-qubit Hilbert space).
3. Ensemble density matrix construction rho_A and rho_B.
4. Von Neumann Entropy S(rho) and Quantum Jensen-Shannon Divergence (QJSD).
5. Quantum Uhlmann-Jozsa Fidelity between mixed quantum states.
6. Side-by-side visual dashboard with keypoint overlay, 2D entropy maps, metric charts, and eigenvalue spectra.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from scipy.linalg import sqrtm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =====================================================================
# 1. Quantum State & Density Matrix Formulations
# =====================================================================

def extract_features_and_density_matrix(image: np.ndarray, max_keypoints: int = 128) -> dict:
    """
    Extracts SIFT (or ORB fallback) descriptors, maps them to quantum pure states,
    and constructs the ensemble mixed state density matrix rho.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    try:
        detector = cv2.SIFT_create(nfeatures=max_keypoints)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
    except Exception:
        detector = cv2.ORB_create(nfeatures=max_keypoints)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if descriptors is not None:
            padded = np.zeros((descriptors.shape[0], 128), dtype=np.float32)
            padded[:, :descriptors.shape[1]] = descriptors
            descriptors = padded

    dim = 128  # 7-qubit space (2^7 = 128)
    
    if descriptors is None or len(descriptors) == 0:
        rho = np.eye(dim, dtype=np.float64) / dim
        return {
            "keypoints": [],
            "descriptors": None,
            "density_matrix": rho,
            "num_kps": 0
        }

    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    psi = (descriptors / norms).astype(np.float64)

    responses = np.array([kp.response for kp in keypoints], dtype=np.float64)
    if np.sum(responses) == 0 or np.isnan(np.sum(responses)):
        weights = np.ones(len(keypoints), dtype=np.float64) / len(keypoints)
    else:
        weights = responses / np.sum(responses)

    rho = np.dot(psi.T * weights, psi)
    rho = 0.5 * (rho + rho.T)
    trace = np.trace(rho)
    if trace > 0:
        rho = rho / trace
    else:
        rho = np.eye(dim, dtype=np.float64) / dim

    return {
        "keypoints": keypoints,
        "descriptors": psi,
        "density_matrix": rho,
        "num_kps": len(keypoints)
    }


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """Computes Von Neumann Entropy: S(rho) = - sum_k lambda_k * log(lambda_k)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    pos_eig = eigenvalues[eigenvalues > 1e-15]
    if len(pos_eig) == 0:
        return 0.0
    entropy = -np.sum(pos_eig * (np.log(pos_eig) / np.log(base)))
    return float(max(0.0, entropy))


def quantum_jensen_shannon_divergence(rho_A: np.ndarray, rho_B: np.ndarray) -> float:
    """Computes Quantum Jensen-Shannon Divergence (QJSD)."""
    rho_mix = 0.5 * (rho_A + rho_B)
    s_mix = von_neumann_entropy(rho_mix, base=2.0)
    s_a = von_neumann_entropy(rho_A, base=2.0)
    s_b = von_neumann_entropy(rho_B, base=2.0)
    return float(max(0.0, s_mix - 0.5 * (s_a + s_b)))


def quantum_state_fidelity(rho_A: np.ndarray, rho_B: np.ndarray) -> float:
    """Computes Uhlmann-Jozsa Quantum Fidelity between mixed states."""
    try:
        sqrt_A = sqrtm(rho_A)
        if np.iscomplexobj(sqrt_A):
            sqrt_A = sqrt_A.real
        M = sqrt_A @ rho_B @ sqrt_A
        sqrt_M = sqrtm(M)
        if np.iscomplexobj(sqrt_M):
            sqrt_M = sqrt_M.real
        fidelity = float(np.trace(sqrt_M)) ** 2
        return float(np.clip(fidelity, 0.0, 1.0))
    except Exception:
        hs = np.trace(rho_A @ rho_B)
        return float(np.clip(hs, 0.0, 1.0))


def compute_fast_entropy_map(img: np.ndarray, size: int = 128, neighborhood: int = 3) -> np.ndarray:
    """Computes 2D sliding window Von Neumann entropy map for visualization."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray_small = cv2.resize(gray, (size, size)).astype(np.float64)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = neighborhood // 2
    windows = sliding_window_view(gray_small, (neighborhood, neighborhood))
    v = windows.reshape(windows.shape[0], windows.shape[1], -1)
    eps = 1e-10
    q = (v + eps) / np.sum(v + eps, axis=2, keepdims=True)
    q_safe = np.where(q > 0, q, 1.0)
    H = -np.sum(q * np.log2(q_safe), axis=2)
    emap = np.zeros((size, size), dtype=np.float64)
    emap[pad:size-pad, pad:size-pad] = H
    return emap


# =====================================================================
# 2. 4-Directional Rotational Quantum Image Matcher
# =====================================================================

def match_images_4direction(img_A: np.ndarray, img_B: np.ndarray, max_keypoints: int = 128) -> dict:
    """
    Matches Image A against Image B across 4 cardinal orientations.
    """
    feat_A = extract_features_and_density_matrix(img_A, max_keypoints=max_keypoints)
    rho_A = feat_A["density_matrix"]
    s_A = von_neumann_entropy(rho_A, base=2.0)

    rotations = [
        ("0 deg", None),
        ("90 deg (CW)", cv2.ROTATE_90_CLOCKWISE),
        ("180 deg", cv2.ROTATE_180),
        ("270 deg (CCW)", cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    best_match = {
        "angle_name": None,
        "match_accuracy": -1.0,
        "qjsd": 1.0,
        "fidelity": 0.0,
        "entropy_B": 0.0,
        "rotated_image": None,
        "feat_B": None
    }
    
    rotation_results = []

    for angle_name, rot_code in rotations:
        rot_img_B = img_B.copy() if rot_code is None else cv2.rotate(img_B, rot_code)
        feat_B = extract_features_and_density_matrix(rot_img_B, max_keypoints=max_keypoints)
        rho_B = feat_B["density_matrix"]
        s_B = von_neumann_entropy(rho_B, base=2.0)

        qjsd = quantum_jensen_shannon_divergence(rho_A, rho_B)
        fidelity = quantum_state_fidelity(rho_A, rho_B)
        sim_qjsd = max(0.0, (1.0 - np.sqrt(qjsd))) * 100.0
        combined_accuracy = 0.6 * (fidelity * 100.0) + 0.4 * sim_qjsd

        res = {
            "angle": angle_name,
            "kps_B": feat_B["num_kps"],
            "entropy_B": s_B,
            "qjsd": qjsd,
            "fidelity": fidelity,
            "sim_qjsd": sim_qjsd,
            "accuracy": combined_accuracy
        }
        rotation_results.append(res)

        if combined_accuracy > best_match["match_accuracy"]:
            best_match["angle_name"] = angle_name
            best_match["match_accuracy"] = combined_accuracy
            best_match["qjsd"] = qjsd
            best_match["fidelity"] = fidelity
            best_match["entropy_B"] = s_B
            best_match["rotated_image"] = rot_img_B
            best_match["feat_B"] = feat_B

    return {
        "entropy_A": s_A,
        "kps_A": feat_A["num_kps"],
        "feat_A": feat_A,
        "rotations": rotation_results,
        "best_match": best_match
    }


# =====================================================================
# 3. Comprehensive Visual Report Generation
# =====================================================================

def plot_comprehensive_match_report(
    img_A: np.ndarray,
    img_B_best: np.ndarray,
    feat_A: dict,
    feat_B_best: dict,
    results: dict,
    save_path: str
):
    """
    Renders a 6-panel side-by-side dashboard comparing images, entropy maps, metrics, and spectra.
    """
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Quantum Computer Vision: Feature Matching & Entropy Analysis Report", fontsize=16, fontweight="bold", y=0.98)
    
    # Panel 1: Image A with SIFT Keypoints
    ax1 = plt.subplot(2, 3, 1)
    img_A_rgb = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB) if len(img_A.shape) == 3 else cv2.cvtColor(img_A, cv2.COLOR_GRAY2RGB)
    kp_img_A = cv2.drawKeypoints(img_A_rgb, feat_A["keypoints"][:60], None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax1.imshow(kp_img_A)
    ax1.set_title(f"Image A (Anchor) - {feat_A['num_kps']} Keypoints", fontsize=12, fontweight="bold")
    ax1.axis('off')
    
    # Panel 2: Image B (Optimal Orientation) with SIFT Keypoints
    ax2 = plt.subplot(2, 3, 2)
    img_B_rgb = cv2.cvtColor(img_B_best, cv2.COLOR_BGR2RGB) if len(img_B_best.shape) == 3 else cv2.cvtColor(img_B_best, cv2.COLOR_GRAY2RGB)
    kp_img_B = cv2.drawKeypoints(img_B_rgb, feat_B_best["keypoints"][:60], None, color=(255, 128, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax2.imshow(kp_img_B)
    best_angle = results["best_match"]["angle_name"]
    ax2.set_title(f"Image B ({best_angle}) - {feat_B_best['num_kps']} Keypoints", fontsize=12, fontweight="bold")
    ax2.axis('off')
    
    # Panel 3: 4-Direction Match Accuracy & Fidelity Bar Chart
    ax3 = plt.subplot(2, 3, 3)
    angles = [r["angle"] for r in results["rotations"]]
    accs = [r["accuracy"] for r in results["rotations"]]
    fids = [r["fidelity"] * 100.0 for r in results["rotations"]]
    x = np.arange(len(angles))
    width = 0.35
    b1 = ax3.bar(x - width/2, accs, width, label='Match Accuracy (%)', color='#2b5c8f', edgecolor='black')
    b2 = ax3.bar(x + width/2, fids, width, label='Quantum Fidelity (%)', color='#2a9d8f', edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(angles, rotation=15, fontsize=9)
    ax3.set_ylim(0, 115)
    ax3.set_ylabel("Score (%)", fontsize=10)
    ax3.set_title("4-Directional Invariance Search", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8)
    for bar in b1:
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"{y:.1f}%", ha='center', va='bottom', fontsize=8)
    for bar in b2:
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, y + 1.5, f"{y:.1f}%", ha='center', va='bottom', fontsize=8)

    # Panel 4: Image A Dense 2D Von Neumann Entropy Map
    ax4 = plt.subplot(2, 3, 4)
    emap_A = compute_fast_entropy_map(img_A)
    im4 = ax4.imshow(emap_A, cmap='viridis')
    ax4.set_title(f"Image A Entropy Map S(x,y)\nAvg S(rho_A): {results['entropy_A']:.4f} bits", fontsize=11, fontweight="bold")
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Panel 5: Image B Dense 2D Von Neumann Entropy Map
    ax5 = plt.subplot(2, 3, 5)
    emap_B = compute_fast_entropy_map(img_B_best)
    im5 = ax5.imshow(emap_B, cmap='viridis')
    s_B_val = results["best_match"]["entropy_B"]
    ax5.set_title(f"Image B ({best_angle}) Entropy Map\nAvg S(rho_B): {s_B_val:.4f} bits", fontsize=11, fontweight="bold")
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Panel 6: Density Matrix Top Eigenvalue Spectra
    ax6 = plt.subplot(2, 3, 6)
    eig_A = np.linalg.eigvalsh(feat_A["density_matrix"])
    eig_B = np.linalg.eigvalsh(feat_B_best["density_matrix"])
    eig_A = np.sort(eig_A)[::-1][:20]
    eig_B = np.sort(eig_B)[::-1][:20]
    ax6.plot(range(1, len(eig_A)+1), eig_A, 'o-', color='#2b5c8f', label=f'Image A Eigenvalues (S={results["entropy_A"]:.2f})', linewidth=2)
    ax6.plot(range(1, len(eig_B)+1), eig_B, 's--', color='#e76f51', label=f'Image B Eigenvalues (S={s_B_val:.2f})', linewidth=2)
    ax6.set_title("Density Matrix Top Eigenvalue Spectra (lambda_k)", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Eigenvalue Rank k (7-Qubit Space)")
    ax6.set_ylabel("Probability lambda_k")
    ax6.grid(True, linestyle="--", alpha=0.5)
    ax6.legend(fontsize=8)

    # Summary Card Box at Bottom
    best = results["best_match"]
    summary_text = (
        f"QUANTUM MATCHING METRICS SUMMARY:\n"
        f"  - Optimal Orientation     : {best['angle_name']}\n"
        f"  - Quantum State Fidelity  : {best['fidelity']*100:.2f}%\n"
        f"  - Quantum JSD Divergence  : {best['qjsd']:.6f}\n"
        f"  - Final Match Accuracy    : {best['match_accuracy']:.2f}%"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#2b5c8f', linewidth=1.5))
             
    plt.subplots_adjust(top=0.92, bottom=0.12, hspace=0.25, wspace=0.20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================================
# 4. CLI Execution Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Image Matching with 4-Direction Rotation and Von Neumann Entropy."
    )
    parser.add_argument("--img1", type=str, default=None, help="Path to reference/anchor image A.")
    parser.add_argument("--img2", type=str, default=None, help="Path to query image B.")
    parser.add_argument("--kps", type=int, default=128, help="Maximum keypoints for quantum state mapping (default: 128).")
    parser.add_argument("--size", type=int, default=None, help="Resize dimension (e.g. 1024) before SIFT extraction (optional).")
    parser.add_argument("--demo", action="store_true", help="Generate demo images and run synthetic benchmark test.")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save accuracy breakdown plot (optional).")

    args = parser.parse_args()

    if args.demo or (args.img1 is None and args.img2 is None):
        print("\n[+] Running in DEMO mode with generated geometric test pair...")
        os.makedirs("demo_images", exist_ok=True)
        img_A = np.zeros((300, 300, 3), dtype=np.uint8) + 240
        cv2.circle(img_A, (90, 90), 45, (200, 40, 40), -1)
        cv2.rectangle(img_A, (170, 70), (250, 150), (40, 180, 40), -1)
        pts = np.array([[150, 190], [80, 270], [220, 270]], np.int32)
        cv2.fillPoly(img_A, [pts], (40, 40, 200))
        cv2.putText(img_A, "QUANTUM CV", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        img_B = cv2.rotate(img_A, cv2.ROTATE_90_CLOCKWISE)
        img1_path, img2_path = "demo_images/image_anchor.png", "demo_images/image_rotated_test.png"
        cv2.imwrite(img1_path, img_A)
        cv2.imwrite(img2_path, img_B)
    else:
        img1_path, img2_path = args.img1, args.img2

    if not os.path.exists(img1_path):
        print(f"[-] Error: Image path '{img1_path}' does not exist.")
        sys.exit(1)
    if not os.path.exists(img2_path):
        print(f"[-] Error: Image path '{img2_path}' does not exist.")
        sys.exit(1)

    img_A = cv2.imread(img1_path)
    img_B = cv2.imread(img2_path)

    if args.size is not None:
        img_A = cv2.resize(img_A, (args.size, args.size), interpolation=cv2.INTER_AREA)
        img_B = cv2.resize(img_B, (args.size, args.size), interpolation=cv2.INTER_AREA)
        print(f"[*] Resized both images to {args.size}x{args.size} resolution.")

    print("\n" + "=" * 70)
    print("      QUANTUM IMAGE MATCHING WITH 4-DIRECTION ROTATION & ENTROPY      ")
    print("=" * 70)
    print(f"Image A (Anchor) : {img1_path} ({img_A.shape[1]}x{img_A.shape[0]})")
    print(f"Image B (Query)  : {img2_path} ({img_B.shape[1]}x{img_B.shape[0]})")
    print(f"Max Quantum Keypoints: {args.kps} (7-Qubit Hilbert Space)\n")

    results = match_images_4direction(img_A, img_B, max_keypoints=args.kps)

    print(f"[*] Anchor Image A - Keypoints: {results['kps_A']} | Entropy S(rho_A): {results['entropy_A']:.4f} bits\n")
    print("------------------------- 4-DIRECTION RESULTS -------------------------")
    print(f"{'Orientation':<16} | {'KPs':<5} | {'S(rho_B)':<9} | {'QJSD':<8} | {'Fidelity':<8} | {'Match Accuracy':<14}")
    print("-" * 72)
    for r in results["rotations"]:
        print(f"{r['angle']:<16} | {r['kps_B']:<5} | {r['entropy_B']:<9.4f} | {r['qjsd']:<8.4f} | {r['fidelity']*100:<7.2f}% | {r['accuracy']:<6.2f}%")

    best = results["best_match"]
    print("=" * 72)
    print(f"[+] BEST MATCH ORIENTATION : {best['angle_name']}")
    print(f"[*] QUANTUM STATE FIDELITY : {best['fidelity'] * 100.0:.2f}%")
    print(f"[*] MINIMUM QJSD DIVERGENCE: {best['qjsd']:.6f}")
    print(f"[+] FINAL MATCH ACCURACY   : {best['match_accuracy']:.2f}%")
    print("=" * 72 + "\n")

    if args.save_plot:
        plot_comprehensive_match_report(
            img_A=img_A,
            img_B_best=best["rotated_image"],
            feat_A=results["feat_A"],
            feat_B_best=best["feat_B"],
            results=results,
            save_path=args.save_plot
        )
        print(f"[+] Comprehensive visual report saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
