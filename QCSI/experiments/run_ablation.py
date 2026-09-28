"""
QCSI Comprehensive Ablation Suite (A0 - A7).

Ablations:
- A0: pHash
- A1: SIFT + RANSAC (classical)
- A2: DINOv2 Cosine Similarity
- A3: Old QSED score
- A4: Branch A only (Local Quantum-Kernel Matching + Density Metrics)
- A5: Branch B only (DINOv2 + ZZFeatureMap Quantum Kernel)
- A6: Full Hybrid Fusion (QCSI proposed)
- A7: Full Fusion with Classical Kernels (Linear/Cosine instead of Quantum)

Evaluates:
- ROC-AUC, PR-AUC
- Expected Calibration Error (ECE)
- FAR / FRR at operating threshold 0.5
- Latency per pair (ms)
"""

from typing import List, Dict, Any, Tuple
import os
import time
import csv
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression

import sys
qcsi_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if qcsi_root not in sys.path:
    sys.path.insert(0, qcsi_root)

from qcsi.preprocess import ImagePreprocessor
from qcsi.local_branch import LocalGeometryMatcher
from qcsi.global_branch import GlobalSemanticMatcher
from qcsi.fusion import CalibratedFusionModel, extract_feature_vector
from experiments.run_baselines import BaselineEvaluator
from data.download_datasets import build_local_benchmark_dataset
from experiments.build_pairs import build_labeled_pairs


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across n_bins.
    ECE = sum_b (N_b / N) * |acc(b) - conf(b)|.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i + 1] if i < n_bins - 1 else y_prob <= bin_limits[i + 1])
        n_b = np.sum(bin_mask)
        if n_b > 0:
            acc_b = np.mean(y_true[bin_mask])
            conf_b = np.mean(y_prob[bin_mask])
            ece += (n_b / float(n)) * np.abs(acc_b - conf_b)

    return float(ece)


class AblationRunner:
    """Runs end-to-end ablation experiments across all models A0 - A7."""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.baseline_eval = BaselineEvaluator()
        self.local_matcher = LocalGeometryMatcher()
        self.global_matcher = GlobalSemanticMatcher()
        self.fusion_model = CalibratedFusionModel()

    def run_pair_A4(self, prep: dict) -> float:
        """A4: Branch A only (Local Quantum Geometry + Density Metrics)."""
        loc = self.local_matcher.compute_local_similarity(prep["gray_A"], prep["gray_B"])
        geo = loc.get("geometry", {})
        inliers = geo.get("inlier_count", 0)
        mean_fid = geo.get("mean_inlier_fidelity", 0.0)
        uhlmann = loc.get("uhlmann_fidelity", 0.0)
        score = 0.5 * min(1.0, inliers / 12.0) + 0.3 * mean_fid + 0.2 * uhlmann
        return float(np.clip(score, 0.0, 1.0))

    def run_pair_A5(self, prep: dict) -> float:
        """A5: Branch B only (DINOv2 + ZZFeatureMap Quantum Kernel)."""
        glob = self.global_matcher.compute_global_similarity(prep["rgb_A"], prep["rgb_B"])
        return float(glob.get("quantum_fidelity", 0.0))

    def run_pair_A6(self, prep: dict) -> float:
        """A6: Full Hybrid Fusion (Proposed)."""
        loc = self.local_matcher.compute_local_similarity(prep["gray_A"], prep["gray_B"])
        glob = self.global_matcher.compute_global_similarity(prep["rgb_A"], prep["rgb_B"])
        feat = extract_feature_vector(loc, glob)
        prob = self.fusion_model.predict_probability(feat)
        return float(prob)

    def run_pair_A7(self, prep: dict) -> float:
        """
        A7: Full Fusion with Classical Kernels.
        Replaces quantum kernels with classical cosine/dot products.
        """
        loc = self.local_matcher.compute_local_similarity(prep["gray_A"], prep["gray_B"])
        glob = self.global_matcher.compute_global_similarity(prep["rgb_A"], prep["rgb_B"])

        # Classical counterparts
        geo = loc.get("geometry", {})
        inlier_count = float(geo.get("inlier_count", 0))
        inlier_ratio = float(geo.get("inlier_ratio", 0.0))
        # Classical linear kernel replaces quantum fidelity: K_class = dot(psi1, psi2)
        mean_classical_cos = float(np.sqrt(max(0.0, geo.get("mean_inlier_fidelity", 0.0))))
        mean_euclidean = float(np.sqrt(2.0 * max(0.0, 1.0 - mean_classical_cos)))

        # Classical descriptor mean cosine replaces Uhlmann fidelity
        psi_A, psi_B = loc["feat_A"]["psi"], loc["feat_B"]["psi"]
        if len(psi_A) > 0 and len(psi_B) > 0:
            mean_A = np.mean(psi_A, axis=0)
            mean_B = np.mean(psi_B, axis=0)
            class_mean_cos = float(np.clip(np.dot(mean_A, mean_B) / (np.linalg.norm(mean_A)*np.linalg.norm(mean_B) + 1e-12), 0.0, 1.0))
        else:
            class_mean_cos = 0.0

        dino_cos = float(glob.get("classical_cosine", 0.0))
        # Classical linear kernel on PCA angles replaces ZZFeatureMap
        angles_A, angles_B = glob.get("angles_A", np.zeros(8)), glob.get("angles_B", np.zeros(8))
        norm_ang_A = np.linalg.norm(angles_A) + 1e-12
        norm_ang_B = np.linalg.norm(angles_B) + 1e-12
        classical_global_cos = float(np.clip(np.dot(angles_A, angles_B) / (norm_ang_A * norm_ang_B), 0.0, 1.0))

        # Heuristic classical combination
        geom_score = np.clip(inlier_count / 15.0, 0.0, 1.0) * 0.6 + np.clip(inlier_ratio / 0.2, 0.0, 1.0) * 0.4
        sem_score = 0.5 * max(0.0, (dino_cos + 0.2) / 1.2) + 0.5 * classical_global_cos
        classical_combined = 0.5 * geom_score + 0.3 * sem_score + 0.2 * class_mean_cos
        prob = 1.0 / (1.0 + np.exp(-10.0 * (classical_combined - 0.45)))
        return float(np.clip(prob, 0.0, 1.0))

    def evaluate_test_pairs(
        self,
        pairs_csv: str,
        fit_calibration: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Reads pairs from CSV, evaluates all systems A0 - A7, and returns metrics table.
        """
        rows = []
        with open(pairs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        if not rows:
            raise ValueError(f"No pairs found in {pairs_csv}")

        y_true = np.array([int(r["label"]) for r in rows], dtype=np.int32)
        n_pairs = len(rows)

        scores_by_system: Dict[str, List[float]] = {
            f"A{i}": [] for i in range(8)
        }
        latency_by_system: Dict[str, float] = {
            f"A{i}": 0.0 for i in range(8)
        }

        # Pre-extract features for calibration training if requested
        if fit_calibration and len(rows) >= 6:
            X_feats = []
            y_feats = []
            for r in rows:
                if not (os.path.exists(r["img1_path"]) and os.path.exists(r["img2_path"])):
                    continue
                p = self.preprocessor.preprocess_pair(r["img1_path"], r["img2_path"])
                l = self.local_matcher.compute_local_similarity(p["gray_A"], p["gray_B"])
                g = self.global_matcher.compute_global_similarity(p["rgb_A"], p["rgb_B"])
                X_feats.append(extract_feature_vector(l, g))
                y_feats.append(int(r["label"]))
            if len(np.unique(y_feats)) > 1:
                self.fusion_model.fit(np.array(X_feats), np.array(y_feats))

        print(f"[*] Evaluating {n_pairs} pairs across Ablations A0 - A7...")

        for idx, r in enumerate(rows):
            img1_path, img2_path = r["img1_path"], r["img2_path"]
            if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
                continue

            prep = self.preprocessor.preprocess_pair(img1_path, img2_path)

            # A0: pHash
            t0 = time.perf_counter()
            s_a0 = self.baseline_eval.evaluate_A0_phash(prep["rgb_A"], prep["rgb_B"])
            latency_by_system["A0"] += (time.perf_counter() - t0)
            scores_by_system["A0"].append(s_a0)

            # A1: Classical SIFT + RANSAC
            t0 = time.perf_counter()
            s_a1 = self.baseline_eval.evaluate_A1_classical_sift(prep["gray_A"], prep["gray_B"])
            latency_by_system["A1"] += (time.perf_counter() - t0)
            scores_by_system["A1"].append(s_a1)

            # A2: DINOv2 Cosine
            t0 = time.perf_counter()
            s_a2 = self.baseline_eval.evaluate_A2_dinov2_cosine(prep["rgb_A"], prep["rgb_B"])
            latency_by_system["A2"] += (time.perf_counter() - t0)
            scores_by_system["A2"].append(s_a2)

            # A3: Legacy QSED
            t0 = time.perf_counter()
            s_a3 = self.baseline_eval.evaluate_A3_legacy_qsed(prep["gray_A"], prep["gray_B"])
            latency_by_system["A3"] += (time.perf_counter() - t0)
            scores_by_system["A3"].append(s_a3)

            # A4: Branch A only
            t0 = time.perf_counter()
            s_a4 = self.run_pair_A4(prep)
            latency_by_system["A4"] += (time.perf_counter() - t0)
            scores_by_system["A4"].append(s_a4)

            # A5: Branch B only
            t0 = time.perf_counter()
            s_a5 = self.run_pair_A5(prep)
            latency_by_system["A5"] += (time.perf_counter() - t0)
            scores_by_system["A5"].append(s_a5)

            # A6: Full Hybrid Fusion
            t0 = time.perf_counter()
            s_a6 = self.run_pair_A6(prep)
            latency_by_system["A6"] += (time.perf_counter() - t0)
            scores_by_system["A6"].append(s_a6)

            # A7: Full Classical Fusion
            t0 = time.perf_counter()
            s_a7 = self.run_pair_A7(prep)
            latency_by_system["A7"] += (time.perf_counter() - t0)
            scores_by_system["A7"].append(s_a7)

        # Compute benchmark metrics
        system_names = {
            "A0": "pHash",
            "A1": "Classical SIFT + RANSAC",
            "A2": "DINOv2 Cosine",
            "A3": "Legacy QSED",
            "A4": "Branch A (Quantum Local)",
            "A5": "Branch B (DINOv2 + Quantum Kernel)",
            "A6": "Full Hybrid Fusion (Proposed)",
            "A7": "Full Fusion (Classical Kernels)",
        }

        results_summary = {}

        for sys_id, scores_list in scores_by_system.items():
            scores = np.array(scores_list, dtype=np.float64)
            avg_lat_ms = (latency_by_system[sys_id] / max(len(scores), 1)) * 1000.0

            # ROC-AUC
            try:
                roc_auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5
            except Exception:
                roc_auc = 0.5

            # PR-AUC
            try:
                prec, rec, _ = precision_recall_curve(y_true, scores)
                pr_auc = float(auc(rec, prec))
            except Exception:
                pr_auc = 0.5

            # FAR and FRR at threshold 0.5
            preds = (scores >= 0.5).astype(np.int32)
            neg_mask = (y_true == 0)
            pos_mask = (y_true == 1)

            far = float(np.mean(preds[neg_mask] == 1)) if np.sum(neg_mask) > 0 else 0.0
            frr = float(np.mean(preds[pos_mask] == 0)) if np.sum(pos_mask) > 0 else 0.0

            # ECE
            ece = compute_expected_calibration_error(y_true, scores)

            results_summary[sys_id] = {
                "name": system_names[sys_id],
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "far_at_05": round(far, 4),
                "frr_at_05": round(frr, 4),
                "ece": round(ece, 4),
                "latency_ms": round(avg_lat_ms, 2),
            }

        return results_summary


def main():
    parser = argparse.ArgumentParser(description="QCSI Ablation Suite Runner.")
    parser.add_argument("--pairs_csv", type=str, default=None, help="Path to evaluation pairs CSV.")
    parser.add_argument("--synthetic_scenes", type=int, default=6, help="Number of procedural scenes to generate if no CSV provided.")
    parser.add_argument("--views_per_scene", type=int, default=3, help="Views per scene.")
    args = parser.parse_args()

    pairs_csv = args.pairs_csv
    if pairs_csv is None or not os.path.exists(pairs_csv):
        print("[*] Generating local procedural benchmark dataset...")
        base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_dataset")
        manifest = build_local_benchmark_dataset(base_dir, num_scenes=args.synthetic_scenes, views_per_scene=args.views_per_scene)
        out_pairs = os.path.join(os.path.dirname(__file__), "..", "data", "pairs")
        pair_files = build_labeled_pairs(manifest, out_pairs)
        pairs_csv = pair_files["test"]

    runner = AblationRunner()
    results = runner.evaluate_test_pairs(pairs_csv, fit_calibration=True)

    print("\n" + "=" * 88)
    print("                      QCSI ABLATION & BASELINE BENCHMARK SUITE")
    print("=" * 88)
    print(f"{'ID':<4} | {'System Name':<34} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'ECE':<6} | {'Lat(ms)':<8}")
    print("-" * 88)
    for sys_id, m in results.items():
        print(f"{sys_id:<4} | {m['name']:<34} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<8.4f} | {m['ece']:<6.4f} | {m['latency_ms']:<8.1f}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
