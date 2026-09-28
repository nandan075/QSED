# QCSI: Hybrid Quantum-Classical Image Similarity

A high-precision, calibrated image similarity framework combining local quantum geometry with global semantic representation.

## Architecture

```
                         Image A            Image B
                            │                  │
                     ┌──────┴──────────────────┴──────┐
                     │ Preprocess: EXIF fix, resize,  │
                     │ CLAHE, RGB/gray                │
                     └──────┬──────────────────┬──────┘
          ┌─────────────────┘                  └─────────────────┐
          ▼                                                      ▼
 BRANCH A: LOCAL (geometry)                       BRANCH B: GLOBAL (semantics)
 SIFT → RootSIFT → centre                         DINOv2 ViT-B/14 → 768-d CLS
   │                                                │  (classical, frozen)
   ▼                                                ▼
 7-qubit amplitude encoding |ψᵢ⟩              PCA (fit on train) → 8–12 dims
   │                                                │
   ▼                                                ▼
 Quantum fidelity kernel K=|⟨ψ|φ⟩|²          Quantum feature map (ZZ/IQP, n qubits)
 Trace distance D=√(1−K)                            │
   │                                                ▼
   ▼                                          Quantum fidelity kernel
 Mutual-NN + ratio test                       K_g = |⟨φ(xA)|φ(xB)⟩|²
   │                                                │
   ▼                                                │
 RANSAC verification                                │
 (inliers, inlier ratio, mean inlier fidelity)      │
   │                                                │
 Mixed-state metrics: HS-cosine,                    │
 Uhlmann fidelity, QJSD                             │
   └────────────────────┬───────────────────────────┘
                        ▼
              Feature vector (≈9 values)
                        ▼
        Calibrated fusion (logistic regression / isotonic)
                        ▼
        Similarity % + explanation (matches image, components)
```

---

## Key Features

1. **Local Geometric Branch (Branch A)**:
   - RootSIFT keypoint extraction ($L_1 \to \sqrt{\cdot} \to L_2$).
   - Mean-centering and 7-qubit amplitude encoding ($|\psi_i\rangle \in \mathbb{C}^{128}$).
   - Exact quantum fidelity kernel $K = |\langle\psi|\phi\rangle|^2$ and trace distance $D_{\text{tr}} = \sqrt{1 - K}$.
   - Mutual Nearest-Neighbor (mutual-NN) matching with Lowe's ratio test on quantum trace distances.
   - Robust RANSAC geometric verification (Homography / Fundamental matrix).
   - Ensemble mixed-state density operator construction $\rho = \sum_i w_i |\psi_i\rangle\langle\psi_i|$ with Von Neumann entropy, Uhlmann-Jozsa fidelity $F(\rho_A, \rho_B)$, Quantum Jensen-Shannon Divergence ($\text{QJSD}$), and simulated SWAP-test estimator.

2. **Global Semantic Branch (Branch B)**:
   - Frozen DINOv2 ViT-B/14 backbone (`facebook/dinov2-base`, Apache 2.0 license).
   - Dimensionality reduction via PCA ($768 \to 8$).
   - $n$-qubit parameterized `zz_feature_map` from Qiskit 2.5 with linear entanglement.
   - Global quantum fidelity kernel $K_g(x_A, x_B) = |\langle\phi(x_A)|\phi(x_B)\rangle|^2$.
   - Classical DINOv2 cosine baseline.

3. **Multimodal Calibrated Fusion**:
   - 9-dimensional feature vector combining geometric, semantic, and density operator evidence.
   - Calibrated posterior probability via Logistic Regression and Isotonic Regression.
   - Output calibrated similarity percentage $[0\%, 100\%]$ with qualitative explanations.

4. **Gallery Search Mode**:
   - High-speed shortlist retrieval via FAISS `IndexFlatIP` (top-50).
   - Second-stage precision quantum geometric re-ranking.

5. **A0 - A7 Ablation & Baseline Suite**:
   - A0: pHash
   - A1: SIFT + RANSAC (fully classical)
   - A2: DINOv2 cosine similarity
   - A3: Legacy QSED score
   - A4: Branch A only (quantum local)
   - A5: Branch B only (DINOv2 + quantum kernel)
   - A6: Full hybrid fusion (proposed)
   - A7: Full fusion with classical kernels

---

## Directory Structure

```
QCSI/
├── data/
│   ├── download_datasets.py     # Procedural & standard benchmark dataset builder
│   └── synthetic_generator.py   # Realistic photographic & geometric transforms
├── qcsi/
│   ├── preprocess.py            # EXIF fix, CLAHE, resizing
│   ├── quantum.py               # Pure & mixed state quantum metrics, SWAP test
│   ├── local_branch.py          # Branch A: RootSIFT, 7-qubit encoding, RANSAC
│   ├── global_branch.py         # Branch B: DINOv2, PCA, ZZFeatureMap kernel
│   ├── fusion.py                # 9-dim fusion, Logistic/Isotonic calibration
│   ├── gallery.py               # FAISS shortlist + quantum re-ranking
│   └── api.py                   # FastAPI REST service
├── experiments/
│   ├── build_pairs.py           # Grouped splits & hard negative mining
│   ├── run_baselines.py         # A0-A3 baselines
│   ├── run_ablation.py          # A0-A7 ablation suite (ROC, PR, ECE, latency)
│   └── plots.py                 # Reliability diagrams, dashboards, ROC/PR curves
├── tests/                       # Complete unit and integration test suite
├── run_demo.py                  # CLI demo runner & 6-panel visual dashboard generator
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Run Interactive Pairwise Demo
```bash
python run_demo.py --demo --save_plot demo_report.png
```
This generates a test pair with perspective transformation, evaluates both branches, and saves a 6-panel verification dashboard.

To test two custom images:
```bash
python run_demo.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg --save_plot match_report.png
```

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Run Benchmark & Ablation Suite (A0 - A7)
```bash
python experiments/run_ablation.py --synthetic_scenes 6 --views_per_scene 3
```

### 4. Launch FastAPI REST Service
```bash
uvicorn qcsi.api:app --host 0.0.0.0 --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.
