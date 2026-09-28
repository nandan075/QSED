# Quantum Computer Vision & Quantum Image Edge Detection (Quantum_CV)

This workspace contains research implementations and algorithms for **Quantum Computer Vision (QIP/Quantum_CV)**, featuring:
1. **8-Directional Sobel Quantum Image Edge Detection (QSED)**: Full reversible quantum circuit reproduction of Liu & Wang (2023) with $\mathcal{O}(n^2 + q^2)$ complexity.
2. **Quantum Image Similarity Using Von Neumann Entropy**: Multi-scale structural entropy deviation and spatial pyramid matching engine solving the natural image entropy saturation problem.
3. **Quantum SIFT/SURF State Matching**: 7-qubit Hilbert space density matrix formulation with Uhlmann-Jozsa fidelity, Quantum Jensen-Shannon Divergence (QJSD), and 4-directional cardinal rotation invariance.

---

## Key Modules & Locations

- **[`QSED/`](file:///c:/Users/vuppa/Desktop/Quantum_CV/QSED/)**: The primary package containing core circuits, feature engineering, similarity scripts, and tests.
  - Complete documentation: [QSED Technical Documentation](file:///c:/Users/vuppa/Desktop/Quantum_CV/QSED/README.md)
  - Dense Spatial Entropy Similarity: `QSED/scripts/run_similarity.py`
  - SIFT/Pixel Density Matrix Similarity: `QSED/run_similarity.py`
  - Core Edge Detection: `QSED/src/main.py`
- **[`quantum_sift_matcher.py`](file:///c:/Users/vuppa/Desktop/Quantum_CV/quantum_sift_matcher.py)**: Standalone 6-panel diagnostic dashboard generator for rotation-invariant SIFT matching.
- **[`demo_images/`](file:///c:/Users/vuppa/Desktop/Quantum_CV/demo_images/)**: Sample anchor and rotated test images.

---

## Quick Execution Commands

### 1. Quantum Image Similarity via Von Neumann Entropy
```powershell
cd QSED
python scripts/run_similarity.py --image1 images/lena.png --image2 images/cameraman.png --size 512
```

### 2. Quantum SIFT Density Matrix Matching (with 4-Direction Rotation Search)
```powershell
cd QSED
python run_similarity.py --img1 images/lena.png --img2 images/lena.png --mode sift_entropy --rotate
```

### 3. Visual Match Report Dashboard
```powershell
python quantum_sift_matcher.py --img1 demo_images/image_anchor.png --img2 demo_images/image_rotated_test.png --out match_report.png
```

---

## Full Documentation & Theory
For full mathematical derivations of $S(\rho)$, the 8-directional Sobel quantum circuits, and Table 3 benchmark reproduction, refer to:
- **[QSED/README.md](file:///c:/Users/vuppa/Desktop/Quantum_CV/QSED/README.md)**
