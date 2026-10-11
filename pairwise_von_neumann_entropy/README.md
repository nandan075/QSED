# Pairwise Von Neumann Entropy Keypoint Matching

A self-contained image-matching module based on **Pairwise Von Neumann Entropy** between keypoint quantum state representations.

This implementation provides a separate, third entropy-based approach alongside existing single-keypoint scalar entropy and 24-dimensional local neighborhood entropy methods in the repository.

---

## 1. Problem Statement & Motivation

Traditional feature matching algorithms in computer vision rely on comparing local descriptor vectors (e.g. SIFT, ORB) pairwise between individual points in isolation. While effective for simple transformations, local descriptor matching frequently degrades under severe viewpoint deformation, non-rigid distortion, repetitive textures, and illumination shifts because it lacks global geometric and relational context.

In quantum information theory, the **Von Neumann entropy** of a mixed density operator measures the statistical uncertainty (mixedness) of a quantum ensemble. By representing the relationship between pairs of keypoints $(k_i, k_j)$ as an equal statistical mixture of their respective quantum pure state vectors:

$$\rho_{ij} = \frac{1}{2}\left( |\psi_i\rangle\langle\psi_i| + |\psi_j\rangle\langle\psi_j| \right)$$

the pairwise Von Neumann entropy $S_{ij} = -\operatorname{Tr}(\rho_{ij} \log_2 \rho_{ij})$ provides an intrinsic, continuous information-theoretic distance metric between keypoints. By evaluating this metric for all pairs within an image, we construct an $n \times n$ symmetric relational geometry matrix $E$ and $(n - 1)$-dimensional keypoint descriptors that capture the structural configuration of keypoints across the entire image.

---

## 2. Comparison with Existing Repository Approaches

| Dimension | Approach 1: Scalar Entropy Keypoints (`QSED/src/feature/von_neumann.py`) | Approach 2: 24-D Local Neighborhood Entropy (`QSED/src/feature/entropy_descriptor.py`) | Approach 3: Pairwise Von Neumann Entropy (`pairwise_von_neumann_entropy/`) |
| :--- | :--- | :--- | :--- |
| **Scope** | Single pixel $3 \times 3$ window | Fixed $5 \times 5$ neighborhood around single KP | Relational network across ALL pairs of keypoints $(k_i, k_j)$ in an image |
| **Descriptor Size** | Scalar $S \in \mathbb{R}$ (1 value) | Fixed 24-D vector $F(KP) \in \mathbb{R}^{24}$ | $(n - 1)$-D relational vector $D(k_i) \in \mathbb{R}^{n - 1}$ + $n \times n$ matrix $E$ |
| **Quantum State** | Intensity diagonal matrix $\operatorname{diag}(q)$ | 24 diagonal matrices around KP | Pure state vectors $|\psi_i\rangle \in \mathcal{H}_d$, mixed state $\rho_{ij} = \frac{1}{2}(|\psi_i\rangle\langle\psi_i| + |\psi_j\rangle\langle\psi_j|)$ |
| **Matching Nature** | Direct entropy map correlation / pyramid matching | Euclidean vector distance matching in $\mathbb{R}^{24}$ | Permutation-aware spectral graph matching of relational matrices $E^A, E^B$ |
| **Geometric Context** | Strictly local ($3 \times 3$ patch) | Strictly local ($5 \times 5$ patch) | Global intra-image relational graph + RANSAC verification |

---

## 3. Mathematical Derivation

### A. Quantum State Encoding
Given an image containing $n$ detected keypoints $K = \{k_1, k_2, \dots, k_n\}$:
For each keypoint $k_i$, extract a local $w \times w$ image patch $P_i$ (e.g. $w=9$, $d=81$ pixels) with reflection boundary padding (`mode='reflect'`).
The patch is zero-mean centered and normalized to unit Euclidean norm:

$$|\psi_i\rangle = \frac{v_i}{\|v_i\|_2} \in \mathcal{H}_d, \quad \langle\psi_i|\psi_i\rangle = 1.0$$

### B. Pairwise Density Matrix
For every distinct pair $(k_i, k_j)$ with $i \neq j$, the statistical ensemble density operator is:

$$\rho_{ij} = \frac{1}{2}\left( |\psi_i\rangle\langle\psi_i| + |\psi_j\rangle\langle\psi_j| \right)$$

**Verification of physical validity:**
1. **Hermitian:** $\rho_{ij}^\dagger = \frac{1}{2}( (|\psi_i\rangle\langle\psi_i|)^\dagger + (|\psi_j\rangle\langle\psi_j|)^\dagger ) = \rho_{ij}$.
2. **Unit Trace:** $\operatorname{Tr}(\rho_{ij}) = \frac{1}{2}(\langle\psi_i|\psi_i\rangle + \langle\psi_j|\psi_j\rangle) = \frac{1}{2}(1 + 1) = 1.0$.
3. **Positive Semi-Definite:** For any state $|\phi\rangle$, $\langle\phi|\rho_{ij}|\phi\rangle = \frac{1}{2}(|\langle\phi|\psi_i\rangle|^2 + |\langle\phi|\psi_j\rangle|^2) \ge 0$.

### C. Pairwise Von Neumann Entropy
The Von Neumann entropy in bits (base 2) is:

$$S_{ij} = -\operatorname{Tr}(\rho_{ij}\log_2\rho_{ij}) = -\sum_{\lambda_k > 0} \lambda_k \log_2 \lambda_k$$

### D. Subspace Eigenvalue Optimization
Constructing and diagonalizing the $d \times d$ matrix $\rho_{ij}$ for all $\frac{n(n-1)}{2}$ pairs requires $O(n^2 d^3)$ operations.
However, because $\rho_{ij}$ is formed by two pure states, its range is confined to at most the 2-dimensional subspace $\operatorname{span}\{|\psi_i\rangle, |\psi_j\rangle\}$. All remaining $d - 2$ eigenvalues are identically zero.

Let $c_{ij} = |\langle\psi_i|\psi_j\rangle|^2 \in [0, 1]$ denote the transition probability (quantum fidelity).
The trace identities in the 2D subspace yield:
- $\operatorname{Tr}(\rho_{ij}) = \lambda_+ + \lambda_- = 1$
- $\operatorname{Tr}(\rho_{ij}^2) = \lambda_+^2 + \lambda_-^2 = \frac{1 + c_{ij}}{2}$
- $(\lambda_+ - \lambda_-)^2 = 2\operatorname{Tr}(\rho_{ij}^2) - (\operatorname{Tr}(\rho_{ij}))^2 = 1 + c_{ij} - 1 = c_{ij}$

Therefore:

$$\lambda_{\pm} = \frac{1 \pm \sqrt{c_{ij}}}{2}$$

The pairwise entropy can be evaluated directly without matrix diagonalization:

$$S_{ij} = -\lambda_+\log_2(\lambda_+) - \lambda_-\log_2(\lambda_-)$$

**Extreme limits:**
- **Identical states** ($|\psi_i\rangle = |\psi_j\rangle \implies c_{ij} = 1$): $\lambda_+ = 1, \lambda_- = 0 \implies S_{ij} = 0.0$ bits.
- **Orthogonal states** ($\langle\psi_i|\psi_j\rangle = 0 \implies c_{ij} = 0$): $\lambda_+ = 0.5, \lambda_- = 0.5 \implies S_{ij} = 1.0$ bit.
- **Bounded range:** $0.0 \le S_{ij} \le 1.0$ bit for all pairs.

---

## 4. Keypoint Descriptor Construction

1. Compute the symmetric $n \times n$ matrix $E$:

   $$E_{ij} = E_{ji} = S_{ij}, \quad E_{ii} = 0.0$$

   *(Note: The diagonal zero is a storage convention for this algorithm, not a physical claim that the individual entropy of a mixed state is zero).*

2. For each keypoint $k_i$, extract row $i$ of $E$ and omit the diagonal element $E_{ii}$ to obtain the $(n - 1)$-dimensional descriptor:

   $$D(k_i) = [S_{i1}, \dots, S_{i, i-1}, S_{i, i+1}, \dots, S_{in}] \in \mathbb{R}^{n - 1}$$

3. Store coordinates $(x_i, y_i)$, detector metadata, the state vector $|\psi_i\rangle$, and $D(k_i)$ in a `KeypointPairwiseDescriptor` object.

---

## 5. Permutation-Aware Matching Strategy

When comparing Image A ($n_A$ keypoints) and Image B ($n_B$ keypoints), direct index-by-index comparison of $D(k_i^A)$ and $D(k_j^B)$ is mathematically invalid because $n_A - 1 \neq n_B - 1$ in general, and the keypoint orderings are arbitrary.

### Step 1: Candidate Seed Correspondence Generation
Compute quantum state fidelity $A(i, j) = |\langle\psi_i^A | \psi_j^B\rangle|^2$.
Select candidate matches $M = \{m_1, \dots, m_K\}$ via mutual nearest neighbors or Lowe's ratio test.

### Step 2: Relational Pairwise Entropy Compatibility
For candidate matches $m = (i, j)$ and $m' = (p, q)$ ($i \neq p, j \neq q$):

$$\Delta E(m, m') = |E_{ip}^A - E_{jq}^B|$$

$$W_{m, m'} = \exp\left(-\frac{\Delta E(m, m')}{\sigma_E}\right) \sqrt{A(i, j) \cdot A(p, q)}$$

If $i = p$ or $j = q$ (conflicting 1-to-many match), set $W_{m, m'} = 0.0$.
Diagonal: $W_{m, m} = A(i, j)$.

### Step 3: Spectral Assignment & Discretization
Compute the leading eigenvector $x^*$ of $W$ via power iteration:

$$x_{t+1} = \frac{W x_t}{\|W x_t\|_2}$$

The vector $x^*$ ranks the mutual relational coherence of candidate correspondences.
Discretize $x^*$ greedily to enforce the 1-to-1 matching constraint and reject candidates below the ambiguity cutoff $\tau_{\text{ambiguity}}$.

### Step 4: RANSAC Geometric Verification
Fit a geometric transformation (Homography or Fundamental matrix) using RANSAC to reject remaining spatial outliers.

---

## 6. Time and Memory Complexity

| Stage | Operation | Time Complexity | Memory Complexity |
| :--- | :--- | :--- | :--- |
| **Patch Encoding** | Extract $n$ patches of size $d = w^2$ | $O(n \cdot d)$ | $O(n \cdot d)$ |
| **Gram Matrix** | $G = \Psi \Psi^T$ | $O(n^2 \cdot d)$ | $O(n^2)$ |
| **Optimized Entropy** | Elementwise $\lambda_{\pm}$ and $S_{ij}$ | $O(n^2)$ | $O(n^2)$ |
| **Reference Entropy** | Explicit diagonalization of $\rho_{ij}$ | $O(n^2 \cdot d^3)$ | $O(n^2 \cdot d^2)$ |
| **Seed Matching** | Cross-overlap $G_{AB} = \Psi_A \Psi_B^T$ | $O(n_A \cdot n_B \cdot d)$ | $O(n_A \cdot n_B)$ |
| **Compatibility Graph** | Construct $K \times K$ affinity matrix $W$ | $O(K^2)$ | $O(K^2)$ ($K \le \min(n_A, n_B)$) |
| **Spectral Assignment** | $T$ power iterations | $O(T \cdot K^2)$ | $O(K)$ |
| **RANSAC** | $I$ hypothesis iterations | $O(I \cdot K)$ | $O(K)$ |

---

## 7. Dependencies & Installation (For Later Device)

Install requirements on your target device:

```bash
pip install -r pairwise_von_neumann_entropy/requirements.txt
```

Core dependencies:
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `matplotlib >= 3.7.0`
- `pillow >= 9.5.0`
- `opencv-python >= 4.7.0` (optional, for SIFT/ORB detection and OpenCV RANSAC)
- `pytest >= 7.3.0` (for test execution)

---

## 8. Example Execution Commands (For Later Device)

### A. Run Unit Test Suite
```bash
pytest pairwise_von_neumann_entropy/tests/ -v
```

### B. Two-Image Matching via CLI
```bash
python -m pairwise_von_neumann_entropy.main \
    --img1 QSED/images/cameraman.png \
    --img2 QSED/images/lena.png \
    --detector sift \
    --max-kps 100 \
    --entropy-method optimized \
    --geom-model homography \
    --output-dir outputs/pairwise_matching
```

### C. Using Harris Detector (No OpenCV Dependency)
```bash
python -m pairwise_von_neumann_entropy.main \
    --img1 QSED/images/cameraman.png \
    --img2 QSED/images/cameraman.png \
    --detector harris \
    --max-kps 60
```

---

## 9. Expected Inputs & Outputs

### Inputs:
- Grayscale or RGB image files (PNG, JPEG, BMP).
- Command-line parameters or configuration objects.

### Outputs:
1. **Console Report:** Number of keypoints, candidate matches, RANSAC inliers, mean relational error in bits, and overall similarity score percentage.
2. **`pairwise_entropy_matches.png`:** Side-by-side correspondence plot with green inlier and red outlier lines.
3. **`pairwise_entropy_matrices.png`:** Heatmaps of $E^A$ and $E^B$.
4. **`keypoint_relational_profile_a0.png`:** $(n - 1)$-D bar profile of keypoint $k_0$.

---

## 10. Known Limitations

1. **Quadratic Scaling in Keypoint Count:** Constructing the $n \times n$ matrix requires $O(n^2)$ space. For very large keypoint counts ($n > 2000$), keypoint filtering or hierarchical clustering is recommended.
2. **Missing Keypoints & Occlusion:** If significant portions of an image are occluded, corresponding keypoints may be absent, altering the relational graph topology.
3. **In-Plane Rotation Sensitivity of Raw Patches:** The `patch_amplitude` encoding is sensitive to large rotations. When rotation invariance is required, use `--encoding rootsift` or align patches with local dominant gradient orientations.
4. **Scale Variations:** Fixed patch sizes ($w \times w$) capture fixed spatial radii. Canonical scale adaptation via multi-scale detection is necessary for large scale changes.

---

## 11. Future Benchmarking Plan

When executed on the evaluation device, compare:
1. **Matching Accuracy & AUC:** Against existing scalar entropy matching (`QSED/run_similarity.py --mode pixel`) and 24-D local entropy descriptor (`--mode local_entropy_24d`).
2. **Runtime Efficiency:** Measure speedup of analytical eigenvalue calculation vs reference density matrix diagonalization across patch sizes $d \in \{25, 81, 225\}$.
3. **Robustness to Illumination & Noise:** Evaluate on synthetic noise and illumination ramps.
