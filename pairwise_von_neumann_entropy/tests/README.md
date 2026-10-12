# Test Suite: Pairwise Von Neumann Entropy Keypoint Matching

This directory contains comprehensive unit tests verifying all mathematical formulations, density operator constraints, descriptor dimensions, permutation invariance, and image matching behaviors.

## Test Coverage Overview

| Test Module | Test Coverage |
| :--- | :--- |
| `test_entropy.py` | 1. Identical states produce $S_{ij} = 0.0$ bits.<br>2. Orthogonal states produce $S_{ij} = 1.0$ bit.<br>3. Symmetry $S_{ij} = S_{ji}$.<br>4. Density matrix Hermiticity, PSD, and unit trace.<br>5. Boundedness $S_{ij} \in [0.0, 1.0]$.<br>6. Numerical equivalence between optimized subspace eigenvalues and reference density matrix diagonalization. |
| `test_descriptors.py` | 1. Pairwise entropy matrix $E$ symmetry ($E = E^T$).<br>2. Diagonal entries $E_{ii} = 0.0$ storage convention.<br>3. Descriptors $D(k_i)$ contain exactly $(n - 1)$ entries.<br>4. Boundary cases: $n = 0$ (empty) and $n = 1$ (single keypoint).<br>5. Keypoint coordinate and detector metadata preservation. |
| `test_matching.py` | 1. Differing keypoint counts ($n_A \neq n_B$).<br>2. Permutation invariance (shuffling keypoint list ordering).<br>3. Rejection of ambiguous and incoherent correspondences.<br>4. Safe handling of $< 4$ matches in RANSAC geometric verification.<br>5. Synthetic image pair end-to-end matching. |

## Execution Commands (For Later Use on Another Device)

Run all tests from the repository root:
```bash
pytest pairwise_von_neumann_entropy/tests/ -v
```

Or run an individual test module:
```bash
pytest pairwise_von_neumann_entropy/tests/test_entropy.py -v
pytest pairwise_von_neumann_entropy/tests/test_descriptors.py -v
pytest pairwise_von_neumann_entropy/tests/test_matching.py -v
```
