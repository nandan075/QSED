"""
QCSI: Hybrid Quantum-Classical Image Similarity Framework.

Provides:
- Preprocessing: EXIF normalization, CLAHE, resizing.
- Branch A (Local): RootSIFT, 7-qubit amplitude encoding, fidelity kernel,
                   trace distance, mutual-NN + ratio test, RANSAC geometric verification,
                   density matrix metrics (Uhlmann fidelity, QJSD, HS-cosine).
- Branch B (Global): DINOv2 ViT-B/14, PCA (768 -> 8), parameterized ZZFeatureMap,
                    global quantum fidelity kernel.
- Fusion: 9-dimensional multimodal feature extraction, Logistic/Isotonic calibration,
          explainability engine.
- Gallery: FAISS candidate shortlisting + quantum re-ranking.
- API: FastAPI service for pairwise comparison and gallery search.
"""

__version__ = "1.0.0"
