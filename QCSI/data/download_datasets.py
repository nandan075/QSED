"""
QCSI Dataset Downloader & Local Benchmark Builder.

Provides:
1. Helpers to prepare standard evaluation benchmarks (INRIA Holidays, HPatches).
2. Synthetic benchmark generator creating 10+ distinct grouped scenes with diverse objects,
   shapes, and textures for calibration and offline test suites.
"""

from typing import List, Dict, Tuple, Any
import os
import cv2
import numpy as np
from data.synthetic_generator import SyntheticImagePerturber


def create_procedural_scene(scene_id: int, size: Tuple[int, int] = (400, 400)) -> np.ndarray:
    """
    Synthesizes rich, unique geometric and photographic patterns for benchmarking.
    """
    h, w = size
    rng = np.random.default_rng(scene_id * 101 + 42)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Base color gradient
    base_color = rng.integers(50, 200, size=3)
    for y in range(h):
        grad = y / float(h)
        img[y, :] = np.clip(base_color * (0.6 + 0.8 * grad), 0, 255).astype(np.uint8)

    # Draw complex geometric features to provide rich SIFT keypoints
    num_shapes = rng.integers(12, 20)
    for _ in range(num_shapes):
        shape_type = rng.choice(["circle", "rect", "poly", "text", "lines"])
        col = tuple(int(c) for c in rng.integers(20, 255, size=3))

        if shape_type == "circle":
            center = (int(rng.integers(50, w - 50)), int(rng.integers(50, h - 50)))
            radius = int(rng.integers(15, 60))
            cv2.circle(img, center, radius, col, -1)
            cv2.circle(img, center, radius, (255 - col[0], 255 - col[1], 255 - col[2]), 2)

        elif shape_type == "rect":
            x1, y1 = int(rng.integers(20, w - 80)), int(rng.integers(20, h - 80))
            x2, y2 = x1 + int(rng.integers(30, 80)), y1 + int(rng.integers(30, 80))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)

        elif shape_type == "poly":
            pts = rng.integers(20, min(w, h) - 20, size=(4, 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], col)

        elif shape_type == "lines":
            for _ in range(5):
                pt1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                pt2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                cv2.line(img, pt1, pt2, col, int(rng.integers(1, 4)))

        elif shape_type == "text":
            pos = (int(rng.integers(30, w - 120)), int(rng.integers(30, h - 30)))
            text = f"SCENE-{scene_id:02d}"
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    return img


def build_local_benchmark_dataset(
    output_dir: str,
    num_scenes: int = 10,
    views_per_scene: int = 4
) -> List[Dict[str, Any]]:
    """
    Generates a grouped multi-view benchmark dataset with known ground-truth scenes.
    
    Structure:
    output_dir/
      scene_00/
        view_00.png (base anchor)
        view_01.png (rotation/perspective)
        ...
    """
    os.makedirs(output_dir, exist_ok=True)
    perturber = SyntheticImagePerturber(seed=123)
    manifest = []

    for s_idx in range(num_scenes):
        scene_dir = os.path.join(output_dir, f"scene_{s_idx:02d}")
        os.makedirs(scene_dir, exist_ok=True)

        anchor = create_procedural_scene(s_idx)
        anchor_path = os.path.join(scene_dir, "view_00_anchor.png")
        cv2.imwrite(anchor_path, cv2.cvtColor(anchor, cv2.COLOR_RGB2BGR))

        manifest.append({
            "scene_id": s_idx,
            "view_id": 0,
            "file_path": anchor_path,
            "transform": "anchor",
        })

        for v_idx in range(1, views_per_scene):
            transformed, desc = perturber.generate_random_transform(anchor)
            view_path = os.path.join(scene_dir, f"view_{v_idx:02d}_{desc}.png")
            cv2.imwrite(view_path, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))

            manifest.append({
                "scene_id": s_idx,
                "view_id": v_idx,
                "file_path": view_path,
                "transform": desc,
            })

    return manifest
