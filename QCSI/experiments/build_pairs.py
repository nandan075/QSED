"""
QCSI Pair Generation Script with Group-Based Splitting & Hard Negatives.

Generates:
- Positive pairs: same scene with distinct views or synthetic perturbations.
- Random negative pairs: distinct scenes.
- Hard negative pairs: distinct scenes with highest DINOv2 semantic cosine similarity.
- Grouped split: partitions by scene ID to strictly prevent scene leakage across train/val/test.
"""

from typing import List, Dict, Any, Tuple
import os
import csv
import itertools
import numpy as np

from qcsi.preprocess import ImagePreprocessor
from qcsi.global_branch import DINOv2Extractor


def build_labeled_pairs(
    dataset_manifest: List[Dict[str, Any]],
    output_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_hard_negatives_per_scene: int = 2,
    seed: int = 42
) -> Dict[str, str]:
    """
    Constructs train, validation, and test pair CSVs with strict group-based splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Group by scene_id
    scene_groups: Dict[int, List[Dict[str, Any]]] = {}
    for item in dataset_manifest:
        s_id = item["scene_id"]
        scene_groups.setdefault(s_id, []).append(item)

    unique_scenes = sorted(list(scene_groups.keys()))
    rng.shuffle(unique_scenes)

    n_scenes = len(unique_scenes)
    n_train = max(1, int(round(n_scenes * train_ratio)))
    n_val = max(1, int(round(n_scenes * val_ratio))) if n_scenes > 2 else 0

    train_scenes = set(unique_scenes[:n_train])
    val_scenes = set(unique_scenes[n_train:n_train + n_val])
    test_scenes = set(unique_scenes[n_train + n_val:])
    if not test_scenes and val_scenes:
        test_scenes = set([unique_scenes[-1]])

    split_scenes = {
        "train": train_scenes,
        "val": val_scenes if val_scenes else train_scenes,
        "test": test_scenes if test_scenes else train_scenes,
    }

    # Initialize extractor for hard negative mining
    preproc = ImagePreprocessor()
    extractor = DINOv2Extractor()

    # Precompute embeddings for all manifest items
    for item in dataset_manifest:
        img_rgb = preproc.load_image(item["file_path"])
        item["embedding"] = extractor.extract_embedding(img_rgb)

    output_files = {}

    for split_name, scenes in split_scenes.items():
        items_in_split = [item for item in dataset_manifest if item["scene_id"] in scenes]
        pairs = []

        # 1. Positives (same scene)
        for s_id in scenes:
            scene_items = scene_groups[s_id]
            for it1, it2 in itertools.combinations(scene_items, 2):
                pairs.append({
                    "img1_path": it1["file_path"],
                    "img2_path": it2["file_path"],
                    "label": 1,
                    "scene1": s_id,
                    "scene2": s_id,
                    "pair_type": "positive",
                })

        num_positives = len(pairs)

        # 2. Hard Negatives via DINOv2 cosine similarity
        scene_list = list(scenes)
        for s_id in scene_list:
            other_items = [it for it in dataset_manifest if it["scene_id"] != s_id]
            if not other_items:
                continue

            scene_anchors = [it for it in scene_groups[s_id] if it in items_in_split]
            if not scene_anchors:
                continue
            anchor = scene_anchors[0]

            other_embs = np.vstack([it["embedding"] for it in other_items])
            sims = np.dot(other_embs, anchor["embedding"])
            top_hard_indices = np.argsort(sims)[::-1][:num_hard_negatives_per_scene]

            for hard_idx in top_hard_indices:
                hard_item = other_items[hard_idx]
                pairs.append({
                    "img1_path": anchor["file_path"],
                    "img2_path": hard_item["file_path"],
                    "label": 0,
                    "scene1": s_id,
                    "scene2": hard_item["scene_id"],
                    "pair_type": "hard_negative",
                })

        # 3. Random Negatives to balance classes
        num_neg_needed = max(num_positives - (len(pairs) - num_positives), 2)
        all_possible_neg_pairs = []
        for it1 in items_in_split:
            for it2 in dataset_manifest:
                if it1["scene_id"] != it2["scene_id"]:
                    all_possible_neg_pairs.append((it1, it2))

        rng.shuffle(all_possible_neg_pairs)
        for it1, it2 in all_possible_neg_pairs[:num_neg_needed]:
            pairs.append({
                "img1_path": it1["file_path"],
                "img2_path": it2["file_path"],
                "label": 0,
                "scene1": it1["scene_id"],
                "scene2": it2["scene_id"],
                "pair_type": "random_negative",
            })

        # Write split CSV
        csv_path = os.path.join(output_dir, f"pairs_{split_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["img1_path", "img2_path", "label", "scene1", "scene2", "pair_type"]
            )
            writer.writeheader()
            for p in pairs:
                writer.writerow(p)

        output_files[split_name] = csv_path

    return output_files
