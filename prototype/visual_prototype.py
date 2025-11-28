"""
Visual prototype building (Stage 1.1 parity).
"""
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from utils import list_training_images, parse_labels_from_filename


# Thresholds for detecting clean background patches (glass)
BG_PIXEL_INTENSITY_THR = 225
BG_AREA_RATIO_THR = 0.70
BG_SATURATION_THR = 20


def _is_visual_background(pil_img: Image.Image) -> bool:
    """
    Detect clean background (glass) patches using HSV heuristics.
    """
    try:
        img_hsv = pil_img.convert("HSV")
        _, s, v = img_hsv.split()
        np_v = np.array(v, dtype=np.uint8)
        np_s = np.array(s, dtype=np.uint8)

        is_white = (np_v > BG_PIXEL_INTENSITY_THR) & (np_s < BG_SATURATION_THR)
        white_ratio = float(is_white.mean())
        return white_ratio > BG_AREA_RATIO_THR
    except Exception:
        gray = np.array(pil_img.convert("L"), dtype=np.uint8)
        return float((gray > BG_PIXEL_INTENSITY_THR).mean()) > BG_AREA_RATIO_THR


def build_visual_prototypes(
    train_dir: str,
    clip_model,
    clip_preprocess,
    classes: List[str],
    device: str = "cuda",
    n_prototypes_per_class: int = 16,
    batch_size: int = 64,
    feat_dim: int = None,
    max_feats_per_class: int = 20_000,
    bg_class: str = "Background",
):
    """
    Build visual prototypes by extracting global CLIP features per patch
    and clustering (Stage 1.1 logic).

    Args:
        train_dir: Directory containing training patches.
        clip_model: Frozen CLIP model.
        clip_preprocess: CLIP preprocessing pipeline.
        classes: List of classes (FG + Background).
        device: Device for feature extraction.
        n_prototypes_per_class: Target #prototypes per class.
        batch_size: Batch size for feature extraction.
        feat_dim: Optional feature dimension override.
        max_feats_per_class: Cap on features per class before clustering.
        bg_class: Background class name.

    Returns:
        Dictionary containing visual prototypes, counts and metadata.
    """
    clip_model.eval()
    clip_model.to(device).float()

    if feat_dim is None:
        if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "output_dim"):
            feat_dim = int(clip_model.visual.output_dim)
        else:
            test_tensor = clip_preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device)
            with torch.no_grad():
                feat_dim = int(clip_model.encode_image(test_tensor).shape[-1])

    classes_fg = [c for c in classes if c != bg_class]
    classes_all_visual = classes_fg + [bg_class]

    print(f"Using feature dimension: {feat_dim}")

    all_img_paths = list_training_images(train_dir)
    print(f"Found {len(all_img_paths)} training images in {train_dir}")

    features_pure: Dict[str, List[torch.Tensor]] = {c: [] for c in classes_all_visual}
    features_mixed: Dict[str, List[torch.Tensor]] = {c: [] for c in classes_fg}

    counts = defaultdict(int)
    num_skipped_other = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(all_img_paths), batch_size), desc="Extracting & Sorting Features"):
            batch_paths = all_img_paths[start:start + batch_size]

            pil_imgs: List[Image.Image] = []
            tensors: List[torch.Tensor] = []
            valid_idx: List[int] = []

            for idx, path in enumerate(batch_paths):
                try:
                    pil_img = Image.open(path).convert("RGB")
                except Exception as exc:
                    print(f"[WARN] Cannot open {path}: {exc}")
                    continue
                pil_imgs.append(pil_img)
                tensors.append(clip_preprocess(pil_img))
                valid_idx.append(idx)

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(device)
            feats = clip_model.encode_image(batch_tensor)
            feats = F.normalize(feats, p=2, dim=-1).cpu()

            for local_idx, src_idx in enumerate(valid_idx):
                feat_vec = feats[local_idx]
                pil_img = pil_imgs[local_idx]
                path = batch_paths[src_idx]

                if _is_visual_background(pil_img):
                    features_pure[bg_class].append(feat_vec)
                    counts[bg_class] += 1
                    continue

                labels = parse_labels_from_filename(path)
                fg_labels = [c for c in labels if c in classes_fg]

                if len(fg_labels) == 1:
                    cls = fg_labels[0]
                    features_pure[cls].append(feat_vec)
                    counts[cls] += 1
                elif len(fg_labels) > 1:
                    for cls in fg_labels:
                        features_mixed[cls].append(feat_vec)
                        counts[f"{cls}_mixed"] += 1
                else:
                    num_skipped_other += 1

    print("\n[Stage 1.1] Extraction Summary:")
    print(f"  >> BACKGROUND (pure glass) patches: {counts[bg_class]}")
    print(f"  >> OTHER unlabeled non-BG patches skipped: {num_skipped_other}")
    for cls in classes_fg:
        pure_n = counts[cls]
        mixed_n = counts[f"{cls}_mixed"]
        print(f"  >> {cls:22s}: Pure={pure_n:5d} | Mixed={mixed_n:5d}")

    visual_prototypes: Dict[str, torch.Tensor] = {}
    prototype_meta: Dict[str, Dict] = {}

    def _cluster_feats(feats_tensor: torch.Tensor, class_name: str) -> torch.Tensor:
        if feats_tensor.shape[0] > max_feats_per_class:
            perm = torch.randperm(feats_tensor.shape[0])[:max_feats_per_class]
            feats_tensor = feats_tensor[perm]

        num_samples = feats_tensor.shape[0]
        n_clusters = min(n_prototypes_per_class, num_samples)
        print(f"  Computing Prototypes for '{class_name}' | Samples: {num_samples} -> K={n_clusters}")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            random_state=42,
            n_init="auto",
        )
        kmeans.fit(feats_tensor.numpy().astype(np.float32))

        centers = torch.from_numpy(kmeans.cluster_centers_)
        centers = F.normalize(centers, p=2, dim=-1)
        return centers

    for cls in classes_all_visual:
        pure_feats = features_pure.get(cls, [])
        mixed_feats = features_mixed.get(cls, [])

        tensors = []
        src_type = "none"

        if pure_feats:
            pure_tensor = torch.stack(pure_feats, dim=0)
            tensors.append(pure_tensor)
            src_type = "pure_only"

        if tensors and mixed_feats:
            tensors.append(torch.stack(mixed_feats, dim=0))
            src_type = "pure+mixed"
        elif not tensors and mixed_feats:
            tensors.append(torch.stack(mixed_feats, dim=0))
            src_type = "mixed_only"

        if not tensors:
            print(f"[WARN] Class '{cls}': no features gathered; skipping.")
            continue

        feats_tensor = torch.cat(tensors, dim=0)
        centers = _cluster_feats(feats_tensor, cls)

        visual_prototypes[cls] = centers
        prototype_meta[cls] = {
            "n_samples": int(feats_tensor.shape[0]),
            "source": src_type,
        }

    fg_prototypes = {
        cls: visual_prototypes[cls]
        for cls in classes_fg
        if cls in visual_prototypes
    }
    bg_prototypes = visual_prototypes.get(bg_class)

    metadata = {
        "clip_model": getattr(clip_model, "name", "CLIP"),
        "feature_dim": feat_dim,
        "bg_class": bg_class,
        "thresholds": {
            "pixel_intensity": BG_PIXEL_INTENSITY_THR,
            "area_ratio": BG_AREA_RATIO_THR,
            "saturation": BG_SATURATION_THR,
        },
        "num_skipped_other": num_skipped_other,
        "prototype_meta": prototype_meta,
    }

    features_counts = dict(counts)

    return {
        "classes": classes_fg,
        "bg_class": bg_class,
        "classes_all_visual": classes_all_visual,
        "visual_prototypes": fg_prototypes,
        "background_prototypes": bg_prototypes,
        "features_per_class_counts": features_counts,
        "metadata": metadata,
    }
