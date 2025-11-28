"""
Visual prototype building
"""
import os
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from model.clip_encoder import encode_image_to_tokens
from utils import parse_labels_from_filename, list_training_images


def _prepare_text_prototypes(
    text_proto_path: str,
    classes: List[str],
    device: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
    if not os.path.exists(text_proto_path):
        raise FileNotFoundError(f"Text prototype file not found at {text_proto_path}")

    bank = torch.load(text_proto_path, map_location="cpu")
    per_prompt_bank: Dict[str, torch.Tensor] = bank["per_prompt_text_features"]

    mean_text_vecs: Dict[str, torch.Tensor] = {}
    prompt_text_vecs: Dict[str, torch.Tensor] = {}

    for c in classes:
        if c == "Background":
            continue
        if c not in per_prompt_bank:
            raise KeyError(f"Class '{c}' missing in text prototype bank.")

        feats = per_prompt_bank[c].float()
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        feats = F.normalize(feats, p=2, dim=-1)

        prompt_text_vecs[c] = feats.to(device)
        mean_text_vecs[c] = F.normalize(feats.mean(dim=0), p=2, dim=-1).to(device)

    any_cls = next(iter(prompt_text_vecs))
    feat_dim = int(prompt_text_vecs[any_cls].shape[-1])

    return mean_text_vecs, prompt_text_vecs, feat_dim


def build_visual_prototypes(
    train_dir: str,
    clip_model,
    clip_preprocess,
    classes: List[str],
    device: str = "cuda",
    n_prototypes_per_class: int = 16,
    batch_size: int = 64,
    feat_dim: int = None,
    text_proto_path: str = None,
    sim_threshold: float = 0.2,
    refine_threshold: float = 0.35,
    cam_topk: float = 0.3,
):
    """
    Build visual prototypes from training images using CAM-guided regional pooling.

    Args:
        train_dir: Directory containing training images.
        clip_model: CLIP model.
        clip_preprocess: CLIP preprocessing function.
        classes: List of class names (including Background).
        device: Device to run on.
        n_prototypes_per_class: Target number of prototypes per class.
        batch_size: Batch size for feature extraction.
        feat_dim: Feature dimension (auto-detected if None).
        text_proto_path: Path to text prototypes for CAM guidance.
        sim_threshold: Global cosine similarity threshold vs. text prototype.
        refine_threshold: Cosine similarity threshold vs. coarse prototype.
        cam_topk: Fraction of top-CAM tokens to pool.

    Returns:
        Dictionary with visual prototypes and metadata.
    """
    if text_proto_path is None:
        raise ValueError("text_proto_path must be provided for CAM-guided pooling.")

    sim_threshold = float(sim_threshold)
    refine_threshold = float(refine_threshold)
    cam_topk = float(cam_topk)
    cam_topk = min(max(cam_topk, 1e-3), 1.0)

    clip_model.eval()
    clip_model.to(device)
    clip_model.float()  # align CLIP weights with float32 image tensors

    # Auto-detect feature dimension from CLIP if needed
    if feat_dim is None:
        if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "output_dim"):
            feat_dim = int(clip_model.visual.output_dim)
        else:
            test_img = Image.new("RGB", (224, 224))
            test_tensor = clip_preprocess(test_img).unsqueeze(0).to(device)
            with torch.no_grad():
                test_feat = clip_model.encode_image(test_tensor)
                feat_dim = int(test_feat.shape[-1])

    text_mean_vecs, text_prompt_vecs, text_feat_dim = _prepare_text_prototypes(
        text_proto_path, classes, device
    )
    if feat_dim != text_feat_dim:
        raise ValueError(
            f"Feature dimension mismatch between CLIP ({feat_dim}) and text prototypes ({text_feat_dim})."
        )

    print(f"Using feature dimension: {feat_dim}")

    # List training images
    all_img_paths = list_training_images(train_dir)
    print(f"Found {len(all_img_paths)} training images in {train_dir}")

    img_labels: List[List[str]] = []
    for p in all_img_paths:
        cls_list = parse_labels_from_filename(p)
        img_labels.append(cls_list)

    class_img_count = defaultdict(int)
    for cls_list in img_labels:
        for c in cls_list:
            class_img_count[c] += 1

    print("Training image counts per class (multi-label counts):")
    for c in classes:
        if c == "Background":
            continue
        print(f"  {c:24s}: {class_img_count.get(c, 0)}")

    candidate_features: Dict[str, List[torch.Tensor]] = {
        c: [] for c in classes if c != "Background"
    }
    candidate_stats = {c: {"candidates": 0, "retained": 0} for c in candidate_features}

    with torch.no_grad():
        for i in tqdm(
            range(0, len(all_img_paths), batch_size),
            desc="Extracting CLIP regional features",
        ):
            batch_paths = all_img_paths[i : i + batch_size]
            batch_labels = img_labels[i : i + batch_size]

            images = []
            valid_idx = []
            for j, (path, _) in enumerate(zip(batch_paths, batch_labels)):
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    print(f"[WARN] Cannot open image {path}: {e}")
                    continue
                images.append(clip_preprocess(img))
                valid_idx.append(j)

            if not images:
                continue

            images_tensor = torch.stack(images).to(device)
            image_features = clip_model.encode_image(images_tensor)
            if image_features.shape[-1] != feat_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {feat_dim}, "
                    f"got {image_features.shape[-1]}"
                )
            image_features = F.normalize(image_features, p=2, dim=-1)

            tokens = encode_image_to_tokens(clip_model, images_tensor, project=True)
            patch_feats = tokens[:, 1:, :]  # drop CLS token
            if patch_feats.shape[-1] != feat_dim:
                raise ValueError(
                    f"Patch feature dimension mismatch: expected {feat_dim}, "
                    f"got {patch_feats.shape[-1]}"
                )
            patch_feats = F.normalize(patch_feats, p=2, dim=-1)

            num_tokens = patch_feats.shape[1]
            sqrt_tokens = int(num_tokens ** 0.5)
            if sqrt_tokens * sqrt_tokens != num_tokens:
                raise ValueError(
                    f"Unexpected number of tokens ({num_tokens}); cannot reshape to square grid."
                )

            for j, idx_in_batch in enumerate(valid_idx):
                lbls = batch_labels[idx_in_batch]
                if not lbls:
                    continue

                for c in lbls:
                    if c == "Background" or c not in candidate_features:
                        continue

                    global_sim = torch.dot(image_features[j], text_mean_vecs[c])
                    if global_sim.item() < sim_threshold:
                        continue

                    prompt_vecs = text_prompt_vecs[c]
                    sim_prompt = patch_feats[j] @ prompt_vecs.t()  # [N, K]
                    sim_vals, _ = sim_prompt.max(dim=1)

                    k = max(1, int(cam_topk * sim_vals.numel()))
                    topk_vals, topk_idx = sim_vals.topk(k, sorted=False)
                    mask = torch.zeros_like(sim_vals, dtype=torch.bool)
                    mask[topk_idx] = True

                    selected_feats = patch_feats[j][mask]
                    if selected_feats.numel() == 0:
                        continue

                    weights = topk_vals.unsqueeze(1)
                    agg_feat = (selected_feats * weights).sum(dim=0) / (
                        weights.sum(dim=0) + 1e-6
                    )
                    agg_feat = F.normalize(agg_feat, p=2, dim=-1)

                    candidate_features[c].append(agg_feat.cpu())
                    candidate_stats[c]["candidates"] += 1

    features_per_class: Dict[str, torch.Tensor] = {}

    print("\nCoarse prototype refinement:")
    for c, feats in candidate_features.items():
        if not feats:
            features_per_class[c] = torch.empty(0, feat_dim)
            print(f"  {c:24s}: 0 candidates -> 0 retained")
            continue

        stack = torch.stack(feats, dim=0)
        coarse = F.normalize(stack.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)

        refined_feats = []
        for feat in feats:
            sim = torch.dot(feat, coarse).item()
            if sim >= refine_threshold:
                refined_feats.append(feat)

        if not refined_feats:
            refined_feats = feats  # fallback

        features_per_class[c] = torch.stack(refined_feats, dim=0)
        candidate_stats[c]["retained"] = len(refined_feats)
        print(
            f"  {c:24s}: {len(feats)} candidates -> {len(refined_feats)} retained "
            f"(refine thr={refine_threshold:.2f})"
        )

    print("\nPer-class feature shapes:")
    for c, tensor in features_per_class.items():
        print(f"  {c:24s}: {tuple(tensor.shape)}")

    # Cluster to get visual prototypes
    visual_prototypes = {}
    prototype_meta = {}

    for c, feats in features_per_class.items():
        if feats.shape[0] == 0:
            print(f"[WARN] No features for class {c}, skipping prototype generation.")
            continue

        n_clusters = min(n_prototypes_per_class, feats.shape[0])
        print(
            f"\nRunning MiniBatchKMeans for class '{c}' with {feats.shape[0]} samples, "
            f"{n_clusters} clusters..."
        )

        X = feats.numpy().astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            max_iter=100,
            random_state=42,
            verbose=0,
            n_init=3,
        )
        kmeans.fit(X)

        centers = torch.from_numpy(kmeans.cluster_centers_)
        centers = F.normalize(centers, p=2, dim=-1)

        visual_prototypes[c] = centers
        prototype_meta[c] = {
            "n_samples": int(feats.shape[0]),
            "n_clusters": int(n_clusters),
        }

    print("\nVisual prototype shapes:")
    for c, proto in visual_prototypes.items():
        print(
            f"  {c:24s}: {tuple(proto.shape)} "
            f"(n_samples={prototype_meta[c]['n_samples']})"
        )

    return {
        "classes": [c for c in classes if c != "Background"],
        "visual_prototypes": visual_prototypes,
        "features_per_class_counts": {
            c: int(features_per_class[c].shape[0]) for c in features_per_class
        },
        "metadata": {
            "model_name": "CLIP",
            "n_prototypes_per_class_target": n_prototypes_per_class,
            "prototype_meta": prototype_meta,
            "feat_dim": feat_dim,
            "sim_threshold": sim_threshold,
            "refine_threshold": refine_threshold,
            "cam_topk": cam_topk,
        },
        "filter_stats": candidate_stats,
    }

