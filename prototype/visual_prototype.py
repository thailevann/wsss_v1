"""
Visual prototype building
"""
import os
import re
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from utils import parse_labels_from_filename, list_training_images


def build_visual_prototypes(
    train_dir: str,
    clip_model,
    clip_preprocess,
    classes: List[str],
    device: str = "cuda",
    n_prototypes_per_class: int = 16,
    batch_size: int = 64,
    feat_dim: int = None
):
    """
    Build visual prototypes from training images.
    
    Args:
        train_dir: Directory containing training images
        clip_model: CLIP model
        clip_preprocess: CLIP preprocessing function
        classes: List of class names (excluding Background)
        device: Device to run on
        n_prototypes_per_class: Number of prototypes per class
        batch_size: Batch size for feature extraction
        feat_dim: Feature dimension (auto-detected if None)
    
    Returns:
        Dictionary with visual prototypes and metadata
    """
    # Auto-detect feature dimension if not provided
    if feat_dim is None:
        # Try to get from model
        if hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'output_dim'):
            feat_dim = clip_model.visual.output_dim
        else:
            # Do a test encoding
            test_img = Image.new('RGB', (224, 224))
            test_tensor = clip_preprocess(test_img).unsqueeze(0).to(device)
            with torch.no_grad():
                test_feat = clip_model.encode_image(test_tensor)
                feat_dim = test_feat.shape[-1]

    print(f"Using feature dimension: {feat_dim}")

    clip_model.eval()
    clip_model.to(device)

    # List training images
    all_img_paths = list_training_images(train_dir)
    print(f"Found {len(all_img_paths)} training images in {train_dir}")

    img_labels: List[List[str]] = []
    for p in all_img_paths:
        cls_list = parse_labels_from_filename(p)
        img_labels.append(cls_list)

    # Count images per class
    class_img_count = defaultdict(int)
    for cls_list in img_labels:
        for c in cls_list:
            class_img_count[c] += 1

    print("Training image counts per class (multi-label counts):")
    for c in classes:
        if c == "Background":
            continue
        print(f"  {c:24s}: {class_img_count.get(c, 0)}")

    # Extract CLIP features per class
    features_per_class: Dict[str, List[torch.Tensor]] = {
        c: [] for c in classes if c != "Background"
    }

    with torch.no_grad():
        for i in tqdm(range(0, len(all_img_paths), batch_size), 
                     desc="Extracting CLIP features"):
            batch_paths = all_img_paths[i: i + batch_size]
            batch_labels = img_labels[i: i + batch_size]

            images = []
            valid_idx = []
            for j, (path, lbls) in enumerate(zip(batch_paths, batch_labels)):
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
            
            # Ensure dimension consistency
            if image_features.shape[-1] != feat_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {feat_dim}, "
                    f"got {image_features.shape[-1]}"
                )
            
            image_features = F.normalize(image_features, p=2, dim=-1)

            for j, idx_in_batch in enumerate(valid_idx):
                lbls = batch_labels[idx_in_batch]
                if not lbls:
                    continue
                # Use first label (single-label assumption for prototype building)
                c = lbls[0]
                if c in features_per_class:
                    feat = image_features[j].detach().cpu()
                    features_per_class[c].append(feat)

    # Convert lists to tensors
    for c in features_per_class:
        if len(features_per_class[c]) > 0:
            features_per_class[c] = torch.stack(features_per_class[c], dim=0)
        else:
            features_per_class[c] = torch.empty(0, feat_dim)

    print("\nPer-class feature shapes:")
    for c in features_per_class:
        print(f"  {c:24s}: {tuple(features_per_class[c].shape)}")

    # Cluster to get visual prototypes
    visual_prototypes = {}
    prototype_meta = {}

    for c, feats in features_per_class.items():
        if feats.shape[0] == 0:
            print(f"[WARN] No features for class {c}, skipping prototype generation.")
            continue

        n_clusters = min(n_prototypes_per_class, feats.shape[0])
        print(f"\nRunning MiniBatchKMeans for class '{c}' with {feats.shape[0]} samples, "
              f"{n_clusters} clusters...")

        X = feats.numpy().astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            max_iter=100,
            random_state=42,
            verbose=0,
            n_init=3
        )
        kmeans.fit(X)

        centers = kmeans.cluster_centers_
        centers = torch.from_numpy(centers)
        centers = F.normalize(centers, p=2, dim=-1)

        visual_prototypes[c] = centers
        prototype_meta[c] = {
            "n_samples": int(feats.shape[0]),
            "n_clusters": int(n_clusters),
        }

    print("\nVisual prototype shapes:")
    for c, proto in visual_prototypes.items():
        print(f"  {c:24s}: {tuple(proto.shape)} (n_samples={prototype_meta[c]['n_samples']})")

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
            "feat_dim": feat_dim,  # Store detected dimension
        },
    }

