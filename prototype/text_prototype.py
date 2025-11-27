"""
Text prototype building
"""
import os
import torch
import torch.nn.functional as F
from typing import Dict, List


def build_text_prototypes(
    classes: List[str],
    class2prompts: Dict[str, List[str]],
    model,
    device: str = "cuda",
    prompt_prefix: str = "a histopathology image patch showing ",
    feat_dim: int = None
):
    """
    Build text prototypes from prompts.
    
    Args:
        classes: List of class names
        class2prompts: Dictionary mapping class to list of prompts
        model: CLIP model
        device: Device to run on
        prompt_prefix: Prefix to add to prompts
        feat_dim: Feature dimension (auto-detected if None)
    
    Returns:
        Dictionary with:
            - classes: list of classes
            - raw_prompts: original prompts
            - per_prompt_text_features: features per prompt per class
            - class_text_prototypes: mean prototype per class
            - feat_dim: feature dimension (auto-detected)
    """
    import clip
    
    per_prompt_text_features = {}
    class_text_prototypes = {}
    all_raw_prompts = {}

    with torch.no_grad():
        for cls in classes:
            prompts = class2prompts[cls]
            # Add prefix for better context
            full_texts = [prompt_prefix + p for p in prompts]

            # Tokenize & encode
            tokens = clip.tokenize(full_texts).to(device)
            text_features = model.encode_text(tokens)

            # Auto-detect feature dimension from first encoding
            if feat_dim is None:
                feat_dim = text_features.shape[-1]

            # Ensure dimension consistency
            if text_features.shape[-1] != feat_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {feat_dim}, "
                    f"got {text_features.shape[-1]}"
                )

            # L2 normalize
            text_features = F.normalize(text_features, p=2, dim=-1)

            # Save per-prompt features
            per_prompt_text_features[cls] = text_features.cpu()

            # Prototype = mean of prompts, then normalize
            # Handle both [num_prompts, feat_dim] and [feat_dim] cases
            if text_features.dim() == 2:
                proto = text_features.mean(dim=0, keepdim=True)
            else:
                proto = text_features.unsqueeze(0)
            proto = F.normalize(proto, p=2, dim=-1)
            # Keep original shape if it's [2, 512], otherwise squeeze
            if proto.shape[0] == 1:
                class_text_prototypes[cls] = proto.squeeze(0).cpu()
            else:
                class_text_prototypes[cls] = proto.cpu()

            all_raw_prompts[cls] = full_texts

    return {
        "classes": classes,
        "raw_prompts": all_raw_prompts,
        "per_prompt_text_features": per_prompt_text_features,
        "class_text_prototypes": class_text_prototypes,
        "feat_dim": feat_dim,  # Store detected dimension
    }

