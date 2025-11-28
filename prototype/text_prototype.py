"""
Text prototype building (Stage 0 parity with notebook pipeline).
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


DEFAULT_PROMPT_TEMPLATES = [
    "a histopathology image patch showing {}",
    "an H&E stained breast histology patch with {}",
    "a microscopic image of {} in breast cancer tissue",
]


def build_text_prototypes(
    classes: List[str],
    class2prompts: Dict[str, List[str]],
    model,
    device: str = "cuda",
    prompt_templates: Optional[List[str]] = None,
    classes_fg: Optional[List[str]] = None,
    bg_class: Optional[str] = "Background",
    noise_classes: Optional[List[str]] = None,
) -> Dict:
    """
    Build text prototypes using multiple prompt templates and global centering.

    Mirrors the Stage 0 logic from the reference notebook:
        - expand each base phrase via several natural language templates
        - encode with CLIP, normalize, average per class
        - perform global centering across all classes for better separation

    Args:
        classes: All classes to build prototypes for (FG + BG).
        class2prompts: Mapping class -> list of base phrases.
        model: Frozen CLIP model.
        device: Device to run encoding on.
        prompt_templates: Optional list of sentence templates. Uses defaults if None.
        classes_fg: Optional list of foreground classes.
        bg_class: Name of background class (default "Background").
        noise_classes: Optional list of classes treated as noise (for metadata).

    Returns:
        Dictionary containing per-prompt features, class prototypes, metadata.
    """
    import clip

    prompt_templates = prompt_templates or DEFAULT_PROMPT_TEMPLATES
    classes_fg = classes_fg or [c for c in classes if c != bg_class]
    noise_classes = noise_classes or []

    per_prompt_text_features: Dict[str, torch.Tensor] = {}
    class_text_prototypes: Dict[str, torch.Tensor] = {}
    all_raw_prompts: Dict[str, List[str]] = {}

    feat_dim: Optional[int] = None

    model.eval()
    with torch.no_grad():
        for cls in classes:
            if cls not in class2prompts:
                raise KeyError(f"Missing prompts for class '{cls}'.")

            base_phrases = class2prompts[cls]
            full_texts = [
                template.format(phrase)
                for phrase in base_phrases
                for template in prompt_templates
            ]

            tokens = clip.tokenize(full_texts).to(device)
            text_features = model.encode_text(tokens).float()

            if feat_dim is None:
                feat_dim = int(text_features.shape[-1])
            elif text_features.shape[-1] != feat_dim:
                raise ValueError(
                    f"Feature dimension mismatch for '{cls}': "
                    f"expected {feat_dim}, got {text_features.shape[-1]}"
                )

            text_features = F.normalize(text_features, p=2, dim=-1)
            per_prompt_text_features[cls] = text_features.cpu()

            proto = text_features.mean(dim=0, keepdim=True)
            proto = F.normalize(proto, p=2, dim=-1)
            class_text_prototypes[cls] = proto.squeeze(0).cpu()

            all_raw_prompts[cls] = full_texts

    # Global centering across all classes (angular separation)
    all_proto = torch.stack([class_text_prototypes[c] for c in classes], dim=0)
    mean_global = all_proto.mean(dim=0, keepdim=True)
    for cls in classes:
        centered = class_text_prototypes[cls] - mean_global.squeeze(0)
        class_text_prototypes[cls] = F.normalize(centered, p=2, dim=-1)

    class2idx = {c: i for i, c in enumerate(classes)}

    return {
        "classes": classes,
        "classes_fg": classes_fg,
        "bg_class": bg_class,
        "noise_classes": noise_classes,
        "class2idx": class2idx,
        "prompt_templates": prompt_templates,
        "base_phrases": class2prompts,
        "raw_prompts": all_raw_prompts,
        "per_prompt_text_features": per_prompt_text_features,
        "class_text_prototypes": class_text_prototypes,
        "feat_dim": feat_dim if feat_dim is not None else 0,
        "config": {
            "global_centering": True,
            "note": (
                "Text prototypes built with multi-template expansion and global centering. "
                "Noise classes retained for downstream ProGrad."
            ),
        },
    }
