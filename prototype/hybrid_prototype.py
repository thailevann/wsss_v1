"""
Hybrid prototype building (Stage 1.3 parity).
"""
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F


ALPHA_V_CFG_DEFAULT = {
    "Tumor": 0.7,
    "Stroma": 0.5,
    "Lymphocytic infiltrate": 0.6,
    "Necrosis": 0.7,
}
ALPHA_V_DEFAULT = 0.6


def build_hybrid_prototypes(
    text_proto_path: str,
    vision_proto_path: str,
    feat_dim: int = None,
    alpha_v_cfg: Optional[Dict[str, float]] = None,
    alpha_v_default: float = ALPHA_V_DEFAULT,
):
    """
    Combine learned visual prototypes and per-prompt text features into hybrid prototypes:
        H_c,m,k = normalize( alpha_v(c) * P_c,m + (1 - alpha_v(c)) * T_c,k )

    Args:
        text_proto_path: Path to Stage 0 text prototypes.
        vision_proto_path: Path to Stage 1.2 visual prototypes.
        feat_dim: Optional override for feature dimension.
        alpha_v_cfg: Optional per-class weighting dict. If None, uses defaults.
        alpha_v_default: Default alpha_v when class not in cfg.

    Returns:
        Dictionary mirroring the notebook Stage 1.3 output.
    """
    if not os.path.exists(text_proto_path):
        raise FileNotFoundError(f"Text prototype file not found at {text_proto_path}")
    if not os.path.exists(vision_proto_path):
        raise FileNotFoundError(f"Learned vision prototype file not found at {vision_proto_path}")

    text_proto_bank = torch.load(text_proto_path, map_location="cpu")
    vision_proto_bank = torch.load(vision_proto_path, map_location="cpu")

    per_prompt_text_features: Dict[str, torch.Tensor] = text_proto_bank["per_prompt_text_features"]
    vision_prototypes_learned: Dict[str, torch.Tensor] = vision_proto_bank["vision_prototypes_learned"]
    classes = vision_proto_bank["classes"]

    if feat_dim is None:
        if "feat_dim" in text_proto_bank:
            feat_dim = int(text_proto_bank["feat_dim"])
        elif "feat_dim" in vision_proto_bank:
            feat_dim = int(vision_proto_bank["feat_dim"])
        else:
            first_cls = classes[0]
            proto = text_proto_bank["class_text_prototypes"][first_cls]
            feat_dim = int(proto.shape[-1]) if proto.ndim >= 1 else int(proto.shape[0])

    alpha_v_cfg = alpha_v_cfg or ALPHA_V_CFG_DEFAULT

    print("Classes in vision prototype bank:", classes)
    print("Feature dim:", feat_dim)

    hybrid_prototypes: Dict[str, torch.Tensor] = {}
    meta_info: Dict[str, Dict] = {}

    for cls in classes:
        if cls not in vision_prototypes_learned:
            raise KeyError(f"Class '{cls}' not found in vision prototypes.")
        if cls not in per_prompt_text_features:
            raise KeyError(f"Class '{cls}' not found in text per-prompt features.")

        P_c = vision_prototypes_learned[cls].float()
        T_c = per_prompt_text_features[cls].float()

        if P_c.ndim == 1:
            P_c = P_c.unsqueeze(0)
        if T_c.ndim == 1:
            T_c = T_c.unsqueeze(0)

        if P_c.ndim != 2 or T_c.ndim != 2:
            raise ValueError(f"Unexpected prototype shapes for class '{cls}': vision {P_c.shape}, text {T_c.shape}")

        if P_c.shape[-1] != feat_dim or T_c.shape[-1] != feat_dim:
            raise ValueError(
                f"Feature dimension mismatch for class '{cls}': "
                f"vision {P_c.shape[-1]}, text {T_c.shape[-1]}, expected {feat_dim}"
            )

        P_c = F.normalize(P_c, p=2, dim=-1)
        T_c = F.normalize(T_c, p=2, dim=-1)

        alpha_v = float(alpha_v_cfg.get(cls, alpha_v_default))
        alpha_t = 1.0 - alpha_v

        K_v, _ = P_c.shape
        K_t, _ = T_c.shape

        P_exp = P_c.unsqueeze(1).expand(K_v, K_t, feat_dim)
        T_exp = T_c.unsqueeze(0).expand(K_v, K_t, feat_dim)

        H_c = alpha_v * P_exp + alpha_t * T_exp
        H_c = F.normalize(H_c, p=2, dim=-1)

        hybrid_prototypes[cls] = H_c.cpu()
        meta_info[cls] = {
            "num_vision_protos": int(K_v),
            "num_text_prompts": int(K_t),
            "alpha_v": alpha_v,
            "alpha_t": alpha_t,
            "hybrid_shape": tuple(H_c.shape),
        }

        print(
            f"Class '{cls}': vision K_v={K_v}, text K_t={K_t}, "
            f"alpha_v={alpha_v:.2f}, alpha_t={alpha_t:.2f}, "
            f"hybrid shape={tuple(H_c.shape)}"
        )

    return {
        "classes": classes,
        "classes_fg": classes,
        "hybrid_prototypes": hybrid_prototypes,
        "feat_dim": int(feat_dim),
        "alpha_v_cfg": dict(alpha_v_cfg),
        "alpha_v_default": float(alpha_v_default),
        "meta_info": meta_info,
        "notes": (
            "Hybrid prototypes built with per-class alpha weights: "
            "H_c,m,k = normalize(alpha_v(c) * P_c,m + (1-alpha_v(c)) * T_c,k)."
        ),
    }
