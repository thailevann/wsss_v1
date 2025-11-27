"""
Hybrid prototype building
"""
import os
import torch
import torch.nn.functional as F
from typing import Dict


def build_hybrid_prototypes(
    text_proto_path: str,
    vision_proto_path: str,
    feat_dim: int = None
):
    """
    Build hybrid prototypes from text and vision prototypes.
    
    Args:
        text_proto_path: Path to text prototype file
        vision_proto_path: Path to learned vision prototype file
        feat_dim: Feature dimension (auto-detected if None)
    
    Returns:
        Dictionary with hybrid prototypes and metadata
    """
    if not os.path.exists(text_proto_path):
        raise FileNotFoundError(f"Text prototype file not found at {text_proto_path}")
    if not os.path.exists(vision_proto_path):
        raise FileNotFoundError(f"Learned vision prototypes file not found at {vision_proto_path}")

    # Load prototype banks
    text_proto_bank = torch.load(text_proto_path, map_location='cpu')
    vision_proto_bank = torch.load(vision_proto_path, map_location='cpu')

    # Get dimensions
    if feat_dim is None:
        if 'feat_dim' in text_proto_bank:
            feat_dim = int(text_proto_bank['feat_dim'])
        elif 'feat_dim' in vision_proto_bank:
            feat_dim = int(vision_proto_bank['feat_dim'])
        else:
            # Auto-detect from first prototype
            first_class = list(text_proto_bank['class_text_prototypes'].keys())[0]
            proto = text_proto_bank['class_text_prototypes'][first_class]
            if proto.dim() == 1:
                feat_dim = proto.shape[0]
            elif proto.dim() == 2:
                feat_dim = proto.shape[-1]  # Use last dimension (handles [2, 512] case)
            else:
                raise ValueError(f"Cannot detect feature dimension from shape {proto.shape}")

    per_prompt_text_features: Dict[str, torch.Tensor] = text_proto_bank["per_prompt_text_features"]
    class_text_prototypes: Dict[str, torch.Tensor] = text_proto_bank["class_text_prototypes"]

    vision_prototypes_learned: Dict[str, torch.Tensor] = vision_proto_bank["vision_prototypes_learned"]
    classes = vision_proto_bank["classes"]

    print("Classes in vision prototype bank:", classes)
    print("Feature dim:", feat_dim)

    # Build hybrid prototypes H_c,m,k = normalize((P_c,m + T_c,k) / 2)
    hybrid_prototypes: Dict[str, torch.Tensor] = {}
    metadata = {}

    for c in classes:
        if c not in vision_prototypes_learned:
            raise KeyError(f"Class '{c}' not found in learned vision prototypes.")
        if c not in per_prompt_text_features:
            raise KeyError(f"Class '{c}' not found in per_prompt_text_features.")

        P_c = vision_prototypes_learned[c].float()          # [K_v, D]
        T_c_prompts = per_prompt_text_features[c].float()   # [K_t, D] or [D] or [2, D]
        
        # Ensure P_c is 2D [K_v, D]
        if P_c.ndim == 1:
            P_c = P_c.unsqueeze(0)  # [1, D]
        elif P_c.ndim != 2:
            raise ValueError(f"Vision prototypes for {c} must be 1D or 2D, got {P_c.shape}")
        
        # Ensure T_c_prompts is 2D [K_t, D]
        if T_c_prompts.ndim == 1:
            T_c_prompts = T_c_prompts.unsqueeze(0)  # [1, D]
        elif T_c_prompts.ndim != 2:
            raise ValueError(f"Text prompt features for {c} must be 1D or 2D, got {T_c_prompts.shape}")
        
        # Check dimensions - use last dimension for feat_dim
        if P_c.shape[-1] != feat_dim:
            raise ValueError(
                f"Dimension mismatch for class {c} vision: "
                f"expected {feat_dim}, got {P_c.shape[-1]}"
            )
        if T_c_prompts.shape[-1] != feat_dim:
            raise ValueError(
                f"Dimension mismatch for class {c} text: "
                f"expected {feat_dim}, got {T_c_prompts.shape[-1]}"
            )

        K_v, D = P_c.shape
        K_t, _ = T_c_prompts.shape

        # Broadcast: [K_v,1,D] + [1,K_t,D] -> [K_v,K_t,D]
        P_exp = P_c.unsqueeze(1).expand(K_v, K_t, D)
        T_exp = T_c_prompts.unsqueeze(0).expand(K_v, K_t, D)

        H = (P_exp + T_exp) * 0.5
        H = F.normalize(H, p=2, dim=-1)  # [K_v,K_t,D]

        hybrid_prototypes[c] = H.cpu()
        metadata[c] = {
            "num_vision": int(K_v),
            "num_text_prompts": int(K_t),
        }

        print(f"Class '{c}': vision prototypes = {K_v}, text prompts = {K_t}, hybrid shape = {tuple(H.shape)}")

    return {
        "classes": classes,
        "hybrid_prototypes": hybrid_prototypes,  # dict[class] -> [K_v, K_t, D]
        "feat_dim": int(feat_dim),
        "metadata": metadata,
        "notes": (
            "H_c,m,k = normalize((P_c,m + T_c,k)/2). "
            "You can flatten to [K_v*K_t, D] or further learn hybrid MLP in Stage 2."
        ),
    }

