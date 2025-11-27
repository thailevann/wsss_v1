"""
Vision Prototype Learner for two-stage optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class VisionPrototypeLearner(nn.Module):
    """
    Learnable vision prototypes with automatic dimension handling
    """
    def __init__(self, vis_dict: Dict[str, torch.Tensor], 
                 txt_mat: torch.Tensor, classes: List[str]):
        """
        Args:
            vis_dict: Dictionary of visual prototypes per class
            txt_mat: Text prototype matrix [num_classes, feat_dim]
            classes: List of class names
        """
        super().__init__()
        self.classes = classes
        
        # Auto-detect feature dimension
        first_class = list(vis_dict.keys())[0]
        vis_proto = vis_dict[first_class].float()
        if vis_proto.dim() == 2:
            self.feat_dim = vis_proto.shape[1]
        elif vis_proto.dim() == 1:
            self.feat_dim = vis_proto.shape[0]
        else:
            raise ValueError(f"Unexpected visual prototype shape: {vis_proto.shape}")
        
        # Check text prototype dimension - handle [num_classes, feat_dim] or [feat_dim]
        if txt_mat.dim() == 2:
            # [num_classes, feat_dim] or [num_prompts, feat_dim]
            txt_dim = txt_mat.shape[-1]  # Use last dimension
        elif txt_mat.dim() == 1:
            txt_dim = txt_mat.shape[0]
        else:
            raise ValueError(f"Unexpected text prototype shape: {txt_mat.shape}")
        
        if txt_dim != self.feat_dim:
            raise ValueError(
                f"Dimension mismatch: visual={self.feat_dim}, text={txt_dim}"
            )
        
        self.vision_protos = nn.ParameterDict()
        self.orig_visual_protos = {}
        
        for c in classes:
            V_c = vis_dict[c].float()
            # Ensure dimension consistency
            if V_c.dim() == 2 and V_c.shape[1] != self.feat_dim:
                raise ValueError(
                    f"Dimension mismatch for class {c}: "
                    f"expected {self.feat_dim}, got {V_c.shape[1]}"
                )
            self.vision_protos[c] = nn.Parameter(V_c.clone())
            self.orig_visual_protos[c] = V_c.clone()
        
        self.register_buffer("text_protos", txt_mat.clone())

    def get_data(self, c: str, device: str):
        """
        Get prototypes for a class
        Returns: (vision_proto, orig_vision_proto, text_proto)
        Handles both [feat_dim] and [num_prompts, feat_dim] text prototypes
        """
        idx = self.classes.index(c)
        text_proto = self.text_protos[idx].to(device)
        # If text_proto is [num_prompts, feat_dim], keep it; if [feat_dim], keep it
        return (
            self.vision_protos[c].to(device),
            self.orig_visual_protos[c].to(device),
            text_proto
        )

    def get_all(self, device: str):
        """
        Get all prototypes concatenated
        Returns: (all_protos, class_indices)
        """
        ps, idxs = [], []
        for i, c in enumerate(self.classes):
            p = self.vision_protos[c].to(device)
            ps.append(p)
            idxs.append(torch.full((p.size(0),), i, dtype=torch.long, device=device))
        return torch.cat(ps), torch.cat(idxs)

