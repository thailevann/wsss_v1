"""
ClassNet++ model definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ResidualContextBlock(nn.Module):
    """Dilated Residual Block for Context Awareness"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True)
        )
        # Branch 1: Local details
        self.local = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True)
        )
        # Branch 2: Global context
        self.global_ctx = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        res = x
        out = self.reduce(x)
        l = self.local(out)
        g = self.global_ctx(out)
        out = torch.cat([l, g], dim=1)
        out = self.proj(out)
        return self.relu(out + res)


class ClassNetPP(nn.Module):
    """
    ClassNet++ model with automatic dimension handling
    """
    def __init__(self, hybrid_dict: Dict[str, torch.Tensor], classes: list, 
                 input_dim: int, proto_dim: int = None):
        """
        Args:
            hybrid_dict: Dictionary of hybrid prototypes per class
            classes: List of class names
            input_dim: Input dimension from CLIP (auto-detected)
            proto_dim: Prototype dimension (auto-detected from hybrid_dict if None)
        """
        super().__init__()
        self.classes = classes
        self.input_dim = input_dim
        
        # Auto-detect proto_dim from hybrid_dict if not provided
        if proto_dim is None:
            first_class = list(hybrid_dict.keys())[0]
            H = hybrid_dict[first_class].float()
            if H.dim() == 3:
                proto_dim = H.shape[2]  # [K_v, K_t, D]
            elif H.dim() == 2:
                proto_dim = H.shape[1]  # [K, D]
            elif H.dim() == 1:
                proto_dim = H.shape[0]  # [D]
            else:
                raise ValueError(f"Unexpected hybrid prototype shape: {H.shape}")
        
        self.proto_dim = proto_dim

        # Store Hybrid Prototypes - automatically flatten if needed
        self._protos = {}
        for c in classes:
            H = hybrid_dict[c].float()
            K_v, K_t, D = H.shape
            # Ensure dimension matches
            if D != proto_dim:
                raise ValueError(
                    f"Dimension mismatch for class {c}: "
                    f"expected {proto_dim}, got {D}"
                )
            self._protos[c] = F.normalize(H.view(K_v * K_t, D), p=2, dim=-1)

        # Advanced Adapter
        self.adapter = nn.Sequential(
            nn.Conv2d(input_dim, proto_dim, 1, bias=False),  # Project
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(True),
            ResidualContextBlock(proto_dim, hidden_dim=128),  # Context
            nn.Conv2d(proto_dim, proto_dim, 1)  # Smooth
        )
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, patch_feats):
        """
        Args:
            patch_feats: [B, N, Din] patch features from CLIP
        Returns:
            logits: [B, num_classes] classification logits
            act_maps: [B, num_classes, H, W] activation maps
            patch_feats_norm: [B, N, proto_dim] normalized patch features
        """
        B, N, Din = patch_feats.shape
        
        # Ensure input dimension matches
        if Din != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {Din}"
            )
        
        H = W = int(N ** 0.5)

        # [B, N, D] -> [B, D, H, W]
        x = patch_feats.permute(0, 2, 1).view(B, Din, H, W)
        x = self.adapter(x)  # [B, proto_dim, H, W]

        # Back to [B, N, D]
        x = x.view(B, self.proto_dim, N).permute(0, 2, 1)
        x = F.normalize(x, p=2, dim=-1)

        logits, act_maps = [], []
        for c in self.classes:
            proto = self._protos[c].to(x.device)  # [K, D]
            sims = x @ proto.t()                   # [B, N, K]

            # CAM
            A = sims.max(dim=-1).values            # [B, N]
            act_maps.append(A.view(B, H, W).unsqueeze(1))

            # Logit
            logits.append(A.max(dim=-1).values.unsqueeze(-1))

        return torch.cat(logits, dim=-1) * self.logit_scale, torch.cat(act_maps, dim=1), x

