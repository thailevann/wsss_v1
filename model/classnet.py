"""
ClassNet++ model definition with ProGrad and hybrid prototype matching.
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProGradFunction(torch.autograd.Function):
    """
    Prototype-guided gradient rectification (soft projection onto noise prototypes).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, noise_protos: torch.Tensor, alpha: float):
        if noise_protos is None or noise_protos.numel() == 0 or alpha <= 0.0:
            ctx.save_for_backward(None)
            return x

        alpha_tensor = torch.tensor(alpha, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(noise_protos, alpha_tensor)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        if not saved or saved[0] is None:
            return grad_output, None, None

        noise_protos, alpha_tensor = saved
        alpha = float(alpha_tensor.item())

        grad = grad_output
        P = F.normalize(noise_protos, p=2, dim=-1)

        B, N, D = grad.shape
        grad_flat = grad.view(-1, D)

        sim = grad_flat @ P.t()
        pos_sim = F.relu(sim)
        proj = pos_sim @ P

        grad_clean = grad_flat - alpha * proj
        return grad_clean.view(B, N, D), None, None


class ProGradLayer(nn.Module):
    """
    Wrapper that applies ProGrad during training.
    """

    def __init__(self, noise_protos: torch.Tensor = None, alpha: float = 1.0):
        super().__init__()
        if noise_protos is not None:
            self.register_buffer("noise_protos", noise_protos)
        else:
            self.register_buffer("noise_protos", torch.empty(0))
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.noise_protos.numel() == 0 or self.alpha <= 0.0:
            return x
        return ProGradFunction.apply(x, self.noise_protos, self.alpha)


class ResidualContextBlock(nn.Module):
    """
    Dilated residual context block combining local and global receptive fields.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False)
        self.conv_local = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.conv_global = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv_out = nn.Conv2d(hidden_dim * 2, in_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.relu(self.conv_reduce(x), inplace=True)
        y_local = F.relu(self.conv_local(y), inplace=True)
        y_global = F.relu(self.conv_global(y), inplace=True)
        y_cat = torch.cat([y_local, y_global], dim=1)
        out = self.conv_out(y_cat)
        out = self.bn(out)
        return self.relu(out + residual)


class ClassNetPP(nn.Module):
    """
    ClassNet++ model that adapts CLIP patch tokens to hybrid prototypes with ProGrad.
    """

    def __init__(
        self,
        hybrid_dict: Dict[str, torch.Tensor],
        classes: List[str],
        input_dim: int,
        proto_dim: int,
        noise_protos: torch.Tensor = None,
        tau: float = 0.25,
        prograd_alpha: float = 1.0,
        logit_scale_init: float = 5.0,
    ):
        super().__init__()
        self.classes = list(classes)
        self.input_dim = int(input_dim)
        self.proto_dim = int(proto_dim)
        self.tau = float(tau)

        if self.tau <= 0:
            raise ValueError("tau must be > 0.")

        # Flatten and cache hybrid prototypes per class
        self._protos: Dict[str, torch.Tensor] = {}
        for c in self.classes:
            if c not in hybrid_dict:
                raise KeyError(f"Class '{c}' not found in hybrid prototype dictionary.")
            H_c = hybrid_dict[c].float()
            if H_c.dim() != 3 or H_c.shape[-1] != self.proto_dim:
                raise ValueError(
                    f"Hybrid prototype for class '{c}' has incompatible shape {tuple(H_c.shape)}; "
                    f"expected [K_v, K_t, {self.proto_dim}]."
                )
            K_v, K_t, _ = H_c.shape
            self._protos[c] = F.normalize(H_c.view(K_v * K_t, self.proto_dim), p=2, dim=-1)

        # Adapter that maps CLIP tokens to prototype space
        self.adapter = nn.Sequential(
            nn.Conv2d(self.input_dim, self.proto_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.proto_dim),
            nn.ReLU(inplace=True),
            ResidualContextBlock(self.proto_dim, hidden_dim=128),
            nn.Conv2d(self.proto_dim, self.proto_dim, kernel_size=1, bias=False),
        )

        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.prograd = ProGradLayer(noise_protos, prograd_alpha) if noise_protos is not None else None

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: [B, N, Din] CLIP patch tokens (CLS removed).
        Returns:
            logits: [B, C]
            act_maps: [B, C, H, W]
            patch_feats: [B, N, proto_dim] normalized token features
        """
        if tokens.dim() != 3:
            raise ValueError(f"Expected tokens with shape [B, N, D], got {tokens.shape}")

        B, N, Din = tokens.shape
        if Din != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {Din}")

        spatial_size = int(math.sqrt(N))
        if spatial_size * spatial_size != N:
            raise ValueError(f"Token count {N} is not a perfect square.")

        x = tokens.permute(0, 2, 1).view(B, Din, spatial_size, spatial_size)
        x = self.adapter(x)
        x = x.view(B, self.proto_dim, N).permute(0, 2, 1)
        x = F.normalize(x, p=2, dim=-1)

        if self.prograd is not None:
            x = self.prograd(x)

        logits, act_maps = [], []
        device = x.device

        for c in self.classes:
            protos_c = self._protos[c].to(device)
            sims = x @ protos_c.t()  # [B, N, K]
            K_p = sims.size(-1)

            cam_tokens = self.tau * (torch.logsumexp(sims / self.tau, dim=-1) - math.log(K_p))
            act_maps.append(cam_tokens.view(B, spatial_size, spatial_size).unsqueeze(1))

            N_t = cam_tokens.size(-1)
            logit_c = self.tau * (torch.logsumexp(cam_tokens / self.tau, dim=-1) - math.log(N_t))
            logits.append(logit_c.unsqueeze(-1))

        logits = torch.cat(logits, dim=-1) * self.logit_scale
        act_maps = torch.cat(act_maps, dim=1)
        return logits, act_maps, x

