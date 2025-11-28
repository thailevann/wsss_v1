"""
Stage 1.2 – optimize visual prototypes with text alignment, margin, and noise constraints.
"""
import os
import argparse
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# -----------------------------------------------------------------------------#
# Hyper-parameters matching the Stage 1.2 notebook pipeline
# -----------------------------------------------------------------------------#
NUM_EPOCHS_EXT = 200
LR_EXT = 0.02
PATIENCE = 25

LAMBDA_TEXT_ALIGN = 0.5
LAMBDA_VIS_ANCHOR = 0.01
LAMBDA_MARGIN = 2.0
LAMBDA_DIVERSITY = 0.01
LAMBDA_NOISE_PUSH = 0.8
MARGIN_INTER_CLASS = 0.2
NOISE_EPS = 0.05


def gram_schmidt_basis(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Construct an orthonormal basis from the given vectors."""
    basis: List[torch.Tensor] = []
    for v in vectors:
        w = v.clone()
        for b in basis:
            w = w - torch.dot(w, b) * b
        norm = w.norm(p=2)
        if norm > eps:
            basis.append(w / norm)
    if not basis:
        return torch.empty(0, vectors.size(-1), device=vectors.device)
    return torch.stack(basis, dim=0)


class VisionPrototypeLearner(nn.Module):
    """
    Learnable visual prototypes with text targets and optional noise basis.
    """

    def __init__(
        self,
        vis_dict: Dict[str, torch.Tensor],
        text_mat: torch.Tensor,
        classes: List[str],
        noise_basis: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.classes = classes
        self.register_buffer("text_protos", text_mat.clone())

        if noise_basis is not None and noise_basis.numel() > 0:
            self.register_buffer("noise_protos", F.normalize(noise_basis.clone(), p=2, dim=-1))
        else:
            self.noise_protos = None

        self.vision_protos = nn.ParameterDict()
        self.orig_visual_protos: Dict[str, torch.Tensor] = {}

        for cls in classes:
            V = vis_dict[cls].float()
            if V.ndim == 1:
                V = V.unsqueeze(0)
            self.vision_protos[cls] = nn.Parameter(V.clone())
            self.orig_visual_protos[cls] = V.clone()

    def get_data(self, cls: str, device: torch.device):
        P = self.vision_protos[cls].to(device)
        V = self.orig_visual_protos[cls].to(device)
        idx = self.classes.index(cls)
        T = self.text_protos[idx].to(device)
        return P, V, T

    def get_all(self, device: torch.device):
        protos, indices = [], []
        for i, cls in enumerate(self.classes):
            p = self.vision_protos[cls].to(device)
            protos.append(p)
            indices.append(torch.full((p.size(0),), i, dtype=torch.long, device=device))
        return torch.cat(protos, dim=0), torch.cat(indices, dim=0)


def text_align_loss(P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    Pn = F.normalize(P, p=2, dim=-1)
    Tn = F.normalize(T, p=2, dim=-1)
    sim = (Pn * Tn.unsqueeze(0)).sum(-1)
    return (1.0 - sim).mean()


def intra_class_diversity_loss(P: torch.Tensor) -> torch.Tensor:
    if P.size(0) <= 1:
        return torch.tensor(0.0, device=P.device)
    Pn = F.normalize(P, p=2, dim=-1)
    sim = Pn @ Pn.t()
    I = torch.eye(P.size(0), device=P.device)
    return ((sim - I) ** 2).mean()


def inter_class_margin_loss(P_all: torch.Tensor, idxs: torch.Tensor, margin: float) -> torch.Tensor:
    Pn = F.normalize(P_all, p=2, dim=-1)
    sim = Pn @ Pn.t()
    mask = (idxs[:, None] != idxs[None, :]).float()
    violation = F.relu(sim - margin) * mask
    return violation.sum() / (mask.sum() + 1e-6)


def noise_push_loss(P_all: torch.Tensor, noise_mat: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    if noise_mat is None or noise_mat.numel() == 0:
        return torch.tensor(0.0, device=P_all.device)
    Pn = F.normalize(P_all, p=2, dim=-1)
    Nn = F.normalize(noise_mat, p=2, dim=-1)
    sim = Pn @ Nn.t()
    violation = F.relu(sim - eps)
    return violation.mean()


def main():
    parser = argparse.ArgumentParser(description="Stage 1.2 visual prototype optimization")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--text_proto_path", type=str, default=None,
                        help="Path to text prototypes (default: data_root/text_prototypes_clip.pt)")
    parser.add_argument("--visual_proto_path", type=str, default=None,
                        help="Path to visual prototypes (default: data_root/visual_prototypes_clip.pt)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: data_root/vision_prototypes_learned.pt)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_EXT, help="Max epochs")
    parser.add_argument("--lr", type=float, default=LR_EXT, help="Learning rate")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--noise_eps", type=float, default=NOISE_EPS, help="Noise cosine threshold")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    text_proto_path = args.text_proto_path or os.path.join(args.data_root, "text_prototypes_clip.pt")
    visual_proto_path = args.visual_proto_path or os.path.join(args.data_root, "visual_prototypes_clip.pt")
    
    if not os.path.exists(text_proto_path) or not os.path.exists(visual_proto_path):
        raise FileNotFoundError("Missing prototype banks. Run Stage 0 and Stage 1.1 first.")

    text_proto_bank = torch.load(text_proto_path, map_location="cpu")
    visual_proto_bank = torch.load(visual_proto_path, map_location="cpu")

    classes_fg = text_proto_bank.get("classes_fg") or visual_proto_bank.get("classes") or \
        ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis"]
    bg_class = visual_proto_bank.get("bg_class", "Background")

    class_text_dict: Dict[str, torch.Tensor] = text_proto_bank["class_text_prototypes"]
    per_prompt_text: Dict[str, torch.Tensor] = text_proto_bank["per_prompt_text_features"]

    if "visual_prototypes" in visual_proto_bank:
        vis_dict_raw = visual_proto_bank["visual_prototypes"]
    else:
        vis_dict_raw = {c: visual_proto_bank[c] for c in classes_fg}
    bg_visual = visual_proto_bank.get("background_prototypes")

    feat_dim = int(text_proto_bank.get("feat_dim")
                   or visual_proto_bank.get("feat_dim")
                   or list(class_text_dict.values())[0].shape[-1])

    text_matrix = []
    for cls in classes_fg:
        proto = class_text_dict[cls].float()
        if proto.dim() == 1:
            proto = proto.unsqueeze(0)
        text_matrix.append(F.normalize(proto.mean(dim=0, keepdim=True), p=2, dim=-1))
    text_matrix = torch.cat(text_matrix, dim=0).to(device)

    vis_dict: Dict[str, torch.Tensor] = {}
    for cls in classes_fg:
        if cls not in vis_dict_raw:
            raise KeyError(f"Class '{cls}' missing in visual prototype bank.")
        V = vis_dict_raw[cls].float()
        if V.ndim == 1:
            V = V.unsqueeze(0)
        vis_dict[cls] = F.normalize(V, p=2, dim=-1)

    # Build noise pool (visual BG, text BG, noise classes)
    noise_vectors = []
    if bg_visual is not None:
        noise_vectors.append(F.normalize(bg_visual.float(), p=2, dim=-1))
    elif bg_class in vis_dict_raw:
        noise_vectors.append(F.normalize(vis_dict_raw[bg_class].float(), p=2, dim=-1))

    if bg_class in class_text_dict:
        noise_vectors.append(F.normalize(class_text_dict[bg_class].float(), p=2, dim=-1))

    for noise_cls in text_proto_bank.get("noise_classes", []):
        if noise_cls != bg_class and noise_cls in class_text_dict:
            noise_vectors.append(F.normalize(class_text_dict[noise_cls].float(), p=2, dim=-1))

    if noise_vectors:
        noise_raw = torch.cat(
            [vec if vec.ndim == 2 else vec.unsqueeze(0) for vec in noise_vectors], dim=0
        )
        noise_raw = F.normalize(noise_raw, p=2, dim=-1)
        noise_basis = gram_schmidt_basis(noise_raw)
        print(f"[Stage 1.2] Collected {noise_raw.shape[0]} raw noise prototypes.")
        if noise_basis.numel() == 0:
            print("[Stage 1.2] WARNING: Gram-Schmidt produced empty noise basis.")
            noise_basis = None
        else:
            print(f"[Stage 1.2] Orthonormal noise basis size: {noise_basis.shape[0]}")
    else:
        noise_raw = None
        noise_basis = None
        print("[Stage 1.2] WARNING: No noise prototypes identified.")

    model = VisionPrototypeLearner(
        vis_dict=vis_dict,
        text_mat=text_matrix,
        classes=classes_fg,
        noise_basis=noise_basis,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    patience_counter = 0

    print("\n[Stage 1.2] Start Optimization Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        l_txt = 0.0
        l_vis = 0.0
        l_div = 0.0

        for cls in classes_fg:
            P, V, T = model.get_data(cls, device)
            l_txt += text_align_loss(P, T)
            l_vis += F.mse_loss(P, V)
            l_div += intra_class_diversity_loss(P)

        l_txt /= len(classes_fg)
        l_vis /= len(classes_fg)
        l_div /= len(classes_fg)

        P_all, idx_all = model.get_all(device)
        l_mar = inter_class_margin_loss(P_all, idx_all, MARGIN_INTER_CLASS)
        l_noise = noise_push_loss(P_all, model.noise_protos, args.noise_eps)

        total_loss = (
            LAMBDA_TEXT_ALIGN * l_txt +
            LAMBDA_VIS_ANCHOR * l_vis +
            LAMBDA_DIVERSITY * l_div +
            LAMBDA_MARGIN * l_mar +
            LAMBDA_NOISE_PUSH * l_noise
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[Ep {epoch:3d}] Loss={total_loss.item():.4f} | "
                f"Txt={l_txt.item():.4f} | NoisePush={l_noise.item():.4f} | "
                f"Margin={l_mar.item():.4f}"
            )

        if total_loss.item() < best_loss - 1e-4:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Stage 1.2] Early stopping at epoch {epoch}. Best Loss: {best_loss:.4f}")
                break

    learned_protos: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for cls in classes_fg:
            P = model.vision_protos[cls].detach().cpu()
            learned_protos[cls] = F.normalize(P, p=2, dim=-1)
        if bg_visual is not None:
            learned_protos[bg_class] = F.normalize(bg_visual.float(), p=2, dim=-1)

    output_path = args.output or os.path.join(args.data_root, "vision_prototypes_learned.pt")
    save_dict = {
        "classes": classes_fg,
        "bg_class": bg_class,
        "vision_prototypes_learned": learned_protos,
        "feat_dim": feat_dim,
        "noise_raw": noise_raw.cpu() if noise_raw is not None else None,
        "noise_basis": noise_basis.cpu() if noise_basis is not None else None,
        "config": {
            "lambdas": {
                "text": LAMBDA_TEXT_ALIGN,
                "vis": LAMBDA_VIS_ANCHOR,
                "div": LAMBDA_DIVERSITY,
                "margin": LAMBDA_MARGIN,
                "noise": LAMBDA_NOISE_PUSH,
            },
            "noise_eps": args.noise_eps,
            "note": (
                "Stage 1.2 optimized foreground prototypes with orthonormal noise basis "
                "for downstream ProGrad."
            ),
        },
    }
    os.makedirs(args.data_root, exist_ok=True)
    torch.save(save_dict, output_path)
    print(f"\n[Stage 1.2] [SUCCESS] Saved optimized prototypes to: {output_path}")


if __name__ == "__main__":
    main()

