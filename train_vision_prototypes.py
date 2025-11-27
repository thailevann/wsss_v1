"""
Script to train vision prototypes with two-stage optimization
"""
import os
import argparse
import torch
import torch.optim as optim
import clip
from prototype.vision_learner import VisionPrototypeLearner
from prototype.text_prototype import build_text_prototypes


def train_stage(vpl_model, classes, device, num_epochs, lambda_text, lambda_vis, 
                lambda_margin, lr, stage_name, margin=0.2):
    """Stage 1: Margin + Visual regularization"""
    optimizer = optim.Adam(vpl_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\n=== {stage_name}: {num_epochs} epochs ===")
    for epoch in range(1, num_epochs + 1):
        vpl_model.train()
        optimizer.zero_grad()
        
        # 1. Text alignment
        l_txt = 0.0
        count = 0
        for c in classes:
            P, V, T = vpl_model.get_data(c, device)
            if P.shape[0] == 0:
                continue
            # Handle different dimensions
            if P.dim() == 1:
                P = P.unsqueeze(0)
            if T.dim() == 0:
                T = T.unsqueeze(0)
            sim_txt = (torch.nn.functional.normalize(P, p=2, dim=-1) * 
                      torch.nn.functional.normalize(T, p=2, dim=-1).unsqueeze(0)).sum(-1)
            l_txt += torch.mean(1.0 - sim_txt)
            count += 1
        if count > 0:
            l_txt /= count
        
        # 2. Visual regularization
        l_vis = 0.0
        for c in classes:
            P, V, _ = vpl_model.get_data(c, device)
            if P.shape[0] == 0:
                continue
            l_vis += torch.nn.functional.mse_loss(P, V)
        if count > 0:
            l_vis /= count
        
        # 3. Margin loss
        P_all, idxs = vpl_model.get_all(device)
        P_n = torch.nn.functional.normalize(P_all, p=2, dim=-1)
        sim_mat = P_n @ P_n.t()
        mask_diff = idxs[:, None] != idxs[None, :]
        l_mar = (torch.nn.functional.relu(sim_mat - margin) * mask_diff).sum() / (mask_diff.sum() + 1e-6)
        
        # Total loss
        loss = lambda_text * l_txt + lambda_vis * l_vis + lambda_margin * l_mar
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"[{stage_name} Ep {epoch}] Loss: {loss.item():.4f} | "
                  f"Text: {l_txt.item():.4f} | Vis: {l_vis.item():.4f} | "
                  f"Margin: {l_mar.item():.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")


def train_stage2_improved(vpl_model, classes, device, num_epochs, lambda_text, 
                          lambda_vis, lambda_margin, lr, stage_name, temp=0.05, margin=0.2):
    """Stage 2: Text Alignment Fine-tune with hard prompt mining"""
    optimizer = optim.Adam(vpl_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\n=== {stage_name}: {num_epochs} epochs ===")
    for epoch in range(1, num_epochs + 1):
        vpl_model.train()
        optimizer.zero_grad()
        
        # 1. Hard Prompt / Weighted Text Alignment
        l_txt = 0.0
        count = 0
        for c in classes:
            P, V, T_all = vpl_model.get_data(c, device)
            if P.shape[0] == 0:
                continue
            
            # Ensure 2D
            if P.dim() == 1:
                P = P.unsqueeze(0)
            if T_all.dim() == 1:
                T_all = T_all.unsqueeze(0)
            
            # Normalize
            P_n = torch.nn.functional.normalize(P, p=2, dim=-1)
            T_n = torch.nn.functional.normalize(T_all, p=2, dim=-1)
            
            # Cosine similarity scaled by temperature
            sim = P_n @ T_n.t() / temp  # [num_proto, num_prompt]
            
            # Hard prompt mining
            sim_max, _ = sim.max(dim=1, keepdim=True)
            hard_mask = sim < sim_max
            sim_hard = sim * hard_mask.float()
            
            # Compute loss per prototype
            denom = hard_mask.sum(dim=1).clamp(min=1)
            loss_proto = (1.0 - sim_hard).sum(dim=1) / denom
            l_txt += loss_proto.mean()
            count += 1
        if count > 0:
            l_txt /= count
        
        # 2. Visual regularization
        l_vis = 0.0
        for c in classes:
            P, V, _ = vpl_model.get_data(c, device)
            if P.shape[0] == 0:
                continue
            l_vis += torch.nn.functional.mse_loss(P, V)
        if count > 0:
            l_vis /= count
        
        # 3. Hard Negative Margin
        P_all, idxs = vpl_model.get_all(device)
        P_n_all = torch.nn.functional.normalize(P_all, p=2, dim=-1)
        sim_mat = P_n_all @ P_n_all.t()
        mask_diff = idxs[:, None] != idxs[None, :]
        hard_mask_margin = (sim_mat > margin) * mask_diff
        l_mar = (torch.nn.functional.relu(sim_mat - margin) * hard_mask_margin).sum() / (hard_mask_margin.sum() + 1e-6)
        
        # Total loss
        loss = lambda_text * l_txt + lambda_vis * l_vis + lambda_margin * l_mar
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"[{stage_name} Ep {epoch}] Loss: {loss.item():.4f} | "
                  f"Text: {l_txt.item():.4f} | Vis: {l_vis.item():.4f} | "
                  f"Margin: {l_mar.item():.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Train vision prototypes")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory for data")
    parser.add_argument("--text_proto_path", type=str, default=None,
                       help="Text prototype path (default: data_root/text_prototypes_clip.pt)")
    parser.add_argument("--visual_proto_path", type=str, default=None,
                       help="Visual prototype path (default: data_root/visual_prototypes_clip.pt)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num_epochs_stage1", type=int, default=100,
                       help="Number of epochs for stage 1")
    parser.add_argument("--num_epochs_stage2", type=int, default=100,
                       help="Number of epochs for stage 2")
    parser.add_argument("--lr_stage1", type=float, default=0.02,
                       help="Learning rate for stage 1")
    parser.add_argument("--lr_stage2", type=float, default=0.01,
                       help="Learning rate for stage 2")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: data_root/vision_prototypes_learned.pt)")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    classes_no_bg = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis"]
    
    # Load CLIP
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval().to(device).float()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("CLIP Frozen.")
    
    # Load prototypes
    text_proto_path = args.text_proto_path or os.path.join(args.data_root, "text_prototypes_clip.pt")
    visual_proto_path = args.visual_proto_path or os.path.join(args.data_root, "visual_prototypes_clip.pt")
    
    text_proto_bank = torch.load(text_proto_path, map_location=device)
    visual_proto_bank = torch.load(visual_proto_path, map_location=device)
    
    # Auto-detect feat_dim
    if 'feat_dim' in text_proto_bank:
        feat_dim = int(text_proto_bank['feat_dim'])
    elif 'feat_dim' in visual_proto_bank.get('metadata', {}):
        feat_dim = int(visual_proto_bank['metadata']['feat_dim'])
    else:
        # Detect from first prototype
        first_class = classes_no_bg[0]
        proto = text_proto_bank["class_text_prototypes"][first_class]
        if proto.dim() == 1:
            feat_dim = proto.shape[0]
        elif proto.dim() == 2:
            feat_dim = proto.shape[1]
        else:
            raise ValueError(f"Cannot detect feat_dim from shape {proto.shape}")
    
    print(f"Using feature dimension: {feat_dim}")
    
    # Build text prototype matrix
    text_protos = torch.stack([
        text_proto_bank["class_text_prototypes"][c].float() for c in classes_no_bg
    ], dim=0)
    text_protos = torch.nn.functional.normalize(text_protos, p=2, dim=-1).to(device)
    
    # Ensure dimensions match
    if text_protos.shape[-1] != feat_dim:
        raise ValueError(
            f"Text prototype dimension mismatch: expected {feat_dim}, "
            f"got {text_protos.shape[-1]}"
        )
    
    # Initialize Vision Prototype Learner
    vpl_model = VisionPrototypeLearner(
        visual_proto_bank["visual_prototypes"],
        text_protos,
        classes_no_bg
    ).to(device)
    
    # Stage 1: Margin + Visual
    train_stage(
        vpl_model, classes_no_bg, device,
        num_epochs=args.num_epochs_stage1,
        lambda_text=0.0,
        lambda_vis=0.01,
        lambda_margin=2.0,
        lr=args.lr_stage1,
        stage_name="Stage1-Stability"
    )
    
    # Stage 2: Text Alignment Fine-tune
    train_stage2_improved(
        vpl_model, classes_no_bg, device,
        num_epochs=args.num_epochs_stage2,
        lambda_text=1.0,
        lambda_vis=0.005,
        lambda_margin=0.5,
        lr=args.lr_stage2,
        stage_name="Stage2-TextAlign-Improved",
        temp=0.05
    )
    
    # Save optimized prototypes
    output_path = args.output or os.path.join(args.data_root, "vision_prototypes_learned.pt")
    out = {}
    for c in classes_no_bg:
        out[c] = torch.nn.functional.normalize(
            vpl_model.vision_protos[c].detach().cpu().float(), p=2, dim=-1
        )
    
    torch.save({
        "classes": classes_no_bg,
        "vision_prototypes_learned": out,
        "feat_dim": feat_dim,
        "config": {"strategy": "TwoStage_ExtraOpt"}
    }, output_path)
    
    print(f"\n[OK] Saved Two-Stage optimized prototypes to: {output_path}")


if __name__ == "__main__":
    main()

