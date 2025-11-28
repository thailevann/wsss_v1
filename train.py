"""
Train ClassNet++ with ProGrad and background-aware regularization.
"""
import os
import glob
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import clip
from model.classnet import ClassNetPP
from model.clip_encoder import encode_image_to_tokens
from dataloader import load_training_data, create_evaluation_dataloader, build_targets
from utils import compute_presence_metrics


# -----------------------------------------------------------------------------
# Loss functions & utilities
# -----------------------------------------------------------------------------
def proto_contrastive_loss(
    feat_tokens: torch.Tensor,
    act_maps: torch.Tensor,
    targets: torch.Tensor,
    class_mean_protos: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """
    Encourage tokens of positive classes to stay close to their class mean prototypes.
    """
    B, N, _ = feat_tokens.shape
    C = class_mean_protos.size(0)

    act = act_maps.view(B, C, -1)
    pos_feats, pos_cls = [], []

    for b in range(B):
        for c in range(C):
            if targets[b, c] < 0.5:
                continue
            scores = act[b, c]
            if scores.max() <= 0:
                continue
            k = min(32, scores.numel())
            idx = scores.topk(k).indices
            pos_feats.append(feat_tokens[b, idx])
            pos_cls.append(torch.full((k,), c, dtype=torch.long, device=feat_tokens.device))

    if not pos_feats:
        return feat_tokens.sum() * 0.0

    pos_feats = torch.cat(pos_feats, dim=0)
    pos_cls = torch.cat(pos_cls, dim=0)

    if pos_feats.size(0) > 1024:
        select = torch.randperm(pos_feats.size(0), device=feat_tokens.device)[:1024]
        pos_feats = pos_feats[select]
        pos_cls = pos_cls[select]

    pos_feats = F.normalize(pos_feats, p=2, dim=-1)
    mean_protos = F.normalize(class_mean_protos, p=2, dim=-1)

    pos_proto = mean_protos[pos_cls]
    sim_pos = (pos_feats * pos_proto).sum(-1)

    rand_shift = torch.randint(1, C, (pos_cls.size(0),), device=feat_tokens.device)
    neg_idx = (pos_cls + rand_shift) % C
    neg_proto = mean_protos[neg_idx]
    sim_neg = (pos_feats * neg_proto).sum(-1)

    return F.relu(margin - (sim_pos - sim_neg)).mean()


def explicit_noise_penalty(features: torch.Tensor, noise_protos: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Penalize similarity between learnt tokens and background noise prototypes.
    """
    if noise_protos is None or noise_protos.numel() == 0:
        return features.sum() * 0.0

    feat_flat = features.view(-1, features.size(-1))
    if feat_flat.size(0) > 2048:
        idx = torch.randperm(feat_flat.size(0), device=features.device)[:2048]
        feat_flat = feat_flat[idx]

    feat_flat = F.normalize(feat_flat, p=2, dim=-1)
    noise_norm = F.normalize(noise_protos, p=2, dim=-1)

    sim = feat_flat @ noise_norm.t()
    return F.relu(sim - margin).mean()


def bg_margin_loss(
    features: torch.Tensor,
    class_mean_protos: torch.Tensor,
    noise_protos: torch.Tensor,
    sim_threshold: float,
    margin: float,
) -> torch.Tensor:
    """
    Force tokens that look like background to stay away from tissue classes by a margin.
    """
    if noise_protos is None or noise_protos.numel() == 0:
        return features.sum() * 0.0

    B, N, D = features.shape
    C = class_mean_protos.size(0)

    feat_norm = F.normalize(features.view(-1, D), p=2, dim=-1)
    bg_norm = F.normalize(noise_protos, p=2, dim=-1)
    cls_norm = F.normalize(class_mean_protos, p=2, dim=-1)

    s_bg = (feat_norm @ bg_norm.t()).max(dim=1).values
    s_fg = feat_norm @ cls_norm.t()

    mask = s_bg > sim_threshold
    if mask.sum() == 0:
        return features.sum() * 0.0

    s_bg_sel = s_bg[mask]
    s_fg_sel = s_fg[mask]
    return F.relu(s_fg_sel - s_bg_sel.unsqueeze(-1) + margin).mean()


@torch.no_grad()
def calculate_noise_similarity(features: torch.Tensor, noise_protos: torch.Tensor) -> float:
    if noise_protos is None or noise_protos.numel() == 0:
        return 0.0

    feat_flat = features.view(-1, features.size(-1))
    if feat_flat.size(0) > 2000:
        idx = torch.randperm(feat_flat.size(0), device=features.device)[:2000]
        feat_flat = feat_flat[idx]

    feat_flat = F.normalize(feat_flat, p=2, dim=-1)
    noise_norm = F.normalize(noise_protos, p=2, dim=-1)
    sim = feat_flat @ noise_norm.t()
    return float(sim.max(dim=1).values.mean().item())


def load_noise_prototypes(
    learned_path: str,
    text_proto_path: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Load background prototypes from learned visual bank, fallback to text prompts.
    """
    noise = None

    if learned_path and os.path.exists(learned_path):
        bank = torch.load(learned_path, map_location="cpu")
        candidates: Dict[str, torch.Tensor] = {}
        if "vision_prototypes_learned" in bank:
            candidates = bank["vision_prototypes_learned"]
        elif "visual_prototypes" in bank:
            candidates = bank["visual_prototypes"]
        bg_key = bank.get("bg_class", "Background")
        if bg_key in candidates:
            noise = candidates[bg_key].float()
        elif "Background" in candidates:
            noise = candidates["Background"].float()

    if noise is None and text_proto_path and os.path.exists(text_proto_path):
        text_bank = torch.load(text_proto_path, map_location="cpu")
        per_prompt = text_bank.get("per_prompt_text_features", {})
        class_proto = text_bank.get("class_text_prototypes", {})
        if "Background" in per_prompt:
            noise = per_prompt["Background"].float()
        elif "Background" in class_proto:
            noise = class_proto["Background"].unsqueeze(0).float()

    if noise is None:
        return None
    return F.normalize(noise, p=2, dim=-1).to(device)


def load_image_batch(
    paths: List[str],
    labels: List[List[str]],
    indices: List[int],
    transform,
) -> Tuple[torch.Tensor, List[List[str]]]:
    images = []
    batch_labels = []

    for idx in indices:
        try:
            img = Image.open(paths[idx]).convert("RGB")
            images.append(transform(img))
            batch_labels.append(labels[idx])
        except Exception:
            continue

    if not images:
        return torch.empty(0), []

    return torch.stack(images, dim=0), batch_labels


def compute_split_val_accuracy(
    model: ClassNetPP,
    clip_model,
    img_paths: List[str],
    labels: List[List[str]],
    indices: List[int],
    transform,
    classes: List[str],
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total_correct = 0.0
    total_elems = 0.0

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            imgs, lbls = load_image_batch(img_paths, labels, batch_idx, transform)
            if imgs.numel() == 0:
                continue

            imgs = imgs.to(device)
            targets = build_targets(lbls, classes, device).float()

            tokens = encode_image_to_tokens(clip_model, imgs)
            logits, _, _ = model(tokens[:, 1:])

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == targets).sum().item()
            total_elems += preds.numel()

    return (total_correct / total_elems * 100.0) if total_elems > 0 else 0.0


def validate(model, val_loader, classes, clip_model, device):
    """Validate model on mask-derived labels."""
    model.eval()
    num_classes = len(classes)
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    tn = torch.zeros(num_classes, device=device)
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) != 4:
                continue

            imgs, _, label_vecs, _ = batch
            if imgs.shape[0] == 0:
                continue
            
            imgs = imgs.to(device)
            label_vecs = label_vecs.to(device).float()

            toks = encode_image_to_tokens(clip_model, imgs)
            logits, _, _ = model(toks[:, 1:])

            preds = (torch.sigmoid(logits) > 0.5).float()

            tp += (preds * label_vecs).sum(dim=0)
            fp += (preds * (1.0 - label_vecs)).sum(dim=0)
            fn += ((1.0 - preds) * label_vecs).sum(dim=0)
            tn += ((1.0 - preds) * (1.0 - label_vecs)).sum(dim=0)
            total_samples += imgs.shape[0]

    if total_samples == 0:
        return None

    metrics = compute_presence_metrics(tp, fp, fn, tn, classes)
    metrics["summary"]["num_samples"] = total_samples
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ClassNet++ with ProGrad")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--train_dir", type=str, default=None, help="Training directory (default: data_root/training)")
    parser.add_argument("--hybrid_proto_path", type=str, default=None,
                       help="Hybrid prototype path (default: data_root/hybrid_prototypes.pt)")
    parser.add_argument("--visual_proto_path", type=str, default=None,
                        help="Learned visual prototype path for background noise (optional)")
    parser.add_argument("--text_proto_path", type=str, default=None,
                        help="Text prototype bank for background fallback (default: data_root/text_prototypes_clip.pt)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_proto", type=float, default=0.5, help="Weight for prototype contrastive loss")
    parser.add_argument("--lambda_push", type=float, default=2.0, help="Weight for explicit noise penalty")
    parser.add_argument("--lambda_bg_margin", type=float, default=0.5, help="Weight for BG margin loss")
    parser.add_argument("--proto_margin", type=float, default=0.2, help="Margin for prototype contrastive loss")
    parser.add_argument("--bg_sim_thr", type=float, default=0.40, help="Similarity threshold to consider background")
    parser.add_argument("--bg_margin", type=float, default=0.05, help="Background margin hinge")
    parser.add_argument("--tau", type=float, default=0.25, help="Softmax temperature for ClassNet++")
    parser.add_argument("--prograd_alpha", type=float, default=0.7, help="Soft ProGrad alpha (0-1)")
    parser.add_argument("--val_split", type=float, default=0.0, help="Validation split for label accuracy monitoring")
    parser.add_argument("--val_img_dir", type=str, default=None,
                        help="Validation image directory (default: data_root/val/img)")
    parser.add_argument("--val_mask_dir", type=str, default=None,
                        help="Validation mask directory (default: data_root/val/mask)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path (default: data_root/classnet_pp_stage2_1_prograd.pt)")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    classes_no_bg = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis"]
    train_dir = args.train_dir or os.path.join(args.data_root, "training")
    val_img_dir = args.val_img_dir or os.path.join(args.data_root, "val", "img")
    val_mask_dir = args.val_mask_dir or os.path.join(args.data_root, "val", "mask")
    val_images = glob.glob(os.path.join(val_img_dir, "*.png"))
    val_masks = glob.glob(os.path.join(val_mask_dir, "*.png"))
    use_external_val = len(val_images) > 0 and len(val_masks) > 0
    val_split = args.val_split
    if use_external_val and val_split > 0:
        print(f"[INFO] Detected external validation set at {val_img_dir}. "
              f"Ignoring val_split={val_split} and using full training set.")
        val_split = 0.0
    
    # Load CLIP
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()
    clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    INPUT_DIM = clip_model.visual.positional_embedding.shape[-1]
    print(f"CLIP Input Dim: {INPUT_DIM}")

    # Transforms matching CLIP statistics with data augmentations for training
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])
    
    # Load hybrid prototypes
    hybrid_proto_path = args.hybrid_proto_path or os.path.join(args.data_root, "hybrid_prototypes.pt")
    if not os.path.exists(hybrid_proto_path):
        raise FileNotFoundError(f"Hybrid prototype file not found: {hybrid_proto_path}")
    
    hybrid_bank = torch.load(hybrid_proto_path, map_location="cpu")
    hybrid_dict = hybrid_bank["hybrid_prototypes"]
    proto_dim = int(hybrid_bank["feat_dim"])
    print(f"Prototype dimension: {proto_dim}")

    # Load noise prototypes (visual -> text fallback)
    learned_visual_path = args.visual_proto_path or os.path.join(args.data_root, "vision_prototypes_learned.pt")
    text_proto_path = args.text_proto_path or os.path.join(args.data_root, "text_prototypes_clip.pt")
    noise_protos = load_noise_prototypes(learned_visual_path, text_proto_path, device)
    if noise_protos is None:
        print("[WARN] No noise prototypes found; ProGrad & BG losses disabled.")

    # Class means for contrastive loss
    class_proto_vecs = []
    for c in classes_no_bg:
        H_c = hybrid_dict[c].float()
        vec = H_c.mean(dim=(0, 1))
        class_proto_vecs.append(F.normalize(vec, p=2, dim=-1))
    class_proto_vecs = torch.stack(class_proto_vecs, dim=0).to(device)

    # Initialize model
    classnet = ClassNetPP(
        hybrid_dict=hybrid_dict,
        classes=classes_no_bg,
        input_dim=INPUT_DIM,
        proto_dim=proto_dim,
        noise_protos=noise_protos,
        tau=args.tau,
        prograd_alpha=args.prograd_alpha,
    ).to(device)

    optimizer = torch.optim.Adam(classnet.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Load training data split
    print("Loading training data...")
    train_img_paths, train_labels, train_indices, val_indices = load_training_data(
        train_dir,
        classes_no_bg,
        val_split=val_split,
        seed=args.seed,
    )
    if val_split > 0:
        print(f"Data Split: Train={len(train_indices)} | Val={len(val_indices)} "
              f"(val_split={val_split:.2f})")
    else:
        print(f"Training samples: {len(train_indices)} (no train/val split applied).")
    
    # Validation loader (mask-driven metrics)
    val_loader = create_evaluation_dataloader(
        val_img_dir,
        val_mask_dir,
        transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ]),
        classes_no_bg,
        batch_size=args.batch_size,
        device="cpu",  # keep dataset tensors on CPU; move to GPU during validation step
    )
    if len(val_loader.dataset) == 0:
        print(f"[WARN] No validation samples found in {val_img_dir} / {val_mask_dir}")

    best_metric = float("-inf")
    best_ckpt_path = args.output or os.path.join(args.data_root, "classnet_pp_stage2_1_prograd.pt")
    noise_for_losses = noise_protos if noise_protos is None else noise_protos.to(device)

    for epoch in range(1, args.num_epochs + 1):
        classnet.train()
        random.shuffle(train_indices)

        metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(range(0, len(train_indices), args.batch_size), desc=f"[Ep {epoch}]")
        for start in pbar:
            batch_ids = train_indices[start:start + args.batch_size]
            imgs, lbls = load_image_batch(train_img_paths, train_labels, batch_ids, train_transform)
            if imgs.numel() == 0:
                continue
            
            imgs = imgs.to(device)
            targets = build_targets(lbls, classes_no_bg, device).float()

            with torch.no_grad():
                tokens = encode_image_to_tokens(clip_model, imgs)

            logits, act_maps, feats = classnet(tokens[:, 1:])

            l_cls = bce_loss(logits, targets)
            l_proto = proto_contrastive_loss(feats, act_maps, targets, class_proto_vecs, args.proto_margin)
            l_push = explicit_noise_penalty(feats, noise_for_losses, margin=0.0)
            l_bg = bg_margin_loss(feats, class_proto_vecs, noise_for_losses, args.bg_sim_thr, args.bg_margin)

            loss = (
                l_cls
                + args.lambda_proto * l_proto
                + args.lambda_push * l_push
                + args.lambda_bg_margin * l_bg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_sim = calculate_noise_similarity(feats, noise_for_losses)

            metrics["loss"] += loss.item()
            metrics["cls"] += l_cls.item()
            metrics["proto"] += l_proto.item()
            metrics["push"] += l_push.item()
            metrics["bg_margin"] += l_bg.item()
            metrics["noise_sim"] += noise_sim
            num_batches += 1

            pbar.set_postfix({
                "L": f"{loss.item():.3f}",
                "BG": f"{l_bg.item():.3f}",
                "Push": f"{l_push.item():.3f}",
                "NSim": f"{noise_sim:.3f}",
            })

        scheduler.step()

        avg_stats = {k: (v / max(1, num_batches)) for k, v in metrics.items()}
        print(
            f"[Train Ep {epoch}] "
            f"Loss={avg_stats['loss']:.4f} | "
            f"Cls={avg_stats['cls']:.4f} | "
            f"Proto={avg_stats['proto']:.4f} | "
            f"Push={avg_stats['push']:.4f} | "
            f"BG-M={avg_stats['bg_margin']:.4f} | "
            f"NoiseSim={avg_stats['noise_sim']:.4f}"
        )

        # Monitor label accuracy on held-out split
        val_label_acc_metric = None
        if val_indices:
            val_label_acc_metric = compute_split_val_accuracy(
                classnet,
                clip_model,
                train_img_paths,
                train_labels,
                val_indices,
                eval_transform,
                classes_no_bg,
                device,
                args.batch_size,
            )
            print(f"[Val Ep {epoch}] Mean label ACC: {val_label_acc_metric:.2f}%")

        # Mask-driven validation metrics (requested earlier)
        print("Validating...")
        val_results = validate(classnet, val_loader, classes_no_bg, clip_model, device)
        print("-" * 40)
        if val_results is None:
            print("No validation samples available for evaluation.")
        else:
            summary = val_results["summary"]
            per_class = val_results["per_class"]
            print(f"Samples evaluated: {summary.get('num_samples', 0)}")
            for cls, cls_metrics in per_class.items():
                print(
                    f"  >> {cls:25}: "
                    f"Acc {cls_metrics['accuracy']*100:.2f}% | "
                    f"Prec {cls_metrics['precision']*100:.2f}% | "
                    f"Rec {cls_metrics['recall']*100:.2f}% | "
                    f"F1 {cls_metrics['f1']*100:.2f}% | "
                    f"IoU {cls_metrics['iou']*100:.2f}% | "
                    f"Dice {cls_metrics['dice']*100:.2f}% | "
                    f"bIoU {cls_metrics['biou']*100:.2f}%"
                )
            print("-" * 40)
            print(
                "Mean Acc {0:.2f}% | Mean Prec {1:.2f}% | Mean Rec {2:.2f}% | Mean F1 {3:.2f}%".format(
                    summary["mean_accuracy"] * 100,
                    summary["mean_precision"] * 100,
                    summary["mean_recall"] * 100,
                    summary["mean_f1"] * 100,
                )
            )
            print(
                "mIoU {0:.2f}% | FwIoU {1:.2f}% | Mean bIoU {2:.2f}% | Mean Dice {3:.2f}%".format(
                    summary["mIoU"] * 100,
                    summary["FwIoU"] * 100,
                    summary["mean_bIoU"] * 100,
                    summary["mean_dice"] * 100,
                )
            )

            mean_acc = summary["mean_accuracy"] * 100.0
            if mean_acc > best_metric:
                best_metric = mean_acc
                torch.save(
                    {
                        "state_dict": classnet.state_dict(),
                        "config": {
                            "input_dim": INPUT_DIM,
                            "proto_dim": proto_dim,
                            "classes": classes_no_bg,
                            "tau": args.tau,
                            "prograd_alpha": args.prograd_alpha,
                            "lambda_proto": args.lambda_proto,
                            "lambda_push": args.lambda_push,
                            "lambda_bg_margin": args.lambda_bg_margin,
                            "bg_sim_thr": args.bg_sim_thr,
                            "bg_margin": args.bg_margin,
                            "val_label_acc": (
                                val_label_acc_metric if val_label_acc_metric is not None else mean_acc
                            ),
                        },
                    },
                    best_ckpt_path,
                )
                print(f"[Save] New best checkpoint: {mean_acc:.2f}% -> {best_ckpt_path}")
        print("-" * 40)
    
    if best_metric == float("-inf"):
        # No validation performed; save final weights.
        torch.save(
            {
        "state_dict": classnet.state_dict(),
                "config": {
                    "input_dim": INPUT_DIM,
                    "proto_dim": proto_dim,
                    "classes": classes_no_bg,
                    "tau": args.tau,
                    "prograd_alpha": args.prograd_alpha,
                },
            },
            best_ckpt_path,
        )
        print(f"[Warn] Validation skipped. Final model saved to {best_ckpt_path}")
    else:
        print(f"[Done] Best Val Mean Acc = {best_metric:.2f}% | Checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()

