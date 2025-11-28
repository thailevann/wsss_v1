"""
Script to train ClassNet++ model
"""
import os
import argparse
import random
import numpy as np
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import clip
from model.classnet import ClassNetPP
from model.clip_encoder import encode_image_to_tokens
from dataloader import (
    load_training_data,
    create_training_dataloader,
    create_evaluation_dataloader,
    build_targets,
    load_batch_images,
)
from utils import compute_presence_metrics


def proto_contrastive_loss(feat, act, targets, protos, margin=0.2):
    """Contrastive loss for prototypes"""
    B, N, D = feat.shape
    act = act.view(B, len(protos), N)
    pos_feats, pos_cls = [], []

    for b in range(B):
        for c in range(len(protos)):
            if targets[b, c] < 0.5:
                continue
            scores = act[b, c]
            if scores.max() <= 0:
                continue

            # Select Top-K regions for positive sampling
            k = min(32, N)
            idx = scores.topk(k).indices
            pos_feats.append(feat[b, idx])
            pos_cls.append(torch.full((k,), c, dtype=torch.long, device=feat.device))

    if not pos_feats:
        return torch.tensor(0.0, device=feat.device)

    pos_feats = torch.cat(pos_feats)  # [M, D]
    pos_cls = torch.cat(pos_cls)      # [M]

    # Sampling limits
    if pos_feats.shape[0] > 512:
        perm = torch.randperm(pos_feats.shape[0])[:512]
        pos_feats, pos_cls = pos_feats[perm], pos_cls[perm]

    # Contrastive
    pos_feats = F.normalize(pos_feats, p=2, dim=-1)
    protos = F.normalize(protos, p=2, dim=-1)

    pos_p = protos[pos_cls]

    # Negative mining
    rand_idx = torch.randint(0, len(protos), (pos_cls.shape[0],), device=feat.device)
    neg_idx = torch.where(rand_idx == pos_cls, (rand_idx + 1) % len(protos), rand_idx)
    neg_p = protos[neg_idx]

    sim_pos = (pos_feats * pos_p).sum(-1)
    sim_neg = (pos_feats * neg_p).sum(-1)

    return F.relu(margin - (sim_pos - sim_neg)).mean()


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


def main():
    parser = argparse.ArgumentParser(description="Train ClassNet++")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory for data")
    parser.add_argument("--train_dir", type=str, default=None,
                       help="Training directory (default: data_root/training)")
    parser.add_argument("--hybrid_proto_path", type=str, default=None,
                       help="Hybrid prototype path (default: data_root/hybrid_prototypes.pt)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--lambda_proto", type=float, default=1.0,
                       help="Weight for prototype contrastive loss")
    parser.add_argument("--proto_margin", type=float, default=0.2,
                       help="Margin for contrastive loss")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--val_img_dir", type=str, default=None,
                       help="Validation image directory (default: data_root/val/img)")
    parser.add_argument("--val_mask_dir", type=str, default=None,
                       help="Validation mask directory (default: data_root/val/mask)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                       help="Output checkpoint path (default: data_root/classnet_pp_stage2_1.pt)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    classes_no_bg = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis"]
    train_dir = args.train_dir or os.path.join(args.data_root, "training")
    val_img_dir = args.val_img_dir or os.path.join(args.data_root, "val", "img")
    val_mask_dir = args.val_mask_dir or os.path.join(args.data_root, "val", "mask")
    
    # Load CLIP
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()
    clip_model.to(device)
    clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    # Detect CLIP dimension
    v = clip_model.visual
    pos_emb = v.positional_embedding
    INPUT_DIM = pos_emb.shape[-1]
    print(f"CLIP Input Dim: {INPUT_DIM}")
    
    # Load hybrid prototypes
    hybrid_proto_path = args.hybrid_proto_path or os.path.join(args.data_root, "hybrid_prototypes.pt")
    if not os.path.exists(hybrid_proto_path):
        raise FileNotFoundError(f"Hybrid prototype file not found: {hybrid_proto_path}")
    
    hybrid_bank = torch.load(hybrid_proto_path, map_location="cpu")
    hybrid_dict = hybrid_bank["hybrid_prototypes"]
    PROTO_DIM = int(hybrid_bank["feat_dim"])
    
    print(f"Prototype dimension: {PROTO_DIM}")
    
    # Class Prototypes (Mean for Loss)
    class_proto_vecs = []
    for c in classes_no_bg:
        H_c = hybrid_dict[c].float()
        vec = F.normalize(H_c.mean(dim=(0, 1)), p=2, dim=-1)
        class_proto_vecs.append(vec)
    class_proto_vecs = torch.stack(class_proto_vecs).to(device)
    
    # Initialize Model
    classnet = ClassNetPP(hybrid_dict, classes_no_bg, INPUT_DIM, PROTO_DIM).to(device)
    optimizer = torch.optim.Adam(classnet.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Load data
    print("Loading training data...")
    train_img_paths, train_labels, train_indices, val_indices = load_training_data(
        train_dir,
        classes_no_bg,
        val_split=args.val_split,
        seed=args.seed
    )
    
    print(f"Data Split: Train={len(train_indices)} | Val={len(val_indices)}")

    print("Preparing validation dataloader...")
    val_loader = create_evaluation_dataloader(
        val_img_dir,
        val_mask_dir,
        clip_preprocess,
        classes_no_bg,
        batch_size=args.batch_size,
        device=device
    )
    if len(val_loader.dataset) == 0:
        print(f"[WARN] No validation samples found in {val_img_dir} / {val_mask_dir}")
    
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        classnet.train()
        metrics = {"loss": 0, "cls": 0, "proto": 0}
        n_batches = 0

        random.shuffle(train_indices)
        pbar = tqdm(range(0, len(train_indices), args.batch_size), 
                   desc=f"Epoch {epoch}/{args.num_epochs}")

        for start in pbar:
            batch = train_indices[start:start+args.batch_size]

            # Load Batch
            batch_paths = [train_img_paths[i] for i in batch]
            batch_labels = [train_labels[i] for i in batch]
            
            img_t, valid_indices = load_batch_images(batch_paths, clip_preprocess, device)
            if img_t.shape[0] == 0:
                continue
            
            # Filter labels to match valid images
            batch_lbls = [batch_labels[i] for i in valid_indices]

            # Forward
            with torch.no_grad():
                toks = encode_image_to_tokens(clip_model, img_t)

            logits, acts, feats = classnet(toks[:, 1:])
            targets = build_targets(batch_lbls, classes_no_bg, device)

            # Loss
            l_cls = bce_loss(logits, targets)
            l_proto = proto_contrastive_loss(feats, acts, targets, class_proto_vecs, 
                                            margin=args.proto_margin)
            loss = l_cls + args.lambda_proto * l_proto

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["loss"] += loss.item()
            metrics["cls"] += l_cls.item()
            metrics["proto"] += l_proto.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # Validation
        print(f"\n[Epoch {epoch}] Train Loss: {metrics['loss']/n_batches:.4f} "
              f"(Cls: {metrics['cls']/n_batches:.3f} | Proto: {metrics['proto']/n_batches:.3f})")

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
                "Mean Acc {0:.2f}% | Mean Prec {1:.2f}% | Mean Rec {2:.2f}% | "
                "Mean F1 {3:.2f}%".format(
                    summary["mean_accuracy"] * 100,
                    summary["mean_precision"] * 100,
                    summary["mean_recall"] * 100,
                    summary["mean_f1"] * 100,
                )
            )
            print(
                "mIoU {0:.2f}% | FwIoU {1:.2f}% | "
                "Mean bIoU {2:.2f}% | Mean Dice {3:.2f}%".format(
                    summary["mIoU"] * 100,
                    summary["FwIoU"] * 100,
                    summary["mean_bIoU"] * 100,
                    summary["mean_dice"] * 100,
                )
            )
        print("-" * 40)
    
    # Save checkpoint
    output_path = args.output or os.path.join(args.data_root, "classnet_pp_stage2_1.pt")
    save_dict = {
        "state_dict": classnet.state_dict(),
        "config": {"input_dim": INPUT_DIM, "proto_dim": PROTO_DIM},
        "classes": classes_no_bg
    }
    torch.save(save_dict, output_path)
    print(f"[OK] Model saved to {output_path}")


if __name__ == "__main__":
    main()

