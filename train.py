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
    build_targets,
    load_batch_images
)


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


def validate(model, idx_list, paths, labels, classes, clip_model, clip_preprocess, device, batch_size=32):
    """Validate model"""
    model.eval()
    correct = torch.zeros(len(classes), device=device)
    total = 0

    with torch.no_grad():
        for start in range(0, len(idx_list), batch_size):
            batch = idx_list[start:start+batch_size]
            if not batch:
                continue

            batch_paths = [paths[i] for i in batch]
            batch_labels = [labels[i] for i in batch]
            
            img_t, valid_indices = load_batch_images(batch_paths, clip_preprocess, device)
            if img_t.shape[0] == 0:
                continue
            
            # Filter labels to match valid images
            batch_lbls = [batch_labels[i] for i in valid_indices]

            toks = encode_image_to_tokens(clip_model, img_t)
            logits, _, _ = model(toks[:, 1:])

            targs = build_targets(batch_lbls, classes, device)
            preds = (torch.sigmoid(logits) > 0.5).float()

            correct += (preds == targs).float().sum(0)
            total += preds.shape[0]

    if total == 0:
        return {}
    return {c: (correct[i]/total).item()*100 for i, c in enumerate(classes)}


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
        val_acc = validate(classnet, val_indices, train_img_paths, train_labels,
                          classes_no_bg, clip_model, clip_preprocess, device, args.batch_size)
        print("-" * 40)
        print(f"MEAN ACC: {sum(val_acc.values())/len(classes_no_bg):.2f}%")
        for c, acc in val_acc.items():
            print(f"  >> {c:25}: {acc:.2f}%")
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

