"""
Script to evaluate ClassNet++ model with CAA refinement
"""
import os
import glob
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

import clip
from model.classnet import ClassNetPP
from model.clip_encoder import encode_image_to_tokens
from utils import mask_to_class_indices, compute_presence_metrics


def sinkhorn_norm(A: torch.Tensor, num_iters: int = 5):
    """
    Perform Sinkhorn Normalization: D = Sinkhorn(A).
    """
    D = F.relu(A) + 1e-7
    
    for _ in range(num_iters):
        D = D / D.sum(dim=1, keepdim=True)
        D = D / D.sum(dim=0, keepdim=True)
        
    return D


def get_box_mask(cam: torch.Tensor, lambda_thr: float):
    """
    Create bounding box mask Bc from normalized CAM.
    """
    H, W = cam.shape
    
    cam_np = cam.cpu().numpy()
    binary_mask = (cam_np > lambda_thr).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return torch.zeros(1, H * W, dtype=torch.bool, device=cam.device)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_box_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        final_box_mask[y:y+h, x:x+w] = 1
        
    Bc = torch.from_numpy(final_box_mask).flatten().unsqueeze(0).to(cam.device).float()
    
    return Bc


def caa_refinement(cam_norm: torch.Tensor, patch_feats_norm: torch.Tensor,
                   t: int = 3, lambda_thr: float = 0.20):
    """
    Perform Class-aware Attention-based Affinity (CAA) refinement.
    """
    H, W = cam_norm.shape
    N = H * W
    
    # Compute similarity matrix
    Wattn = patch_feats_norm @ patch_feats_norm.t()
    D = sinkhorn_norm(Wattn, num_iters=5)
    A = (D + D.t()) / 2
    
    # A^t
    A_t = torch.matrix_power(A, t)
    
    # Bc (Box Mask)
    Bc = get_box_mask(cam_norm, lambda_thr=lambda_thr)
    
    # Vectorize CAM
    vec_Mc = cam_norm.flatten().unsqueeze(1)
    
    # Refine CAM
    A_t_masked = Bc.t() * A_t
    vec_Mc_aff = A_t_masked @ vec_Mc
    cam_refined = vec_Mc_aff.view(H, W)
    
    # Normalize
    if cam_refined.max() > 1e-7:
        cam_refined = (cam_refined - cam_refined.min()) / (cam_refined.max() + 1e-7)
    
    return cam_refined


def main():
    parser = argparse.ArgumentParser(description="Evaluate ClassNet++")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory for data")
    parser.add_argument("--val_img_dir", type=str, default=None,
                       help="Validation image directory (default: data_root/test/img)")
    parser.add_argument("--val_mask_dir", type=str, default=None,
                       help="Validation mask directory (default: data_root/test/mask)")
    parser.add_argument("--hybrid_proto_path", type=str, default=None,
                       help="Hybrid prototype path (default: data_root/hybrid_prototypes.pt)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Model checkpoint path (default: data_root/classnet_pp_stage2_1.pt)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.8, 1.2],
                       help="Multi-scale evaluation scales")
    parser.add_argument("--caa_iters", type=int, default=5,
                       help="CAA iteration count")
    parser.add_argument("--caa_attn_power", type=int, default=3,
                       help="CAA attention power")
    parser.add_argument("--caa_thr", type=float, default=0.20,
                       help="CAA threshold lambda")
    parser.add_argument("--conf_thr", type=float, default=0.05,
                       help="Confidence threshold for pseudo mask")
    parser.add_argument("--num_visualize", type=int, default=3,
                       help="Number of samples to visualize")
    parser.add_argument("--save_pseudo_dir", type=str, default=None,
                       help="Directory to save pseudo masks (optional)")
    parser.add_argument("--no_caa", action="store_true",
                       help="Skip CAA refinement and use raw CAMs")
    parser.add_argument("--single_image_path", type=str, default=None,
                       help="Optional single image path for standalone pseudo mask generation")
    parser.add_argument("--single_mask_path", type=str, default=None,
                       help="Optional ground-truth mask for the single image")
    parser.add_argument("--single_save_path", type=str, default=None,
                       help="Path to save the single-image visualization (default: data_root/single_pseudo.png)")
    parser.add_argument("--single_show", action="store_true",
                       help="Display the single-image visualization instead of just saving")
    parser.add_argument("--no_caa", action="store_true",
                    help="Skip CAA refinement and use raw CAMs")

    # Trong phần xử lý CAM (khoảng dòng 282-285), thay bằng:
    if args.no_caa:
        refined_cam_tok = cam_norm_tok  # Dùng CAM gốc, không refine
    else:
        refined_cam_tok = caa_refinement(
            cam_norm_tok, token_feats, 
            t=args.caa_attn_power, lambda_thr=args.caa_thr
        )
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    classes_no_bg = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis"]
    NUM_CLASSES = len(classes_no_bg)
    
    val_img_dir = args.val_img_dir or os.path.join(args.data_root, "test", "img")
    val_mask_dir = args.val_mask_dir or os.path.join(args.data_root, "test", "mask")
    hybrid_proto_path = args.hybrid_proto_path or os.path.join(args.data_root, "hybrid_prototypes.pt")
    checkpoint_path = args.checkpoint_path or os.path.join(args.data_root, "classnet_pp_stage2_1.pt")
    
    # Load CLIP
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval().to(device)
    clip_model.float()  # Convert to float32 to match input tensor type
    
    # Load model
    missing_files = []
    if not os.path.exists(hybrid_proto_path):
        missing_files.append(f"Hybrid prototypes: {hybrid_proto_path}")
    if not os.path.exists(checkpoint_path):
        missing_files.append(f"Checkpoint: {checkpoint_path}")
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files:\n  " + "\n  ".join(missing_files) +
            f"\n\nPlease ensure you have:\n"
            f"  1. Run build_hybrid_prototypes.py to generate hybrid_prototypes.pt\n"
            f"  2. Run train.py to generate the model checkpoint"
        )
    
    hybrid_bank = torch.load(hybrid_proto_path, map_location="cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Auto-detect dimensions
    INPUT_DIM = int(ckpt.get("config", {}).get("input_dim", 768))
    PROTO_DIM = int(ckpt.get("config", {}).get("proto_dim", None))
    
    if PROTO_DIM is None:
        # Auto-detect from hybrid bank
        if 'feat_dim' in hybrid_bank:
            PROTO_DIM = int(hybrid_bank['feat_dim'])
        else:
            first_class = list(hybrid_bank['hybrid_prototypes'].keys())[0]
            H = hybrid_bank['hybrid_prototypes'][first_class].float()
            if H.dim() == 3:
                PROTO_DIM = H.shape[2]
            elif H.dim() == 2:
                PROTO_DIM = H.shape[1]
            else:
                raise ValueError(f"Cannot detect PROTO_DIM from shape {H.shape}")
    
    print(f"Using INPUT_DIM={INPUT_DIM}, PROTO_DIM={PROTO_DIM}")
    
    noise_tensor = ckpt["state_dict"].get("prograd.noise_protos")
    if noise_tensor is not None:
        noise_tensor = noise_tensor.to(device)
    classnet = ClassNetPP(
        hybrid_bank["hybrid_prototypes"],
        classes_no_bg,
        INPUT_DIM,
        PROTO_DIM,
        noise_protos=noise_tensor,
    ).to(device)
    classnet.load_state_dict(ckpt["state_dict"])
    classnet.eval()
    print("Model loaded for evaluation.")
    
    use_caa = not args.no_caa
    if not use_caa:
        print("[INFO] CAA refinement disabled. Using raw CAMs for pseudo masks.")
    
    def infer_pseudo_mask(
        img_pil: Image.Image,
        class_filter: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pseudo mask and averaged CAM tensor for a single image.
        Returns (pred_mask, P_resized). Both tensors are on the evaluation device.
        """
        class_filter_set = set(class_filter) if class_filter else None
        refined_cams_per_scale: List[torch.Tensor] = []

        with torch.no_grad():
            for scale in args.scales:
                if scale != 1.0:
                    new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
                    img_in = img_pil.resize(new_size, Image.BICUBIC)
                else:
                    img_in = img_pil

                pp = clip_preprocess(img_in)
                inp = pp.unsqueeze(0).to(device)
                target_size = pp.shape[-2:]

                toks = encode_image_to_tokens(clip_model, inp)
                _, act_maps, patch_feats_norm = classnet(toks[:, 1:])
                act_maps = act_maps.squeeze(0)

                token_feats = patch_feats_norm.squeeze(0)
                H_tok = W_tok = int(token_feats.shape[0] ** 0.5)

                cams_hi = F.interpolate(
                    act_maps.unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                refined = torch.zeros_like(cams_hi)

                for ci, c in enumerate(classes_no_bg):
                    if class_filter_set is not None and c not in class_filter_set:
                        continue

                    cam = cams_hi[ci]
                    cam_norm = (cam - cam.min()) / (cam.max() + 1e-7)

                    cam_norm_tok = F.interpolate(
                        cam_norm.unsqueeze(0).unsqueeze(0),
                        size=(H_tok, W_tok),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

                    if use_caa:
                        refined_cam_tok = caa_refinement(
                            cam_norm_tok,
                            token_feats,
                            t=args.caa_attn_power,
                            lambda_thr=args.caa_thr,
                        )
                    else:
                        refined_cam_tok = cam_norm_tok

                    refined_cam = F.interpolate(
                        refined_cam_tok.unsqueeze(0).unsqueeze(0),
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

                    refined[ci] = refined_cam

                refined_cams_per_scale.append(refined)

        if not refined_cams_per_scale:
            return None, None

        cams_stack = torch.stack(refined_cams_per_scale, dim=0)
        P = cams_stack.mean(dim=0)

        if P.shape[1:] != (224, 224):
            P_resized = F.interpolate(
                P.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            P_resized = P

        max_vals, pred_mask = P_resized.max(dim=0)
        ignore_label = -1
        pred_mask = torch.where(
            max_vals > args.conf_thr,
            pred_mask,
            torch.full_like(pred_mask, ignore_label),
        )

        return pred_mask, P_resized
    
    # Evaluation
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    print(f"Evaluating on {len(val_images)} test images...")
    
    hist = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
    tp_cls = torch.zeros(NUM_CLASSES, device=device)
    fp_cls = torch.zeros(NUM_CLASSES, device=device)
    fn_cls = torch.zeros(NUM_CLASSES, device=device)
    tn_cls = torch.zeros(NUM_CLASSES, device=device)
    visual_samples = []
    pseudo_outputs = []
    
    eval_desc = "Evaluating Pseudo Masks (CAA)" if use_caa else "Evaluating Pseudo Masks (No CAA)"
    for i, img_path in tqdm(list(enumerate(val_images)), total=len(val_images),
                             desc=eval_desc):
        mask_path = os.path.join(val_mask_dir, os.path.basename(img_path))
        if not os.path.exists(mask_path):
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path)

            mask_indices, present_classes = mask_to_class_indices(mask_pil, classes_no_bg)
            gt_mask = torch.from_numpy(mask_indices).long().to(device)
            gt_mask = F.interpolate(
                gt_mask[None, None].float(),
                size=(224, 224),
                mode="nearest"
            ).squeeze(0).squeeze(0).long()

            gt_labels = torch.zeros(NUM_CLASSES, device=device)
            for cls in present_classes:
                gt_labels[classes_no_bg.index(cls)] = 1.0
        except Exception:
            continue

        unique_classes = torch.unique(gt_mask)
        lbls = [classes_no_bg[c] for c in unique_classes
                if 0 <= int(c) < NUM_CLASSES]
        if not lbls:
            continue

        pred_mask, P_resized = infer_pseudo_mask(img_pil, class_filter=lbls)
        if pred_mask is None:
            continue

        valid = (gt_mask >= 0) & (gt_mask < NUM_CLASSES) & (pred_mask >= 0)
        if valid.sum() == 0:
            continue

        hist += torch.bincount(
            NUM_CLASSES * gt_mask[valid] + pred_mask[valid],
            minlength=NUM_CLASSES ** 2
        ).view(NUM_CLASSES, NUM_CLASSES)

        pred_labels = torch.zeros(NUM_CLASSES, device=device)
        for ci in range(NUM_CLASSES):
            if (pred_mask == ci).any():
                pred_labels[ci] = 1.0

        tp_cls += (pred_labels * gt_labels)
        fp_cls += (pred_labels * (1.0 - gt_labels))
        fn_cls += ((1.0 - pred_labels) * gt_labels)
        tn_cls += ((1.0 - pred_labels) * (1.0 - gt_labels))

        sample = {
            "img": np.array(img_pil.resize((224, 224))),
            "pred": pred_mask.cpu().numpy(),
            "gt": gt_mask.cpu().numpy(),
            "name": os.path.basename(img_path),
        }
        if len(visual_samples) < args.num_visualize:
            visual_samples.append(sample)
        pseudo_outputs.append(sample)

    # Calculate mIoU
    ious = []
    print("\n" + "=" * 40)
    print(f"PSEUDO MASK EVALUATION RESULTS ({'CAA' if use_caa else 'No CAA'})")
    print("=" * 40)

    for i, c in enumerate(classes_no_bg):
        cls_tp = hist[i, i]
        union = hist[i, :].sum() + hist[:, i].sum() - cls_tp
        iou = (cls_tp / (union + 1e-7)).item()
        ious.append(iou)
        print(f"Class {c:<25}: {iou * 100:.2f}%")

    miou = sum(ious) / len(ious)
    print("-" * 40)
    print(f"MEAN IOU (mIoU)                       : {miou * 100:.2f}%")
    print("=" * 40)

    presence_metrics = compute_presence_metrics(tp_cls, fp_cls, fn_cls, tn_cls, classes_no_bg)
    summary = presence_metrics["summary"]
    print("CLASS PRESENCE METRICS")
    print("-" * 40)
    for cls, cls_metrics in presence_metrics["per_class"].items():
        print(
            f"{cls:25}: "
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
    print("=" * 40)

    if args.save_pseudo_dir:
        os.makedirs(args.save_pseudo_dir, exist_ok=True)
        for sample in pseudo_outputs:
            out_path = os.path.join(args.save_pseudo_dir, sample["name"])
            mask = Image.fromarray(sample["pred"].astype(np.uint8))
            mask.save(out_path)
        print(f"[INFO] Saved {len(pseudo_outputs)} pseudo masks to {args.save_pseudo_dir}")

    cmap = mcolors.ListedColormap(['red', 'green', 'blue', 'yellow'])
    norm = mcolors.BoundaryNorm(np.arange(NUM_CLASSES + 1) - 0.5, NUM_CLASSES)

    # Visualize
    if visual_samples:
        fig, axes = plt.subplots(len(visual_samples), 3,
                                 figsize=(15, 5 * len(visual_samples)))
        if len(visual_samples) == 1:
            axes = axes[None, :]

        for idx, sample in enumerate(visual_samples):
            axes[idx, 0].imshow(sample["img"])
            axes[idx, 0].set_title(f"Image: {sample['name']}")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(sample["gt"], cmap=cmap, norm=norm,
                                interpolation="nearest")
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(sample["pred"], cmap=cmap, norm=norm,
                                interpolation="nearest")
            axes[idx, 2].set_title("Pseudo Mask Prediction")
            axes[idx, 2].axis("off")

        plt.tight_layout()
        if use_caa:
            save_name = f"eval_caaP{args.caa_attn_power}_thr{args.caa_thr}_conf{args.conf_thr}.png"
        else:
            save_name = f"eval_noCAA_conf{args.conf_thr}.png"
        plt.savefig(os.path.join(args.data_root, save_name))
        print(f"\nVisualization saved to {os.path.join(args.data_root, save_name)}")

    if args.single_image_path:
        single_path = args.single_image_path
        print(f"\n[INFO] Generating pseudo mask for single image: {single_path}")

        if not os.path.exists(single_path):
            print(f"[WARNING] Single image path not found: {single_path}")
        else:
            try:
                single_img_pil = Image.open(single_path).convert("RGB")
            except Exception as exc:
                print(f"[WARNING] Failed to open image {single_path}: {exc}")
                single_img_pil = None

            gt_np = None
            class_filter_single: Optional[List[str]] = None

            if single_img_pil is not None and args.single_mask_path:
                mask_path = args.single_mask_path
                if os.path.exists(mask_path):
                    try:
                        mask_pil = Image.open(mask_path)
                        mask_indices, present_classes = mask_to_class_indices(mask_pil, classes_no_bg)
                        gt_mask = torch.from_numpy(mask_indices).long().to(device)
                        gt_mask = F.interpolate(
                            gt_mask[None, None].float(),
                            size=(224, 224),
                            mode="nearest",
                        ).squeeze(0).squeeze(0).long()
                        gt_np = gt_mask.cpu().numpy()
                        class_filter_single = present_classes
                    except Exception as exc:
                        print(f"[WARNING] Failed to load mask {mask_path}: {exc}")
                else:
                    print(f"[WARNING] Single mask path not found: {mask_path}")

            if single_img_pil is not None:
                pred_mask_single, _ = infer_pseudo_mask(single_img_pil, class_filter=class_filter_single)
                if pred_mask_single is None:
                    print("[WARNING] Unable to generate pseudo mask for the provided image.")
                else:
                    pred_np = pred_mask_single.cpu().numpy()
                    fig_cols = 3 if gt_np is not None else 2
                    fig, axes = plt.subplots(1, fig_cols, figsize=(5 * fig_cols, 5))

                    axes = np.atleast_1d(axes)
                    axes[0].imshow(single_img_pil.resize((224, 224)))
                    axes[0].set_title("Input Image")
                    axes[0].axis("off")

                    if gt_np is not None:
                        axes[1].imshow(gt_np, cmap=cmap, norm=norm, interpolation="nearest")
                        axes[1].set_title("Ground Truth Mask")
                        axes[1].axis("off")
                        pred_axis = axes[2]
                    else:
                        pred_axis = axes[1]

                    pred_axis.imshow(pred_np, cmap=cmap, norm=norm, interpolation="nearest")
                    pred_axis.set_title("Pseudo Mask Prediction")
                    pred_axis.axis("off")

                    plt.tight_layout()

                    default_name = os.path.splitext(os.path.basename(single_path))[0]
                    if args.single_save_path:
                        single_save_path = args.single_save_path
                        save_dir = os.path.dirname(single_save_path)
                        if save_dir:
                            os.makedirs(save_dir, exist_ok=True)
                    else:
                        single_save_path = os.path.join(
                            args.data_root, f"single_pseudo_{default_name}.png"
                        )
                        os.makedirs(os.path.dirname(single_save_path), exist_ok=True)

                    plt.savefig(single_save_path)
                    print(f"[INFO] Single-image visualization saved to {single_save_path}")

                    if args.single_show:
                        plt.show()
                    else:
                        plt.close(fig)


if __name__ == "__main__":
    main()

