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

IGNORE_LABEL = -1


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


def estimate_bg_mask(
    pil_img: Image.Image,
    v_thr: int = 240,
    s_thr: int = 15,
    min_size: int = 32,
) -> np.ndarray:
    """
    Estimate background (glass) regions using HSV heuristics.
    Returns boolean mask where True indicates background.
    """
    hsv = pil_img.convert("HSV")
    _, s_channel, v_channel = hsv.split()
    s = np.array(s_channel, dtype=np.uint8)
    v = np.array(v_channel, dtype=np.uint8)

    bg_mask = (v > v_thr) & (s < s_thr)
    tissue_mask = ~bg_mask

    if tissue_mask.any():
        num_labels, labels = cv2.connectedComponents(tissue_mask.astype(np.uint8))
        for label in range(1, num_labels):
            region = labels == label
            if region.sum() < min_size:
                tissue_mask[region] = False
        bg_mask = ~tissue_mask

    return bg_mask.astype(bool)


def build_histograms(
    samples: List[Dict],
    num_classes: int,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    hist_pos = np.zeros((num_classes, bins), dtype=np.int64)
    hist_neg = np.zeros((num_classes, bins), dtype=np.int64)

    for sample in samples:
        score_map = sample["score_np"]
        gt = sample["gt_np"]
        tissue_mask = sample["tissue_np"]
        valid = (gt != IGNORE_LABEL) & tissue_mask
        if not np.any(valid):
            continue

        for cid in range(num_classes):
            s_vals = score_map[cid][valid]
            if s_vals.size == 0:
                continue
            g = (gt[valid] == cid)
            if not (g.any() or (~g).any()):
                continue
            bin_idx = np.clip((s_vals * (bins - 1)).astype(np.int64), 0, bins - 1)
            np.add.at(hist_pos[cid], bin_idx[g], 1)
            np.add.at(hist_neg[cid], bin_idx[~g], 1)

    return hist_pos, hist_neg


def search_best_thresholds(
    hist_pos: np.ndarray,
    hist_neg: np.ndarray,
    classes: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    num_classes, bins = hist_pos.shape
    best_thr = np.zeros(num_classes, dtype=np.float32)
    best_iou = np.zeros(num_classes, dtype=np.float32)

    for cid in range(num_classes):
        pos = hist_pos[cid]
        neg = hist_neg[cid]
        if pos.sum() == 0:
            best_thr[cid] = 0.5
            best_iou[cid] = 0.0
            print(
                f"[TH-SEARCH] {classes[cid]:<25} insufficient positives. Falling back to 0.5."
            )
            continue

        tp = np.cumsum(pos[::-1])[::-1]
        fp = np.cumsum(neg[::-1])[::-1]
        fn = pos.sum() - tp
        denom = tp + fp + fn
        iou = np.zeros_like(denom, dtype=np.float32)
        nz = denom > 0
        iou[nz] = tp[nz].astype(np.float32) / denom[nz].astype(np.float32)

        idx = int(iou.argmax())
        best_thr[cid] = idx / (bins - 1)
        best_iou[cid] = iou[idx]
        print(
            f"[TH-SEARCH] {classes[cid]:<25} Thr={best_thr[cid]:.3f} "
            f"IoU={best_iou[cid]*100:.2f}%"
        )

    return best_thr, best_iou


def infer_masks_from_scores(
    score_map: np.ndarray,
    thresholds: np.ndarray,
    seed_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    C, H, W = score_map.shape
    max_vals = score_map.max(axis=0)
    argmax = score_map.argmax(axis=0)

    pred_strict = np.full((H, W), IGNORE_LABEL, dtype=np.int16)
    pred_seed = np.full((H, W), IGNORE_LABEL, dtype=np.int16)

    for cid in range(C):
        thr = thresholds[cid]
        mask_strict = (argmax == cid) & (max_vals >= thr)
        mask_seed = (argmax == cid) & (max_vals >= (seed_scale * thr))
        pred_strict[mask_strict] = cid
        pred_seed[mask_seed] = cid

    return pred_strict, pred_seed


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
    parser.add_argument("--use_bg_refine", action="store_true",
                       help="Apply Stage 2.2 background gating and probability weighting")
    parser.add_argument("--use_bg_thresholds", action="store_true",
                       help="Enable Stage 2.2 per-class threshold search (requires masks)")
    parser.add_argument("--bg_bins", type=int, default=256,
                       help="Number of histogram bins for threshold search")
    parser.add_argument("--bg_seed_scale", type=float, default=0.5,
                       help="Seed threshold scale relative to strict threshold")
    parser.add_argument("--bg_v_thr", type=int, default=240,
                       help="HSV value threshold for background detection")
    parser.add_argument("--bg_s_thr", type=int, default=15,
                       help="HSV saturation threshold for background detection")
    parser.add_argument("--bg_min_size", type=int, default=32,
                       help="Minimum tissue component size to keep during BG filtering")
    parser.add_argument("--bg_pred_variant", type=str, default="strict",
                       choices=["strict", "seed"],
                       help="Which Stage 2.2 prediction variant to evaluate when thresholds are enabled")
    parser.add_argument("--single_image_path", type=str, default=None,
                       help="Optional single image path for standalone pseudo mask generation")
    parser.add_argument("--single_mask_path", type=str, default=None,
                       help="Optional ground-truth mask for the single image")
    parser.add_argument("--single_save_path", type=str, default=None,
                       help="Path to save the single-image visualization (default: data_root/single_pseudo.png)")
    parser.add_argument("--single_show", action="store_true",
                       help="Display the single-image visualization instead of just saving")
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
    if args.use_bg_thresholds and not args.use_bg_refine:
        raise ValueError("--use_bg_thresholds requires --use_bg_refine.")
    if args.use_bg_refine:
        print(
            "[INFO] Stage 2.2 background refinement enabled "
            f"(threshold search: {'ON' if args.use_bg_thresholds else 'OFF'})."
        )

    def compute_maps(img_pil: Image.Image) -> Optional[Dict[str, np.ndarray]]:
        refined_maps: List[torch.Tensor] = []
        score_maps: List[torch.Tensor] = []

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
                logits, act_maps, patch_feats_norm = classnet(toks[:, 1:])
                logits = logits.squeeze(0)
                probs = torch.sigmoid(logits)
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
                for ci in range(NUM_CLASSES):
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

                refined = torch.clamp(refined, min=0.0)
                score = torch.clamp(refined * probs.view(-1, 1, 1), min=0.0, max=1.0)
                refined_maps.append(refined)
                score_maps.append(score)

        if not refined_maps:
            return None

        refined_avg = torch.stack(refined_maps, dim=0).mean(dim=0)
        score_avg = torch.stack(score_maps, dim=0).mean(dim=0)

        bg_mask_arr = None
        tissue_arr = np.ones(
            (refined_avg.shape[-2], refined_avg.shape[-1]), dtype=bool
        )
        if args.use_bg_refine:
            bg_mask_bool = estimate_bg_mask(
                img_pil, v_thr=args.bg_v_thr, s_thr=args.bg_s_thr, min_size=args.bg_min_size
            )
            bg_mask_img = Image.fromarray(bg_mask_bool.astype(np.uint8) * 255)
            target_h, target_w = refined_avg.shape[-2], refined_avg.shape[-1]
            bg_mask_resized = (
                np.array(
                    bg_mask_img.resize((target_w, target_h), Image.NEAREST)
                )
                > 0
            )
            bg_mask_tensor = torch.from_numpy(bg_mask_resized).to(
                device=device, dtype=torch.bool
            )
            score_avg[:, bg_mask_tensor] = 0.0
            refined_avg[:, bg_mask_tensor] = 0.0
            bg_mask_arr = bg_mask_resized
            tissue_arr = ~bg_mask_resized

        effective_map = score_avg if args.use_bg_refine else refined_avg
        max_vals, pred_conf = effective_map.max(dim=0)
        pred_conf = torch.where(
            max_vals > args.conf_thr,
            pred_conf,
            torch.full_like(pred_conf, IGNORE_LABEL),
        )

        if args.use_bg_refine and bg_mask_arr is not None:
            mask_tensor = torch.from_numpy(bg_mask_arr).to(
                device=device, dtype=torch.bool
            )
            pred_conf = pred_conf.clone()
            pred_conf[mask_tensor] = IGNORE_LABEL

        return {
            "refined_np": refined_avg.detach().cpu().numpy().astype(np.float32),
            "score_np": score_avg.detach().cpu().numpy().astype(np.float32),
            "pred_conf_np": pred_conf.detach().cpu().numpy().astype(np.int16),
            "bg_np": bg_mask_arr,
            "tissue_np": tissue_arr,
        }

    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    print(f"Evaluating on {len(val_images)} test images...")

    desc_parts = ["CAA" if use_caa else "No CAA"]
    if args.use_bg_refine:
        desc_parts.append("BG refine")
        if args.use_bg_thresholds:
            desc_parts.append("Thr search")
    result_label = " + ".join(desc_parts)
    eval_desc = f"Evaluating Pseudo Masks ({result_label})"

    samples: List[Dict] = []
    for _, img_path in tqdm(list(enumerate(val_images)), total=len(val_images), desc=eval_desc):
        mask_path = os.path.join(val_mask_dir, os.path.basename(img_path))
        if not os.path.exists(mask_path):
            continue

        try:
            img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path)
        except Exception:
            continue

        mask_indices, present_classes = mask_to_class_indices(mask_pil, classes_no_bg)
        gt_mask = torch.from_numpy(mask_indices).long().to(device)
        gt_mask = F.interpolate(
            gt_mask[None, None].float(),
            size=(224, 224),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
        gt_np = gt_mask.cpu().numpy().astype(np.int16)

        gt_labels_np = np.zeros(NUM_CLASSES, dtype=np.float32)
        for cls_name in present_classes:
            if cls_name in classes_no_bg:
                gt_labels_np[classes_no_bg.index(cls_name)] = 1.0

        maps = compute_maps(img_pil)
        if maps is None:
            continue

        samples.append(
            {
                "name": os.path.basename(img_path),
                "img_rgb": np.array(img_pil.resize((224, 224))),
                "gt_np": gt_np,
                "gt_labels_np": gt_labels_np,
                "score_np": maps["score_np"],
                "pred_conf_np": maps["pred_conf_np"],
                "bg_np": maps["bg_np"],
                "tissue_np": maps["tissue_np"],
                "pred_seed_np": None,
                "pred_strict_np": None,
            }
        )

    if not samples:
        print("No samples available for evaluation.")
        return

    best_thr = None
    if args.use_bg_refine and args.use_bg_thresholds:
        print("\n[Stage 2.2] Running per-class threshold search...")
        hist_pos, hist_neg = build_histograms(samples, NUM_CLASSES, args.bg_bins)
        best_thr, best_iou = search_best_thresholds(hist_pos, hist_neg, classes_no_bg)

    for sample in samples:
        if best_thr is not None:
            pred_strict, pred_seed = infer_masks_from_scores(
                sample["score_np"], best_thr, args.bg_seed_scale
            )
            if sample["bg_np"] is not None:
                pred_strict[sample["bg_np"]] = IGNORE_LABEL
                pred_seed[sample["bg_np"]] = IGNORE_LABEL
            sample["pred_strict_np"] = pred_strict
            sample["pred_seed_np"] = pred_seed
            sample["pred_final_np"] = (
                pred_strict if args.bg_pred_variant == "strict" else pred_seed
            )
        else:
            sample["pred_final_np"] = sample["pred_conf_np"]

    hist = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
    tp_cls = torch.zeros(NUM_CLASSES, device=device)
    fp_cls = torch.zeros(NUM_CLASSES, device=device)
    fn_cls = torch.zeros(NUM_CLASSES, device=device)
    tn_cls = torch.zeros(NUM_CLASSES, device=device)
    pseudo_outputs = []

    for sample in samples:
        gt_tensor = torch.from_numpy(sample["gt_np"].astype(np.int64)).to(device)
        pred_tensor = torch.from_numpy(sample["pred_final_np"].astype(np.int64)).to(device)

        valid = (gt_tensor >= 0) & (gt_tensor < NUM_CLASSES) & (pred_tensor >= 0)
        if valid.sum() == 0:
            continue

        hist += torch.bincount(
            NUM_CLASSES * gt_tensor[valid] + pred_tensor[valid],
            minlength=NUM_CLASSES ** 2,
        ).view(NUM_CLASSES, NUM_CLASSES)

        pred_labels = torch.zeros(NUM_CLASSES, device=device)
        for ci in range(NUM_CLASSES):
            if (pred_tensor == ci).any():
                pred_labels[ci] = 1.0

        gt_labels_tensor = torch.from_numpy(sample["gt_labels_np"]).to(device)
        tp_cls += pred_labels * gt_labels_tensor
        fp_cls += pred_labels * (1.0 - gt_labels_tensor)
        fn_cls += (1.0 - pred_labels) * gt_labels_tensor
        tn_cls += (1.0 - pred_labels) * (1.0 - gt_labels_tensor)

        pseudo_outputs.append(
            {
                "img": sample["img_rgb"],
                "gt": sample["gt_np"],
                "pred": sample["pred_final_np"],
                "name": sample["name"],
                "pred_seed": sample["pred_seed_np"],
                "pred_strict": sample["pred_strict_np"],
            }
        )

    # Calculate mIoU
    ious = []
    print("\n" + "=" * 40)
    print(f"PSEUDO MASK EVALUATION RESULTS ({result_label})")
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

    visual_samples = pseudo_outputs[: min(args.num_visualize, len(pseudo_outputs))]
    if visual_samples:
        num_cols = 4 if best_thr is not None else 3
        fig, axes = plt.subplots(
            len(visual_samples),
            num_cols,
            figsize=(5 * num_cols, 5 * len(visual_samples)),
        )
        if len(visual_samples) == 1:
            axes = axes[None, :]

        for idx, sample in enumerate(visual_samples):
            axes[idx, 0].imshow(sample["img"])
            axes[idx, 0].set_title(f"Image: {sample['name']}")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(sample["gt"], cmap=cmap, norm=norm, interpolation="nearest")
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis("off")

            if best_thr is not None:
                seed_pred = sample["pred_seed"]
                strict_pred = sample["pred_strict"]
                if seed_pred is None:
                    seed_pred = sample["pred"]
                if strict_pred is None:
                    strict_pred = sample["pred"]

                axes[idx, 2].imshow(seed_pred, cmap=cmap, norm=norm, interpolation="nearest")
                axes[idx, 2].set_title("Seed Prediction")
                axes[idx, 2].axis("off")

                axes[idx, 3].imshow(strict_pred, cmap=cmap, norm=norm, interpolation="nearest")
                axes[idx, 3].set_title("Strict Prediction")
                axes[idx, 3].axis("off")
            else:
                axes[idx, 2].imshow(sample["pred"], cmap=cmap, norm=norm, interpolation="nearest")
                axes[idx, 2].set_title("Pseudo Mask Prediction")
                axes[idx, 2].axis("off")

        plt.tight_layout()
        tag = result_label.replace(" ", "").replace("+", "-")
        if best_thr is not None:
            save_name = f"eval_{tag}_seed{args.bg_seed_scale:.2f}.png"
        else:
            save_name = f"eval_{tag}_conf{args.conf_thr:.2f}.png"
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
                    except Exception as exc:
                        print(f"[WARNING] Failed to load mask {mask_path}: {exc}")
                else:
                    print(f"[WARNING] Single mask path not found: {mask_path}")

            if single_img_pil is not None:
                maps_single = compute_maps(single_img_pil)
                if maps_single is None:
                    print("[WARNING] Unable to generate pseudo mask for the provided image.")
                else:
                    score_single = maps_single["score_np"]
                    bg_single = maps_single["bg_np"]
                    if args.use_bg_refine and args.use_bg_thresholds and best_thr is not None:
                        single_strict, single_seed = infer_masks_from_scores(
                            score_single, best_thr, args.bg_seed_scale
                        )
                        if bg_single is not None:
                            single_strict[bg_single] = IGNORE_LABEL
                            single_seed[bg_single] = IGNORE_LABEL
                        single_final = (
                            single_strict if args.bg_pred_variant == "strict" else single_seed
                        )
                    else:
                        single_strict = None
                        single_seed = None
                        single_final = maps_single["pred_conf_np"]
                        if args.use_bg_refine and bg_single is not None:
                            single_final = single_final.copy()

                    fig_cols = 3 if gt_np is not None else 2
                    if args.use_bg_refine and args.use_bg_thresholds and best_thr is not None:
                        fig_cols = fig_cols + 1
                    fig, axes = plt.subplots(1, fig_cols, figsize=(5 * fig_cols, 5))
                    axes = np.atleast_1d(axes)

                    axes[0].imshow(single_img_pil.resize((224, 224)))
                    axes[0].set_title("Input Image")
                    axes[0].axis("off")

                    pred_col = 1
                    if gt_np is not None:
                        axes[1].imshow(gt_np, cmap=cmap, norm=norm, interpolation="nearest")
                        axes[1].set_title("Ground Truth Mask")
                        axes[1].axis("off")
                        pred_col = 2

                    axes[pred_col].imshow(single_final, cmap=cmap, norm=norm, interpolation="nearest")
                    axes[pred_col].set_title("Final Prediction")
                    axes[pred_col].axis("off")

                    if args.use_bg_refine and args.use_bg_thresholds and best_thr is not None:
                        other_pred = (
                            single_seed if args.bg_pred_variant == "strict" else single_strict
                        )
                        axes[pred_col + 1].imshow(
                            other_pred, cmap=cmap, norm=norm, interpolation="nearest"
                        )
                        label = "Seed Prediction" if args.bg_pred_variant == "strict" else "Strict Prediction"
                        axes[pred_col + 1].set_title(label)
                        axes[pred_col + 1].axis("off")

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

