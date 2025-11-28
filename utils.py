"""
Utility functions for data processing
"""
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Label parsing helpers
# -----------------------------------------------------------------------------

def parse_labels_from_filename(fname: str) -> List[str]:
    """
    Parse labels from filename.
    
    Supports 2 patterns:
    - ...[abcd].png  -> uses letter2class
    - ...[0/1...].png with length 4 -> uses pos2class:
        index 0 -> Tumor
        index 1 -> Stroma
        index 2 -> Lymphocytic infiltrate
        index 3 -> Necrosis
    """
    letter2class = {
        "a": "Tumor",
        "b": "Stroma",
        "c": "Lymphocytic infiltrate",
        "d": "Necrosis",
    }
    
    pos2class = [
        "Tumor",
        "Stroma",
        "Lymphocytic infiltrate",
        "Necrosis",
    ]
    
    base = os.path.basename(fname)
    m = re.search(r"\[([0-9a-zA-Z]+)\]\.png$", base)
    if m is None:
        return []

    token = m.group(1)

    # Case 1: all characters are a-d
    if all(ch in letter2class for ch in token):
        cls_list = [letter2class[ch] for ch in token]
        return list(dict.fromkeys(cls_list))  # remove dup, keep order

    # Case 2: token is 0/1 string of length 4
    if len(token) == 4 and all(ch in "01" for ch in token):
        cls_list = []
        for i, ch in enumerate(token):
            if ch == "1" and i < len(pos2class):
                cls_list.append(pos2class[i])
        return cls_list

    # If not matching either case, return empty
    return []


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def list_training_images(train_dir: str) -> List[str]:
    """
    List all training images in directory.
    
    Args:
        train_dir: Directory containing training images
        
    Returns:
        List of image file paths
    """
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    img_paths = []
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith(exts):
                img_paths.append(os.path.join(root, f))
    img_paths.sort()
    return img_paths


# -----------------------------------------------------------------------------
# Mask utilities
# -----------------------------------------------------------------------------

CLASS_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "Tumor": (255, 0, 0),                  # Red
    "Stroma": (0, 255, 0),                 # Green
    "Lymphocytic infiltrate": (0, 0, 255), # Blue
    "Necrosis": (153, 0, 255),             # Purple
}
BACKGROUND_COLOR: Tuple[int, int, int] = (255, 255, 255)


def mask_to_class_indices(mask_pil, classes: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a color mask to class indices and list of present classes.

    Args:
        mask_pil: PIL Image instance for the segmentation mask.
        classes: List of class names (order defines index assignment).

    Returns:
        Tuple of (index_mask, present_classes).
        index_mask is an array of shape [H, W] with values in [-1, num_classes-1],
        where -1 marks pixels to ignore (e.g., background).
        present_classes is a sorted list of class names detected in the mask.
    """
    mask_rgb = np.array(mask_pil.convert("RGB"), dtype=np.uint8)
    h, w, _ = mask_rgb.shape

    index_mask = np.full((h, w), fill_value=-1, dtype=np.int64)
    present = set()

    for cls_name in classes:
        color = CLASS_COLOR_MAP.get(cls_name)
        if color is None:
            continue
        matches = np.all(mask_rgb == color, axis=-1)
        if np.any(matches):
            cls_idx = classes.index(cls_name)
            index_mask[matches] = cls_idx
            present.add(cls_name)

    # Everything not mapped so far is treated as background / ignore.
    return index_mask, sorted(present)


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------

def compute_presence_metrics(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    tn: torch.Tensor,
    classes: List[str],
) -> Dict[str, Dict]:
    """
    Compute per-class and aggregate metrics for multi-label presence prediction.

    Args:
        tp, fp, fn, tn: Tensors with shape [num_classes].
        classes: List of class names corresponding to tensor indices.

    Returns:
        Dictionary with keys:
            - "per_class": mapping class -> metrics dict
            - "summary": aggregate metrics (mean accuracy, mIoU, etc.)
    """
    eps = 1e-7
    tp = tp.float()
    fp = fp.float()
    fn = fn.float()
    tn = tn.float()

    per_class = {}
    acc_vals, prec_vals, rec_vals, f1_vals = [], [], [], []
    iou_vals, dice_vals, biou_vals = [], [], []

    pos_freq = tp + fn
    total_pos = pos_freq.sum().item()
    fw_iou_num = 0.0

    for i, cls in enumerate(classes):
        cls_tp = tp[i]
        cls_fp = fp[i]
        cls_fn = fn[i]
        cls_tn = tn[i]
        total = cls_tp + cls_fp + cls_fn + cls_tn

        acc = (cls_tp + cls_tn) / (total + eps)
        prec = cls_tp / (cls_tp + cls_fp + eps)
        rec = cls_tp / (cls_tp + cls_fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        iou = cls_tp / (cls_tp + cls_fp + cls_fn + eps)
        dice = 2 * cls_tp / (2 * cls_tp + cls_fp + cls_fn + eps)
        iou_neg = cls_tn / (cls_tn + cls_fp + cls_fn + eps)
        biou = 0.5 * (iou + iou_neg)

        if pos_freq[i] > 0:
            fw_iou_num += pos_freq[i].item() * iou.item()

        per_class[cls] = {
            "accuracy": acc.item(),
            "precision": prec.item(),
            "recall": rec.item(),
            "f1": f1.item(),
            "iou": iou.item(),
            "dice": dice.item(),
            "biou": biou.item(),
            "support": pos_freq[i].item(),
        }

        acc_vals.append(acc.item())
        prec_vals.append(prec.item())
        rec_vals.append(rec.item())
        f1_vals.append(f1.item())
        iou_vals.append(iou.item())
        dice_vals.append(dice.item())
        biou_vals.append(biou.item())

    num_classes = len(classes)
    summary = {
        "mean_accuracy": float(np.mean(acc_vals)) if acc_vals else 0.0,
        "mean_precision": float(np.mean(prec_vals)) if prec_vals else 0.0,
        "mean_recall": float(np.mean(rec_vals)) if rec_vals else 0.0,
        "mean_f1": float(np.mean(f1_vals)) if f1_vals else 0.0,
        "mIoU": float(np.mean(iou_vals)) if iou_vals else 0.0,
        "mean_dice": float(np.mean(dice_vals)) if dice_vals else 0.0,
        "mean_bIoU": float(np.mean(biou_vals)) if biou_vals else 0.0,
        "FwIoU": fw_iou_num / (total_pos + eps) if total_pos > 0 else 0.0,
        "num_classes": num_classes,
    }

    return {
        "per_class": per_class,
        "summary": summary,
    }

