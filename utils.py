"""
Utility functions for data processing
"""
import os
import re
from typing import List


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

