"""
Data loading utilities for ClassNet++ training and evaluation
"""
import os
import glob
import random
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F

from utils import (
    parse_labels_from_filename,
    list_training_images,
    mask_to_class_indices,
)


class TrainingDataset(Dataset):
    """
    Dataset for training ClassNet++ model.
    Loads images and their labels from filenames.
    """
    def __init__(
        self,
        img_paths: List[str],
        labels: List[List[str]],
        clip_preprocess,
        classes: List[str],
        filter_valid: bool = True
    ):
        """
        Args:
            img_paths: List of image file paths
            labels: List of label lists (one per image)
            clip_preprocess: CLIP preprocessing function
            classes: List of valid class names
            filter_valid: If True, only include images with valid labels
        """
        self.clip_preprocess = clip_preprocess
        self.classes = classes
        
        if filter_valid:
            # Filter to only include images with valid labels
            valid_indices = [
                i for i, lbls in enumerate(labels)
                if any(c in classes for c in lbls)
            ]
            self.img_paths = [img_paths[i] for i in valid_indices]
            self.labels = [labels[i] for i in valid_indices]
        else:
            self.img_paths = img_paths
            self.labels = labels
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_list = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.clip_preprocess(img)
        except Exception as e:
            # Return a black image if loading fails
            img_tensor = torch.zeros(3, 224, 224)
            label_list = []
        
        return img_tensor, label_list, idx


class EvaluationDataset(Dataset):
    """
    Dataset for evaluation with ground truth masks.
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        clip_preprocess,
        classes: List[str],
        device: str = "cpu"
    ):
        """
        Args:
            img_dir: Directory containing validation/test images
            mask_dir: Directory containing ground truth masks
            clip_preprocess: CLIP preprocessing function
            classes: List of class names
            device: Device to load masks on
        """
        self.mask_dir = mask_dir
        self.clip_preprocess = clip_preprocess
        self.classes = classes
        self.device = device
        
        # List all images
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        
        # Filter to only include images with corresponding masks
        self.valid_paths = []
        for img_path in self.img_paths:
            mask_path = os.path.join(mask_dir, os.path.basename(img_path))
            if os.path.exists(mask_path):
                self.valid_paths.append(img_path)
    
    def __len__(self):
        return len(self.valid_paths)
    
    def __getitem__(self, idx):
        img_path = self.valid_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
        
        try:
            img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path)
            
            # Preprocess image
            img_tensor = self.clip_preprocess(img_pil)
            
            # Convert mask to class indices and compute presence vector
            mask_indices, present_classes = mask_to_class_indices(mask_pil, self.classes)
            gt_mask = torch.from_numpy(mask_indices).long().to(self.device)
            gt_mask = F.interpolate(
                gt_mask[None, None].float(),
                size=(224, 224),
                mode="nearest"
            ).squeeze(0).squeeze(0).long()

            label_vec = torch.zeros(len(self.classes), dtype=torch.float32, device=self.device)
            for cls in present_classes:
                label_vec[self.classes.index(cls)] = 1.0
            
            return img_tensor, gt_mask, label_vec, img_path
        except Exception as e:
            # Return dummy data if loading fails
            img_tensor = torch.zeros(3, 224, 224)
            gt_mask = torch.full((224, 224), fill_value=-1, dtype=torch.long, device=self.device)
            label_vec = torch.zeros(len(self.classes), dtype=torch.float32, device=self.device)
            return img_tensor, gt_mask, label_vec, img_path


def build_targets(lbls: List[List[str]], classes: List[str], device: str) -> torch.Tensor:
    """
    Build multi-label targets from label lists.
    
    Args:
        lbls: List of label lists (one per sample)
        classes: List of class names
        device: Device to create tensor on
        
    Returns:
        Tensor of shape [batch_size, num_classes] with 1.0 for present classes
    """
    t = torch.zeros(len(lbls), len(classes), device=device)
    c2i = {c: i for i, c in enumerate(classes)}
    
    for i, l in enumerate(lbls):
        for c in l:
            if c in c2i:
                t[i, c2i[c]] = 1.0
    
    return t


def load_training_data(
    train_dir: str,
    classes: List[str],
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[List[str]], List[int], List[int]]:
    """
    Load training data and split into train/validation sets.
    
    Args:
        train_dir: Directory containing training images
        classes: List of valid class names
        val_split: Fraction of data to use for validation
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (img_paths, labels, train_indices, val_indices)
    """
    random.seed(seed)
    
    # Load all training images and labels
    train_img_paths = list_training_images(train_dir)
    train_labels = [parse_labels_from_filename(p) for p in train_img_paths]
    
    # Filter valid indices
    all_valid_indices = [
        i for i, l in enumerate(train_labels)
        if any(c in classes for c in l)
    ]
    random.shuffle(all_valid_indices)
    
    # Split
    n_val = int(len(all_valid_indices) * val_split)
    val_indices = all_valid_indices[:n_val]
    train_indices = all_valid_indices[n_val:]
    
    return train_img_paths, train_labels, train_indices, val_indices


def create_training_dataloader(
    img_paths: List[str],
    labels: List[List[str]],
    indices: List[int],
    clip_preprocess,
    classes: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        img_paths: List of all image paths
        labels: List of all labels
        indices: Indices to use for this dataloader
        clip_preprocess: CLIP preprocessing function
        classes: List of class names
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    # Filter to only selected indices
    subset_paths = [img_paths[i] for i in indices]
    subset_labels = [labels[i] for i in indices]
    
    dataset = TrainingDataset(
        subset_paths,
        subset_labels,
        clip_preprocess,
        classes,
        filter_valid=False  # Already filtered
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


def create_evaluation_dataloader(
    img_dir: str,
    mask_dir: str,
    clip_preprocess,
    classes: List[str],
    batch_size: int = 1,
    device: str = "cpu",
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for evaluation.
    
    Args:
        img_dir: Directory containing images
        mask_dir: Directory containing masks
        clip_preprocess: CLIP preprocessing function
        classes: List of class names
        batch_size: Batch size (usually 1 for evaluation)
        device: Device to load masks on
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = EvaluationDataset(
        img_dir,
        mask_dir,
        clip_preprocess,
        classes,
        device
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


def load_batch_images(
    paths: List[str],
    clip_preprocess,
    device: str = "cpu"
) -> Tuple[torch.Tensor, List[int]]:
    """
    Load a batch of images from paths.
    
    Args:
        paths: List of image paths
        clip_preprocess: CLIP preprocessing function
        device: Device to load images on
        
    Returns:
        Tuple of (image_tensor, valid_indices)
        valid_indices are indices of successfully loaded images
    """
    imgs = []
    valid_indices = []
    
    for i, path in enumerate(paths):
        try:
            img = clip_preprocess(Image.open(path).convert("RGB"))
            imgs.append(img)
            valid_indices.append(i)
        except Exception:
            continue
    
    if not imgs:
        return torch.empty(0, 3, 224, 224, device=device), []
    
    img_tensor = torch.stack(imgs).to(device)
    return img_tensor, valid_indices

