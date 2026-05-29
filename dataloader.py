"""
PyTorch Dataset & DataLoader for Grasp-Anything + Grasp-Anything-pp.

Each sample contains:
  - image: RGB tensor [3, 416, 416]
  - instruction: text string (grasp instruction)
  - positive_label: grasp label tensor from .pt file

All matched by SHA-256 filename.

Usage:
    from dataloader import get_grasp_dataloader

    train_loader, val_loader, test_loader = get_grasp_dataloader(
        data_dir="./data",
        batch_size=16,
        val_split=0.1,
        num_workers=4,
    )

    for batch in train_loader:
        images = batch["image"]           # [B, 3, 416, 416]
        instructions = batch["instruction"]  # list of strings, len=B
        labels = batch["positive_label"]  # [B, ...] grasp labels
        ...
"""

import os
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class GraspAnythingDataset(Dataset):
    """
    Dataset linking Grasp-Anything images with Grasp-Anything-pp
    instructions and positive grasp labels, matched by SHA filename.
    """

    def __init__(
        self,
        data_dir: str,
        sha_list: Optional[list] = None,
        transform=None,
        load_images: bool = True,
    ):
        """
        Args:
            data_dir: Root data directory containing:
                - images/{sha}.jpg
                - grasp_instructions/{sha}.pkl
                - grasp_label_positive/{sha}.pt
            sha_list: List of SHA identifiers to use. If None, auto-detect
                      from matched_shas.txt or intersection of available files.
            transform: torchvision transforms for images.
            load_images: Whether to load images (set False if images not downloaded yet).
        """
        self.data_dir = Path(data_dir)
        self.load_images = load_images

        self.images_dir = self.data_dir / "images"
        self.instructions_dir = self.data_dir / "grasp_instructions"
        self.labels_dir = self.data_dir / "grasp_label_positive"

        if sha_list is not None:
            self.shas = sha_list
        else:
            matched_file = self.data_dir / "matched_shas.txt"
            if matched_file.exists():
                self.shas = matched_file.read_text().strip().split("\n")
            else:
                instr_shas = {f.stem for f in self.instructions_dir.glob("*.pkl")}
                label_shas = {f.stem for f in self.labels_dir.glob("*.pt")}
                common = instr_shas & label_shas
                if self.load_images and self.images_dir.exists():
                    image_shas = {f.stem for f in self.images_dir.glob("*.jpg")}
                    common = {s for s in common if self._get_image_sha(s) in image_shas}
                self.shas = sorted(common)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        print(f"GraspAnythingDataset: {len(self.shas)} samples loaded")

    @staticmethod
    def _get_image_sha(sample_id: str) -> str:
        """Strip _N_N suffix from sample ID to get image filename."""
        import re
        return re.sub(r'(_\d+)+$', '', sample_id)

    def __len__(self):
        return len(self.shas)

    def __getitem__(self, idx):
        sha = self.shas[idx]

        # Load instruction
        instr_path = self.instructions_dir / f"{sha}.pkl"
        with open(instr_path, "rb") as f:
            instr_data = pickle.load(f)

        # The pkl file may contain a dict or list of instructions.
        # Extract the text instruction(s).
        if isinstance(instr_data, dict):
            # Typical format: {object_name: {"instruction": ..., ...}}
            # Flatten all instructions into a single string or pick one
            instructions = []
            for obj_name, obj_data in instr_data.items():
                if isinstance(obj_data, dict):
                    if "instruction" in obj_data:
                        instructions.append(obj_data["instruction"])
                    elif "grasp_instruction" in obj_data:
                        instructions.append(obj_data["grasp_instruction"])
                    else:
                        # Try first string value
                        for v in obj_data.values():
                            if isinstance(v, str):
                                instructions.append(v)
                                break
                elif isinstance(obj_data, str):
                    instructions.append(obj_data)
                elif isinstance(obj_data, list):
                    for item in obj_data:
                        if isinstance(item, str):
                            instructions.append(item)
                        elif isinstance(item, dict) and "instruction" in item:
                            instructions.append(item["instruction"])
            if instructions:
                instruction = random.choice(instructions)
            else:
                instruction = str(instr_data)
        elif isinstance(instr_data, list):
            # List of instruction strings or dicts
            texts = []
            for item in instr_data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "instruction" in item:
                    texts.append(item["instruction"])
            instruction = random.choice(texts) if texts else str(instr_data)
        elif isinstance(instr_data, str):
            instruction = instr_data
        else:
            instruction = str(instr_data)

        # Load positive label
        label_path = self.labels_dir / f"{sha}.pt"
        positive_label = torch.load(label_path, map_location="cpu", weights_only=False)
        if isinstance(positive_label, dict):
            # If stored as dict, get the tensor value
            for k, v in positive_label.items():
                if isinstance(v, torch.Tensor):
                    positive_label = v
                    break
        if not isinstance(positive_label, torch.Tensor):
            positive_label = torch.tensor(positive_label, dtype=torch.float32)

        # Load image
        # Sample ID is like abc_1_1, image file is abc.jpg
        if self.load_images:
            image_sha = self._get_image_sha(sha)
            img_path = self.images_dir / f"{image_sha}.jpg"
            if not img_path.exists():
                img_path = self.images_dir / f"{image_sha}.png"
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.zeros(3, 416, 416)

        return {
            "image": image,
            "instruction": instruction,
            "positive_label": positive_label,
            "sha": sha,
        }


def grasp_collate_fn(batch):
    """Custom collate function to handle variable-size labels and string instructions."""
    images = torch.stack([item["image"] for item in batch])
    instructions = [item["instruction"] for item in batch]
    shas = [item["sha"] for item in batch]

    # Try to stack labels if they have the same shape
    labels = [item["positive_label"] for item in batch]
    try:
        labels = torch.stack(labels)
    except RuntimeError:
        pass  # keep as list if shapes differ

    return {
        "image": images,
        "instruction": instructions,
        "positive_label": labels,
        "sha": shas,
    }


def get_grasp_dataloader(
    data_dir: str = "./data",
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    load_images: bool = True,
    transform=None,
    seed: int = 42,
):
    """
    Create train, validation, and test DataLoaders for Grasp-Anything dataset.

    Split ratio (default): 70% train, 15% val, 15% test.

    Args:
        data_dir: Path to the processed data directory.
        batch_size: Batch size for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        num_workers: Number of dataloader workers.
        load_images: Whether to load images.
        transform: Custom transform for images.
        seed: Random seed for reproducible split.

    Returns:
        (train_loader, val_loader, test_loader) tuple of DataLoaders.
    """
    dataset = GraspAnythingDataset(
        data_dir=data_dir,
        transform=transform,
        load_images=load_images,
    )

    total = len(dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        import random as _random
        import numpy as _np
        _random.seed(worker_seed)
        _np.random.seed(worker_seed)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=grasp_collate_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=grasp_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=grasp_collate_fn,
        pin_memory=True,
    )

    print(f"Train: {train_size} samples, {len(train_loader)} batches")
    print(f"Val:   {val_size} samples, {len(val_loader)} batches")
    print(f"Test:  {test_size} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader, test_loader = get_grasp_dataloader(
        data_dir="./data",
        batch_size=8,
        num_workers=0,
        load_images=True,
    )

    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Instructions (first 2): {batch['instruction'][:2]}")
    print(f"  Label type: {type(batch['positive_label'])}")
    if isinstance(batch["positive_label"], torch.Tensor):
        print(f"  Label shape: {batch['positive_label'].shape}")