"""
Visualize predictions vs ground truth in a batch grid.

Usage:
    # Visualize on val set
    python visualize.py --checkpoint checkpoints/best.pt --split val --num_samples 16

    # Visualize on test set
    python visualize.py --checkpoint checkpoints/best.pt --split test --num_samples 32

    # Visualize a specific image folder
    python visualize.py --checkpoint checkpoints/best.pt --image_dir ./my_images \
                        --instruction "grasp the cup" --num_samples 8

    # Change grid layout
    python visualize.py --checkpoint checkpoints/best.pt --split val --cols 4 --num_samples 16
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor
import yaml

from models.grasp_model import GraspCLIPModel
from dataloader import get_grasp_dataloader
from utils.checkpoint import load_checkpoint
from utils.label_parser import parse_grasp_label
from train import prepare_batch


def draw_grasp_on_image(image_np, params, color, thickness=2, label_text=None):
    """
    Draw a rotated grasp rectangle on image.

    Args:
        image_np: HxWx3 BGR numpy array
        params: (x, y, w, h, theta) — x,y,w,h normalized [0,1], theta in radians
        color: BGR tuple
        thickness: line thickness
        label_text: optional text to draw near the rectangle
    """
    img = image_np.copy()
    h_img, w_img = img.shape[:2]

    x, y, w, h, theta = params
    cx = x * w_img
    cy = y * h_img
    rw = w * w_img
    rh = h * h_img
    angle_deg = math.degrees(theta)

    rect = ((cx, cy), (rw, rh), angle_deg)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(img, [box], 0, color, thickness)
    cv2.circle(img, (int(cx), int(cy)), 4, color, -1)

    if label_text:
        tx = int(cx + rw / 2 + 5)
        ty = int(cy - rh / 2 - 5)
        tx = min(max(tx, 5), w_img - 60)
        ty = min(max(ty, 15), h_img - 5)
        cv2.putText(img, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def create_single_visualization(image_np, pred_params, gt_params_list, instruction):
    """
    Create visualization for a single sample.

    Args:
        image_np: HxWx3 RGB numpy array
        pred_params: [5] predicted (x, y, w, h, theta_rad)
        gt_params_list: [N, 5] all GT grasps
        instruction: text string

    Returns:
        HxWx3 BGR annotated image
    """
    img_bgr = image_np[:, :, ::-1].copy()
    h_img, w_img = img_bgr.shape[:2]

    # Draw all GT grasps in green
    for i in range(gt_params_list.shape[0]):
        gt = gt_params_list[i]
        label = "GT" if i == 0 else None
        img_bgr = draw_grasp_on_image(img_bgr, gt, color=(0, 200, 0), thickness=2, label_text=label)

    # Draw prediction in red
    img_bgr = draw_grasp_on_image(img_bgr, pred_params, color=(0, 0, 255), thickness=2, label_text="Pred")

    # Add instruction text at top
    text = instruction[:70]
    # Background bar for text
    cv2.rectangle(img_bgr, (0, 0), (w_img, 28), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Add legend at bottom
    cv2.rectangle(img_bgr, (0, h_img - 22), (w_img, h_img), (0, 0, 0), -1)
    cv2.putText(img_bgr, "Green=GT", (5, h_img - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    cv2.putText(img_bgr, "Red=Pred", (100, h_img - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return img_bgr


def create_batch_grid(images, cols=4, cell_size=(416, 416)):
    """
    Arrange images into a grid.

    Args:
        images: list of BGR numpy arrays
        cols: number of columns
        cell_size: (width, height) per cell

    Returns:
        Grid image as numpy array
    """
    n = len(images)
    rows = math.ceil(n / cols)
    cw, ch = cell_size

    grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        resized = cv2.resize(img, (cw, ch))
        grid[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = resized

    return grid


@torch.no_grad()
def visualize_from_dataloader(model, dataloader, processor, device, cfg, num_samples=16, cols=4):
    """Generate batch visualization from a dataloader."""
    model.eval()
    vis_images = []
    count = 0

    for batch in dataloader:
        pixel_values, input_ids, attention_mask, labels = prepare_batch(
            batch, processor, device, cfg["model"]["image_size"]
        )

        output = model(pixel_values, input_ids, attention_mask)
        pred_params = output["params"].cpu().numpy()

        for i in range(pixel_values.shape[0]):
            if count >= num_samples:
                break

            # Denormalize image
            img_tensor = batch["image"][i]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img_tensor * std + mean).clamp(0, 1)
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            pred = pred_params[i]
            gt_all = labels[i].cpu().numpy()
            instruction = batch["instruction"][i]

            vis = create_single_visualization(img_np, pred, gt_all, instruction)
            vis_images.append(vis)
            count += 1

        if count >= num_samples:
            break

    grid = create_batch_grid(vis_images, cols=cols)
    return grid


@torch.no_grad()
def visualize_from_images(model, processor, image_paths, instruction, device, num_samples=16, cols=4):
    """Generate batch visualization from a list of image files."""
    model.eval()
    vis_images = []

    for img_path in image_paths[:num_samples]:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(
            text=[instruction],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        pixel_values = inputs["pixel_values"].to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        output = model(pixel_values, input_ids, attention_mask)
        pred = output["params"][0].cpu().numpy()

        img_np = np.array(image)
        # No GT available for custom images
        vis = create_single_visualization(img_np, pred, np.zeros((0, 5)), instruction)
        vis_images.append(vis)

    grid = create_batch_grid(vis_images, cols=cols)
    return grid


def main():
    parser = argparse.ArgumentParser(description="Batch visualization of grasp predictions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default=None, choices=["val", "test", "train"])
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--instruction", type=str, default="grasp the object")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--output", type=str, default="outputs/visualization.jpg")
    parser.add_argument("--cell_size", type=int, default=416)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = GraspCLIPModel(
        clip_model_name=cfg["model"]["clip_model"],
        grasp_head_hidden=cfg["model"]["grasp_head_hidden"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])

    if args.split:
        # Visualize from dataset split
        print(f"Loading {args.split} data...")
        train_loader, val_loader, test_loader = get_grasp_dataloader(
            data_dir=cfg["data"]["data_dir"],
            batch_size=args.num_samples,
            val_split=cfg["training"]["val_split"],
            test_split=cfg["training"].get("test_split", 0.15),
            num_workers=0,
            load_images=True,
            seed=cfg["training"]["seed"],
        )
        split_map = {"val": val_loader, "test": test_loader, "train": train_loader}
        dataloader = split_map[args.split]

        print(f"Generating visualization ({args.num_samples} samples, {args.cols} cols)...")
        grid = visualize_from_dataloader(
            model, dataloader, processor, device, cfg,
            num_samples=args.num_samples, cols=args.cols,
        )

    elif args.image_dir:
        # Visualize from image directory
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpeg"))
        )
        print(f"Found {len(image_paths)} images, visualizing {min(args.num_samples, len(image_paths))}...")
        grid = visualize_from_images(
            model, processor, image_paths, args.instruction, device,
            num_samples=args.num_samples, cols=args.cols,
        )

    else:
        parser.error("Specify either --split or --image_dir")
        return

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)

    h, w = grid.shape[:2]
    print(f"\nVisualization saved to {output_path} ({w}x{h})")


if __name__ == "__main__":
    main()
