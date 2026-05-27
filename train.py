"""
Training script for GraspCLIP model.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/latest.pt
"""

import os
import sys
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import CLIPProcessor

from dataloader import get_grasp_dataloader
from models.grasp_model import GraspCLIPModel
from utils.losses import GraspLoss
from utils.metrics import GraspMetrics
from utils.checkpoint import CheckpointManager, load_checkpoint
from utils.label_parser import parse_grasp_label


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_batch(batch, processor, device, image_size=416):
    """Prepare a batch for the model: process images through CLIP processor and tokenize text."""
    images_pil = []
    for img_tensor in batch["image"]:
        # Denormalize from ImageNet normalization back to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img_tensor * std + mean
        img = img.clamp(0, 1)
        # Convert to PIL for CLIP processor
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        from PIL import Image
        images_pil.append(Image.fromarray(img_np))

    # Process through CLIP
    inputs = processor(
        text=batch["instruction"],
        images=images_pil,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )

    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Parse labels
    labels_raw = batch["positive_label"]
    if isinstance(labels_raw, torch.Tensor):
        labels = torch.stack([parse_grasp_label(labels_raw[i], image_size) for i in range(labels_raw.shape[0])])
    else:
        labels = torch.stack([parse_grasp_label(l, image_size) for l in labels_raw])
    labels = labels.to(device)

    return pixel_values, input_ids, attention_mask, labels


def train_one_epoch(model, train_loader, optimizer, criterion, processor, device, scaler, cfg):
    model.train()
    total_loss = 0.0
    total_xy = 0.0
    total_wh = 0.0
    total_angle = 0.0
    num_batches = 0
    metrics = GraspMetrics(
        iou_threshold=cfg["eval"]["iou_threshold"],
        angle_threshold=cfg["eval"]["angle_threshold"],
    )

    for batch_idx, batch in enumerate(train_loader):
        pixel_values, input_ids, attention_mask, labels = prepare_batch(
            batch, processor, device, cfg["model"]["image_size"]
        )

        optimizer.zero_grad()

        if cfg["training"]["mixed_precision"] and device.type == "cuda":
            with autocast():
                output = model(pixel_values, input_ids, attention_mask)
                losses = criterion(output, labels)
            scaler.scale(losses["total"]).backward()
            if cfg["training"]["clip_grad_norm"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(pixel_values, input_ids, attention_mask)
            losses = criterion(output, labels)
            losses["total"].backward()
            if cfg["training"]["clip_grad_norm"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["clip_grad_norm"])
            optimizer.step()

        total_loss += losses["total"].item()
        total_xy += losses["xy_loss"].item()
        total_wh += losses["wh_loss"].item()
        total_angle += losses["angle_loss"].item()
        num_batches += 1

        metrics.update(output["params"].detach(), labels)

        if (batch_idx + 1) % 50 == 0:
            avg = total_loss / num_batches
            print(f"    Step [{batch_idx+1}/{len(train_loader)}] loss={avg:.4f}")

    train_metrics = metrics.compute()
    train_metrics.update({
        "loss": total_loss / max(num_batches, 1),
        "xy_loss": total_xy / max(num_batches, 1),
        "wh_loss": total_wh / max(num_batches, 1),
        "angle_loss": total_angle / max(num_batches, 1),
    })
    return train_metrics


@torch.no_grad()
def validate(model, val_loader, criterion, processor, device, cfg):
    model.eval()
    metrics = GraspMetrics(
        iou_threshold=cfg["eval"]["iou_threshold"],
        angle_threshold=cfg["eval"]["angle_threshold"],
    )
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        pixel_values, input_ids, attention_mask, labels = prepare_batch(
            batch, processor, device, cfg["model"]["image_size"]
        )

        if cfg["training"]["mixed_precision"] and device.type == "cuda":
            with autocast():
                output = model(pixel_values, input_ids, attention_mask)
                losses = criterion(output, labels)
        else:
            output = model(pixel_values, input_ids, attention_mask)
            losses = criterion(output, labels)

        total_loss += losses["total"].item()
        num_batches += 1

        metrics.update(output["params"], labels)

    results = metrics.compute()
    results["val_loss"] = total_loss / max(num_batches, 1)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train GraspCLIP model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["checkpoint"]["resume"] = args.resume

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Seed
    torch.manual_seed(cfg["training"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["training"]["seed"])

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_grasp_dataloader(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        val_split=cfg["training"]["val_split"],
        test_split=cfg["training"].get("test_split", 0.1),
        num_workers=cfg["training"]["num_workers"],
        load_images=cfg["data"]["load_images"],
        seed=cfg["training"]["seed"],
    )

    # Model
    print("\nInitializing model...")
    model = GraspCLIPModel(
        clip_model_name=cfg["model"]["clip_model"],
        grasp_head_hidden=cfg["model"]["grasp_head_hidden"],
        dropout=cfg["model"]["dropout"],
        freeze_clip=cfg["model"]["freeze_clip"],
    ).to(device)

    processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = GraspLoss(
        xy_weight=cfg["loss"]["xy_weight"],
        wh_weight=cfg["loss"]["wh_weight"],
        angle_weight=cfg["loss"]["angle_weight"],
        smooth_l1_beta=cfg["loss"]["smooth_l1_beta"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs = cfg["training"]["num_epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = GradScaler() if cfg["training"]["mixed_precision"] else None

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        save_dir=cfg["checkpoint"]["save_dir"],
        keep_top_k=cfg["checkpoint"]["keep_top_k"],
        metric_name="accuracy",
    )

    # Resume
    start_epoch = 0
    if cfg["checkpoint"]["resume"]:
        print(f"\nResuming from {cfg['checkpoint']['resume']}...")
        ckpt = load_checkpoint(
            cfg["checkpoint"]["resume"], model, optimizer, scheduler, scaler, device
        )
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Unfreeze CLIP after freeze_clip_epochs
        if epoch == cfg["model"]["freeze_clip_epochs"] and cfg["model"]["freeze_clip"]:
            print(f"  Unfreezing CLIP backbone at epoch {epoch}")
            model.unfreeze_clip_params()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, processor, device, scaler, cfg
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, processor, device, cfg)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
        print(f"  [Train] Loss: {train_metrics['loss']:.4f} "
              f"(xy={train_metrics['xy_loss']:.4f}, wh={train_metrics['wh_loss']:.4f}, "
              f"angle={train_metrics['angle_loss']:.4f})")
        print(f"  [Train] Accuracy: {train_metrics.get('accuracy', 0):.4f} "
              f"(IoU: {train_metrics.get('iou_accuracy', 0):.4f}, "
              f"Angle: {train_metrics.get('angle_accuracy', 0):.4f})")
        print(f"  [Train] Mean IoU: {train_metrics.get('mean_iou', 0):.4f}, "
              f"Mean Angle Diff: {train_metrics.get('mean_angle_diff', 0):.2f}°")
        print(f"  [Val]   Loss: {val_metrics['val_loss']:.4f}")
        print(f"  [Val]   Accuracy: {val_metrics.get('accuracy', 0):.4f} "
              f"(IoU: {val_metrics.get('iou_accuracy', 0):.4f}, "
              f"Angle: {val_metrics.get('angle_accuracy', 0):.4f})")
        print(f"  [Val]   Mean IoU: {val_metrics.get('mean_iou', 0):.4f}, "
              f"Mean Angle Diff: {val_metrics.get('mean_angle_diff', 0):.2f}°")

        # Save periodic checkpoint every N epochs
        if (epoch + 1) % cfg["checkpoint"]["save_every"] == 0:
            periodic_path = os.path.join(cfg["checkpoint"]["save_dir"], f"epoch_{epoch+1:03d}.pt")
            from utils.checkpoint import save_checkpoint as _save_ckpt
            _save_ckpt(model, optimizer, scheduler, epoch, val_metrics, periodic_path, scaler)
            print(f"  Periodic checkpoint saved: {periodic_path}")

        # Save best checkpoint + run test if val accuracy improves
        current_acc = val_metrics.get("accuracy", 0.0)
        ckpt_manager.save(model, optimizer, scheduler, epoch, val_metrics, scaler)

        if current_acc >= ckpt_manager.best_metric:
            print(f"  New best val accuracy: {current_acc:.4f} — running test...")
            test_metrics = validate(model, test_loader, criterion, processor, device, cfg)
            print(f"  Test Accuracy: {test_metrics.get('accuracy', 0):.4f} "
                  f"(IoU: {test_metrics.get('iou_accuracy', 0):.4f}, "
                  f"Angle: {test_metrics.get('angle_accuracy', 0):.4f})")
            print(f"  Test Mean IoU: {test_metrics.get('mean_iou', 0):.4f}, "
                  f"Mean Angle Diff: {test_metrics.get('mean_angle_diff', 0):.2f}°")

        print()

    # Final test with best checkpoint
    print(f"{'='*60}")
    print(f"Training complete! Loading best checkpoint for final test...")
    print(f"{'='*60}")
    if ckpt_manager.best_checkpoint and os.path.exists(ckpt_manager.best_checkpoint):
        load_checkpoint(ckpt_manager.best_checkpoint, model, device=device)
        test_metrics = validate(model, test_loader, criterion, processor, device, cfg)
        print(f"\nFinal Test Results (best val checkpoint):")
        print(f"  Accuracy:       {test_metrics.get('accuracy', 0):.4f}")
        print(f"  IoU Accuracy:   {test_metrics.get('iou_accuracy', 0):.4f}")
        print(f"  Angle Accuracy: {test_metrics.get('angle_accuracy', 0):.4f}")
        print(f"  Mean IoU:       {test_metrics.get('mean_iou', 0):.4f}")
        print(f"  Mean Angle Diff: {test_metrics.get('mean_angle_diff', 0):.2f}°")
    print(f"\nBest val accuracy: {ckpt_manager.best_metric:.4f}")
    print(f"Best checkpoint: {ckpt_manager.best_checkpoint}")


if __name__ == "__main__":
    main()
