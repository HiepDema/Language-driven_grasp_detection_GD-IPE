"""
Evaluation script for GraspCLIP model.

Evaluates a trained model on the validation set and reports metrics.

Usage:
    python eval.py --checkpoint checkpoints/best_epoch050_0.8500.pt
    python eval.py --checkpoint checkpoints/latest.pt --visualize --num_vis 20
"""

import argparse
import os
import json
from pathlib import Path

import yaml
import torch
from torch.cuda.amp import autocast
from transformers import CLIPProcessor
from tqdm import tqdm

from dataloader import get_grasp_dataloader
from models.grasp_model import GraspCLIPModel
from utils.metrics import GraspMetrics
from utils.checkpoint import load_checkpoint
from utils.label_parser import parse_grasp_label
from utils.visualization import visualize_prediction
from train import prepare_batch


def evaluate(model, dataloader, processor, device, cfg, visualize=False, num_vis=10, output_dir="outputs"):
    model.eval()
    metrics = GraspMetrics(
        iou_threshold=cfg["eval"]["iou_threshold"],
        angle_threshold=cfg["eval"]["angle_threshold"],
    )

    vis_count = 0
    vis_dir = Path(output_dir) / "visualizations"
    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            pixel_values, input_ids, attention_mask, labels = prepare_batch(
                batch, processor, device, cfg["model"]["image_size"]
            )

            output = model(pixel_values, input_ids, attention_mask)
            metrics.update(output["params"], labels)

            # Store predictions
            pred_np = output["params"].cpu().numpy()
            gt_np = labels.cpu().numpy()
            for i in range(pred_np.shape[0]):
                all_predictions.append({
                    "sha": batch["sha"][i],
                    "instruction": batch["instruction"][i],
                    "pred": pred_np[i].tolist(),
                    "gt": gt_np[i].tolist(),
                })

            # Visualize
            if visualize and vis_count < num_vis:
                import numpy as np
                for i in range(min(pixel_values.shape[0], num_vis - vis_count)):
                    img_tensor = batch["image"][i]
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = (img_tensor * std + mean).clamp(0, 1)
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_bgr = img_np[:, :, ::-1].copy()

                    pred_p = tuple(pred_np[i])
                    gt_p = tuple(gt_np[i])

                    vis = visualize_prediction(
                        img_bgr,
                        pred_params=pred_p,
                        gt_params=gt_p,
                        instruction=batch["instruction"][i],
                        save_path=str(vis_dir / f"vis_{vis_count:04d}.jpg"),
                    )
                    vis_count += 1

    results = metrics.compute()
    return results, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraspCLIP model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_vis", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--split", type=str, default="val", choices=["val", "all"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader = get_grasp_dataloader(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        val_split=cfg["training"]["val_split"],
        num_workers=cfg["training"]["num_workers"],
        load_images=cfg["data"]["load_images"],
        seed=cfg["training"]["seed"],
    )
    eval_loader = val_loader if args.split == "val" else train_loader

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = GraspCLIPModel(
        clip_model_name=cfg["model"]["clip_model"],
        grasp_head_hidden=cfg["model"]["grasp_head_hidden"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])

    # Evaluate
    print("\nRunning evaluation...")
    results, predictions = evaluate(
        model, eval_loader, processor, device, cfg,
        visualize=args.visualize,
        num_vis=args.num_vis,
        output_dir=args.output_dir,
    )

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Grasp Accuracy (IoU>{cfg['eval']['iou_threshold']} & Angle<{cfg['eval']['angle_threshold']}°): "
          f"{results['accuracy']:.4f}")
    print(f"  IoU Accuracy:   {results['iou_accuracy']:.4f}")
    print(f"  Angle Accuracy: {results['angle_accuracy']:.4f}")
    print(f"  Mean IoU:       {results['mean_iou']:.4f}")
    print(f"  Mean Angle Diff: {results['mean_angle_diff']:.2f}°")
    print(f"  Mean XY Error:  {results['mean_xy_error']:.4f}")
    print(f"  Mean WH Error:  {results['mean_wh_error']:.4f}")
    print(f"  Total Samples:  {results['total_samples']}")
    print(f"{'='*60}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {predictions_path}")

    if args.visualize:
        print(f"Visualizations saved to {output_dir}/visualizations/")


if __name__ == "__main__":
    main()
