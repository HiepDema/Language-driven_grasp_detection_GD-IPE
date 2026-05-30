"""
Evaluation script for GraspDetection model.

Usage:
    python eval.py --checkpoint checkpoints/best_epoch050_0.8500.pt
    python eval.py --checkpoint checkpoints/latest.pt --visualize --num_vis 20
"""

import argparse
import math
import json
from pathlib import Path

import yaml
import torch
import numpy as np
from torch.cuda.amp import autocast
from transformers import BertTokenizer
from tqdm import tqdm

from dataloader import get_grasp_dataloader
from models.grasp_detection import GraspDetectionModel
from utils.metrics import GraspMetrics, pred_to_params
from utils.checkpoint import load_checkpoint
from utils.label_parser import parse_grasp_label
from utils.visualization import visualize_prediction
from train import prepare_batch


def evaluate(model, dataloader, tokenizer, device, cfg, visualize=False, num_vis=10, output_dir="outputs"):
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
            images, input_ids, attention_mask, labels = prepare_batch(
                batch, tokenizer, device, cfg
            )

            output = model(images, input_ids, attention_mask)
            metrics.update(output, labels)

            params = pred_to_params(output).cpu().numpy()
            params_deg = params.copy()
            params_deg[:, 4] = np.degrees(params_deg[:, 4])

            for i in range(params.shape[0]):
                gt_i = labels[i].cpu().clone()
                gt_i[:, 4] = gt_i[:, 4] * (180.0 / math.pi)
                all_predictions.append({
                    "sha": batch["sha"][i],
                    "instruction": batch["instruction"][i],
                    "pred": params_deg[i].tolist(),
                    "gt": gt_i.numpy().tolist(),
                })

            if visualize and vis_count < num_vis:
                for i in range(min(images.shape[0], num_vis - vis_count)):
                    img_tensor = batch["image"][i]
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = (img_tensor * std + mean).clamp(0, 1)
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_bgr = img_np[:, :, ::-1].copy()

                    pred_p = tuple(params_deg[i])
                    gt_first = labels[i][0].cpu().numpy()
                    gt_first[4] = math.degrees(gt_first[4])
                    gt_p = tuple(gt_first)

                    visualize_prediction(
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
    parser = argparse.ArgumentParser(description="Evaluate GraspDetection model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_vis", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    split_map = {"val": val_loader, "test": test_loader, "train": train_loader}
    eval_loader = split_map[args.split]

    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["bert_model"])

    print(f"\nLoading model from {args.checkpoint}...")
    model = GraspDetectionModel(d_model=cfg["model"]["d_model"]).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\nRunning evaluation...")
    results, predictions = evaluate(
        model, eval_loader, tokenizer, device, cfg,
        visualize=args.visualize,
        num_vis=args.num_vis,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Grasp Accuracy (IoU>{cfg['eval']['iou_threshold']} & Angle<{cfg['eval']['angle_threshold']}deg): "
          f"{results['accuracy']:.4f}")
    print(f"  IoU Accuracy:   {results['iou_accuracy']:.4f}")
    print(f"  Angle Accuracy: {results['angle_accuracy']:.4f}")
    print(f"  Mean IoU:       {results['mean_iou']:.4f}")
    print(f"  Mean Angle Diff: {results['mean_angle_diff']:.2f}deg")
    print(f"  Mean XY Error:  {results['mean_xy_error']:.4f}")
    print(f"  Mean WH Error:  {results['mean_wh_error']:.4f}")
    print(f"  Total Samples:  {results['total_samples']}")

    # Inference speed benchmark
    import time
    model.eval()
    dummy_image = torch.randn(1, 3, 416, 416).to(device)
    dummy_ids = torch.randint(0, 1000, (1, 128)).to(device)
    dummy_mask = torch.ones(1, 128, dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy_image, dummy_ids, dummy_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(50):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy_image, dummy_ids, dummy_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    print(f"\n  Inference speed:")
    print(f"    Latency: {np.mean(times):.1f} +/- {np.std(times):.1f} ms")
    print(f"    FPS:     {1000.0 / np.mean(times):.1f}")
    print(f"{'='*60}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    print(f"\nResults saved to {results_path}")

    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {predictions_path}")

    if args.visualize:
        print(f"Visualizations saved to {output_dir}/visualizations/")


if __name__ == "__main__":
    main()
