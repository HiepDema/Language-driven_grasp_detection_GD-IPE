"""
Test script: quick sanity checks for model, data pipeline, and training loop.

Usage:
    python test.py                    # Run all tests
    python test.py --test model       # Test model only
    python test.py --test data        # Test data pipeline only
    python test.py --test training    # Test one training step
"""

import argparse
import sys
import time

import torch
import numpy as np
from transformers import BertTokenizer

from models.grasp_detection import GraspDetectionModel
from utils.losses import GraspLoss
from utils.metrics import GraspMetrics, compute_grasp_iou, compute_angle_diff, pred_to_params
from utils.label_parser import parse_grasp_label
from utils.checkpoint import save_checkpoint, load_checkpoint


def test_model():
    """Test model forward pass with dummy data."""
    print("=" * 40)
    print("TEST: Model forward pass")
    print("=" * 40)

    device = torch.device("cpu")
    model = GraspDetectionModel(d_model=512).to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dummy_image = torch.randn(2, 3, 416, 416).to(device)
    tokens = tokenizer(
        ["grasp the blue bottle", "pick up the red cup"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    print(f"  Input shapes:")
    print(f"    image: {dummy_image.shape}")
    print(f"    input_ids: {input_ids.shape}")
    print(f"    attention_mask: {attention_mask.shape}")

    output = model(dummy_image, input_ids, attention_mask)

    print(f"  Output keys: {list(output.keys())}")
    print(f"  center shape: {output['center'].shape}")
    print(f"  size shape: {output['size'].shape}")
    print(f"  sin2_cos2 shape: {output['sin2_cos2'].shape}")

    assert output["center"].shape == (2, 2)
    assert output["size"].shape == (2, 2)
    assert output["sin2_cos2"].shape == (2, 2)

    center = output["center"][0].detach().numpy()
    size = output["size"][0].detach().numpy()
    sin2_cos2 = output["sin2_cos2"][0].detach().numpy()
    assert (center >= 0).all() and (center <= 1).all(), "center should be in [0, 1]"
    assert (size >= 0).all() and (size <= 1).all(), "size should be in [0, 1]"
    assert (sin2_cos2 >= 0).all() and (sin2_cos2 <= 1).all(), "sin2_cos2 should be in [0, 1]"
    assert abs(sin2_cos2.sum() - 1.0) < 1e-5, "sin2_cos2 should sum to 1 (softmax)"

    params = pred_to_params(output)
    assert params.shape == (2, 5)
    print(f"  params (converted): {params[0].detach().numpy()}")

    print("  [PASS] Model forward pass OK")
    print()
    return True


def test_loss():
    """Test loss computation."""
    print("=" * 40)
    print("TEST: Loss computation")
    print("=" * 40)

    criterion = GraspLoss()

    # theta=pi/2 -> half=pi/4 -> sin2=0.5, cos2=0.5
    import math
    pred = {
        "center": torch.tensor([[0.5, 0.5]]),
        "size": torch.tensor([[0.3, 0.2]]),
        "sin2_cos2": torch.tensor([[0.5, 0.5]]),
    }
    target = [torch.tensor([[0.5, 0.5, 0.3, 0.2, math.pi / 2]])]

    losses = criterion(pred, target)
    print(f"  Total loss: {losses['total'].item():.6f}")
    print(f"  Center loss: {losses['center_loss'].item():.6f}")
    print(f"  Size loss: {losses['size_loss'].item():.6f}")
    print(f"  Angle loss: {losses['angle_loss'].item():.6f}")

    assert losses["total"].item() >= 0
    assert losses["center_loss"].item() < 0.01
    assert losses["size_loss"].item() < 0.01

    print("  [PASS] Loss computation OK")
    print()
    return True


def test_metrics():
    """Test metrics computation."""
    print("=" * 40)
    print("TEST: Metrics")
    print("=" * 40)

    metrics = GraspMetrics(iou_threshold=0.25, angle_threshold=30.0)

    # theta=pi/2 -> half=pi/4 -> sin2=0.5, cos2=0.5
    pred = {
        "center": torch.tensor([[0.5, 0.5]]),
        "size": torch.tensor([[0.3, 0.2]]),
        "sin2_cos2": torch.tensor([[0.5, 0.5]]),
    }
    gt = [torch.tensor([[0.5, 0.5, 0.3, 0.2, math.pi / 2]])]
    metrics.update(pred, gt)

    results = metrics.compute()
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Mean IoU: {results['mean_iou']:.4f}")
    print(f"  Mean Angle Diff: {results['mean_angle_diff']:.2f}deg")

    iou = compute_grasp_iou(
        (0.5, 0.5, 0.3, 0.2, 0.0),
        (0.5, 0.5, 0.3, 0.2, 0.0),
    )
    assert abs(iou - 1.0) < 0.01, f"Perfect overlap should give IoU~1.0, got {iou}"

    diff = compute_angle_diff(0.0, 15.0)
    assert abs(diff - 15.0) < 0.01

    print("  [PASS] Metrics OK")
    print()
    return True


def test_label_parser():
    """Test label parsing from various formats."""
    print("=" * 40)
    print("TEST: Label parser")
    print("=" * 40)

    label1 = torch.tensor([208.0, 208.0, 100.0, 50.0, 0.5])
    result1 = parse_grasp_label(label1, image_size=416)
    assert result1.shape[1] == 5
    assert 0 <= result1[0, 0] <= 1
    print(f"  [5] pixel -> normalized: {result1[0].numpy()}")

    label2 = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.3])
    result2 = parse_grasp_label(label2, image_size=416)
    assert abs(result2[0, 0] - 0.5) < 0.01
    print(f"  [5] normalized -> {result2[0].numpy()}")

    label3 = torch.tensor([[208.0, 208.0, 100.0, 50.0, 0.5], [100.0, 100.0, 80.0, 40.0, 0.0]])
    result3 = parse_grasp_label(label3, image_size=416)
    assert result3.shape == (2, 5)
    print(f"  [N,5] -> {result3.numpy()}")

    print("  [PASS] Label parser OK")
    print()
    return True


def test_checkpoint():
    """Test checkpoint save/load."""
    print("=" * 40)
    print("TEST: Checkpoint save/load")
    print("=" * 40)

    import tempfile
    import os

    model = GraspDetectionModel(d_model=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tmp_path = os.path.join(tempfile.gettempdir(), "test_ckpt.pt")
    save_checkpoint(model, optimizer, None, epoch=5, metrics={"accuracy": 0.85}, save_path=tmp_path)
    assert os.path.exists(tmp_path)

    model2 = GraspDetectionModel(d_model=64)
    ckpt = load_checkpoint(tmp_path, model2)
    assert ckpt["epoch"] == 5
    assert ckpt["metrics"]["accuracy"] == 0.85

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

    os.remove(tmp_path)
    print("  [PASS] Checkpoint save/load OK")
    print()
    return True


def test_data_pipeline():
    """Test data loading pipeline."""
    print("=" * 40)
    print("TEST: Data pipeline")
    print("=" * 40)

    from pathlib import Path
    data_dir = Path("./data")

    if not (data_dir / "grasp_instructions").exists():
        print("  [SKIP] Data not downloaded yet. Run download_subset.py first.")
        print()
        return True

    from dataloader import get_grasp_dataloader
    train_loader, val_loader, test_loader = get_grasp_dataloader(
        data_dir="./data",
        batch_size=4,
        num_workers=0,
        load_images=True,
    )

    batch = next(iter(train_loader))
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Instructions: {batch['instruction'][:2]}")

    print("  [PASS] Data pipeline OK")
    print()
    return True


def test_training_step():
    """Test one full training step."""
    print("=" * 40)
    print("TEST: Training step")
    print("=" * 40)

    device = torch.device("cpu")
    model = GraspDetectionModel(d_model=512).to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    criterion = GraspLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dummy_image = torch.randn(2, 3, 416, 416).to(device)
    tokens = tokenizer(
        ["grasp the red cup", "pick up the green bottle"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    targets = [
        torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.5]]),
        torch.tensor([[0.3, 0.7, 0.2, 0.15, 1.2]]),
    ]

    model.train()
    output = model(dummy_image, input_ids, attention_mask)
    losses = criterion(output, targets)

    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    print(f"  Loss: {losses['total'].item():.4f}")
    params = pred_to_params(output)
    print(f"  Predictions: {params.detach().numpy()}")
    print("  [PASS] Training step OK")
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "model", "loss", "metrics", "label", "checkpoint", "data", "training"])
    args = parser.parse_args()

    tests = {
        "model": test_model,
        "loss": test_loss,
        "metrics": test_metrics,
        "label": test_label_parser,
        "checkpoint": test_checkpoint,
        "data": test_data_pipeline,
        "training": test_training_step,
    }

    if args.test == "all":
        run_tests = tests
    else:
        run_tests = {args.test: tests[args.test]}

    passed = 0
    failed = 0
    start = time.time()

    for name, test_fn in run_tests.items():
        try:
            result = test_fn()
            if result:
                passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    elapsed = time.time() - start
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    print(f"{'='*40}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
