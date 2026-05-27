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
from PIL import Image
from transformers import CLIPProcessor

from models.grasp_model import GraspCLIPModel
from models.grasp_head import GraspHead
from utils.losses import GraspLoss
from utils.metrics import GraspMetrics, compute_grasp_iou, compute_angle_diff
from utils.label_parser import parse_grasp_label
from utils.checkpoint import save_checkpoint, load_checkpoint


def test_model():
    """Test model forward pass with dummy data."""
    print("=" * 40)
    print("TEST: Model forward pass")
    print("=" * 40)

    device = torch.device("cpu")
    model = GraspCLIPModel(
        clip_model_name="openai/clip-vit-base-patch16",
        grasp_head_hidden=512,
        dropout=0.1,
    ).to(device)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Create dummy inputs
    dummy_image = Image.fromarray(np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8))
    dummy_text = "grasp the blue bottle"

    inputs = processor(
        text=[dummy_text],
        images=[dummy_image],
        return_tensors="pt",
        padding=True,
    )

    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"  Input shapes:")
    print(f"    pixel_values: {pixel_values.shape}")
    print(f"    input_ids: {input_ids.shape}")
    print(f"    attention_mask: {attention_mask.shape}")

    output = model(pixel_values, input_ids, attention_mask)

    print(f"  Output keys: {list(output.keys())}")
    print(f"  params shape: {output['params'].shape}")
    print(f"  params values: {output['params'][0].detach().numpy()}")

    assert output["params"].shape == (1, 5), f"Expected (1,5), got {output['params'].shape}"
    assert output["xy"].shape == (1, 2)
    assert output["wh"].shape == (1, 2)
    assert output["angle"].shape == (1, 1)

    # Check value ranges
    xy = output["xy"][0].detach().numpy()
    wh = output["wh"][0].detach().numpy()
    assert (xy >= 0).all() and (xy <= 1).all(), "xy should be in [0, 1]"
    assert (wh >= 0).all() and (wh <= 1).all(), "wh should be in [0, 1]"

    print("  [PASS] Model forward pass OK")
    print()
    return True


def test_grasp_head():
    """Test grasp head independently."""
    print("=" * 40)
    print("TEST: Grasp Head")
    print("=" * 40)

    head = GraspHead(input_dim=512, hidden_dim=256, dropout=0.0)
    features = torch.randn(4, 512)
    output = head(features)

    assert output["params"].shape == (4, 5)
    assert output["xy"].shape == (4, 2)
    assert output["wh"].shape == (4, 2)
    assert output["angle"].shape == (4, 1)
    assert output["sin_cos"].shape == (4, 2)

    print(f"  Output shape: {output['params'].shape}")
    print("  [PASS] Grasp Head OK")
    print()
    return True


def test_loss():
    """Test loss computation."""
    print("=" * 40)
    print("TEST: Loss computation")
    print("=" * 40)

    criterion = GraspLoss()

    pred = {
        "xy": torch.tensor([[0.5, 0.5]]),
        "wh": torch.tensor([[0.3, 0.2]]),
        "angle": torch.tensor([[0.1]]),
        "sin_cos": torch.tensor([[0.0998, 0.995]]),
        "params": torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.1]]),
    }
    target = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.1]])

    losses = criterion(pred, target)
    print(f"  Total loss: {losses['total'].item():.6f}")
    print(f"  XY loss: {losses['xy_loss'].item():.6f}")
    print(f"  WH loss: {losses['wh_loss'].item():.6f}")
    print(f"  Angle loss: {losses['angle_loss'].item():.6f}")

    assert losses["total"].item() >= 0
    # With matching pred/target, losses should be near zero
    assert losses["xy_loss"].item() < 0.01
    assert losses["wh_loss"].item() < 0.01

    print("  [PASS] Loss computation OK")
    print()
    return True


def test_metrics():
    """Test metrics computation."""
    print("=" * 40)
    print("TEST: Metrics")
    print("=" * 40)

    metrics = GraspMetrics(iou_threshold=0.25, angle_threshold=30.0)

    # Perfect prediction
    pred = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.0]])
    gt = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.0]])
    metrics.update(pred, gt)

    # Slightly off prediction
    pred2 = torch.tensor([[0.52, 0.48, 0.28, 0.19, 0.1]])
    gt2 = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.0]])
    metrics.update(pred2, gt2)

    results = metrics.compute()
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Mean IoU: {results['mean_iou']:.4f}")
    print(f"  Mean Angle Diff: {results['mean_angle_diff']:.2f}°")

    # IoU test
    iou = compute_grasp_iou(
        (0.5, 0.5, 0.3, 0.2, 0.0),
        (0.5, 0.5, 0.3, 0.2, 0.0),
    )
    assert abs(iou - 1.0) < 0.01, f"Perfect overlap should give IoU~1.0, got {iou}"

    # Angle diff test
    diff = compute_angle_diff(0.0, 15.0)
    assert abs(diff - 15.0) < 0.01

    diff_wrap = compute_angle_diff(170.0, -170.0)
    assert diff_wrap < 25.0  # Should handle wrapping

    print("  [PASS] Metrics OK")
    print()
    return True


def test_label_parser():
    """Test label parsing from various formats."""
    print("=" * 40)
    print("TEST: Label parser")
    print("=" * 40)

    # Test [5] format with pixel values
    label1 = torch.tensor([208.0, 208.0, 100.0, 50.0, 0.5])
    result1 = parse_grasp_label(label1, image_size=416)
    assert result1.shape == (5,)
    assert 0 <= result1[0] <= 1
    print(f"  [5] pixel -> normalized: {result1.numpy()}")

    # Test already normalized
    label2 = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.3])
    result2 = parse_grasp_label(label2, image_size=416)
    assert abs(result2[0] - 0.5) < 0.01
    print(f"  [5] normalized -> {result2.numpy()}")

    # Test [N, 5] format
    label3 = torch.tensor([[208.0, 208.0, 100.0, 50.0, 0.5], [100.0, 100.0, 80.0, 40.0, 0.0]])
    result3 = parse_grasp_label(label3, image_size=416)
    assert result3.shape == (5,)
    print(f"  [N,5] -> {result3.numpy()}")

    # Test [4, 2] corners format
    label4 = torch.tensor([[100.0, 100.0], [200.0, 100.0], [200.0, 150.0], [100.0, 150.0]])
    result4 = parse_grasp_label(label4, image_size=416)
    assert result4.shape == (5,)
    print(f"  [4,2] corners -> {result4.numpy()}")

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

    head = GraspHead(input_dim=64, hidden_dim=32, dropout=0.0)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    # Save
    tmp_path = os.path.join(tempfile.gettempdir(), "test_ckpt.pt")
    save_checkpoint(head, optimizer, None, epoch=5, metrics={"accuracy": 0.85}, save_path=tmp_path)
    assert os.path.exists(tmp_path)

    # Load
    head2 = GraspHead(input_dim=64, hidden_dim=32, dropout=0.0)
    ckpt = load_checkpoint(tmp_path, head2)
    assert ckpt["epoch"] == 5
    assert ckpt["metrics"]["accuracy"] == 0.85

    # Check weights match
    for p1, p2 in zip(head.parameters(), head2.parameters()):
        assert torch.allclose(p1, p2)

    os.remove(tmp_path)
    print("  [PASS] Checkpoint save/load OK")
    print()
    return True


def test_data_pipeline():
    """Test data loading pipeline (requires data to be downloaded)."""
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
    train_loader, val_loader = get_grasp_dataloader(
        data_dir="./data",
        batch_size=4,
        num_workers=0,
        load_images=True,
    )

    batch = next(iter(train_loader))
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Instructions: {batch['instruction'][:2]}")
    print(f"  Label type: {type(batch['positive_label'])}")

    if isinstance(batch["positive_label"], torch.Tensor):
        print(f"  Label shape: {batch['positive_label'].shape}")

    print("  [PASS] Data pipeline OK")
    print()
    return True


def test_training_step():
    """Test one full training step."""
    print("=" * 40)
    print("TEST: Training step")
    print("=" * 40)

    device = torch.device("cpu")
    model = GraspCLIPModel(
        clip_model_name="openai/clip-vit-base-patch16",
        grasp_head_hidden=512,
    ).to(device)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    criterion = GraspLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy batch
    batch_size = 2
    images = [Image.fromarray(np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)) for _ in range(batch_size)]
    texts = ["grasp the red cup", "pick up the green bottle"]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    targets = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.1], [0.3, 0.7, 0.2, 0.15, -0.2]], device=device)

    # Forward
    model.train()
    output = model(pixel_values, input_ids, attention_mask)
    losses = criterion(output, targets)

    # Backward
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    print(f"  Loss: {losses['total'].item():.4f}")
    print(f"  Predictions: {output['params'].detach().numpy()}")
    print("  [PASS] Training step OK")
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "model", "head", "loss", "metrics", "label", "checkpoint", "data", "training"])
    args = parser.parse_args()

    tests = {
        "head": test_grasp_head,
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
