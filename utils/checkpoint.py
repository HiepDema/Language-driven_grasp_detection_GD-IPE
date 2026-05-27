"""
Checkpoint management: save, load, and track best models.
"""

import os
import json
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    save_path: str,
    scaler=None,
):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device="cpu",
) -> dict:
    """Load a training checkpoint. Returns the checkpoint dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


class CheckpointManager:
    """Manages saving checkpoints and keeping only the top-k best."""

    def __init__(self, save_dir: str, keep_top_k: int = 3, metric_name: str = "accuracy"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric_name = metric_name
        self.history_file = self.save_dir / "checkpoint_history.json"
        self.history = self._load_history()

    def _load_history(self) -> list:
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []

    def _save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metrics: dict,
        scaler=None,
    ) -> Optional[str]:
        """Save checkpoint if it's among the top-k best. Returns save path or None."""
        metric_value = metrics.get(self.metric_name, 0.0)

        # Always save latest
        latest_path = str(self.save_dir / "latest.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, metrics, latest_path, scaler)

        # Check if this is a top-k checkpoint
        best_path = str(self.save_dir / f"best_epoch{epoch:03d}_{metric_value:.4f}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_path, scaler)

        self.history.append({
            "epoch": epoch,
            "path": best_path,
            "metric": metric_value,
        })

        # Sort by metric (higher is better) and prune
        self.history.sort(key=lambda x: x["metric"], reverse=True)
        while len(self.history) > self.keep_top_k:
            removed = self.history.pop()
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])

        self._save_history()
        return best_path

    @property
    def best_checkpoint(self) -> Optional[str]:
        if not self.history:
            return None
        return self.history[0]["path"]

    @property
    def best_metric(self) -> float:
        if not self.history:
            return 0.0
        return self.history[0]["metric"]
