"""
Inference script for GraspDetection model.

Usage:
    python inference.py --checkpoint checkpoints/best.pt \
                        --image path/to/image.jpg \
                        --instruction "grasp the blue bottle"
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from models.grasp_detection import GraspDetectionModel
from utils.checkpoint import load_checkpoint
from utils.metrics import pred_to_params
from utils.visualization import visualize_prediction


class GraspPredictor:
    """Inference wrapper for GraspDetection model."""

    def __init__(
        self,
        checkpoint_path: str,
        d_model: int = 512,
        bert_model: str = "bert-base-uncased",
        max_seq_len: int = 128,
        device: str = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = GraspDetectionModel(d_model=d_model).to(self.device)
        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_seq_len = max_seq_len

        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image, instruction: str) -> dict:
        """
        Predict grasp pose for an image and instruction.

        Returns:
            dict with x, y, w, h (normalized [0,1]) and theta (degrees)
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        tokens = self.tokenizer(
            [instruction],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        output = self.model(img_tensor, input_ids, attention_mask)
        params = pred_to_params(output)[0].cpu().numpy()

        return {
            "x": float(params[0]),
            "y": float(params[1]),
            "w": float(params[2]),
            "h": float(params[3]),
            "theta": float(math.degrees(params[4])),
            "params": params,
        }

    def predict_and_visualize(self, image_path: str, instruction: str, save_path: str = None) -> tuple:
        """Predict and draw grasp on image."""
        image = Image.open(image_path).convert("RGB")
        result = self.predict(image, instruction)

        img_np = np.array(image)[:, :, ::-1].copy()
        params_deg = (result["x"], result["y"], result["w"], result["h"], result["theta"])
        vis = visualize_prediction(
            img_np,
            pred_params=params_deg,
            instruction=instruction,
            save_path=save_path,
        )
        return result, vis

    @torch.no_grad()
    def benchmark(self, num_runs: int = 100, warmup: int = 10) -> dict:
        """Measure inference latency."""
        import time
        dummy_img = torch.randn(1, 3, 416, 416).to(self.device)
        dummy_ids = torch.randint(0, 1000, (1, self.max_seq_len)).to(self.device)
        dummy_mask = torch.ones(1, self.max_seq_len, dtype=torch.long).to(self.device)

        for _ in range(warmup):
            self.model(dummy_img, dummy_ids, dummy_mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.model(dummy_img, dummy_ids, dummy_mask)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times_ms = [t * 1000 for t in times]
        return {
            "mean_ms": np.mean(times_ms),
            "std_ms": np.std(times_ms),
            "fps": 1000.0 / np.mean(times_ms),
            "device": str(self.device),
        }


def main():
    parser = argparse.ArgumentParser(description="GraspDetection Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    predictor = GraspPredictor(checkpoint_path=args.checkpoint, device=args.device)
    print(f"Model loaded on {predictor.device}")

    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    print(f"\nBenchmarking inference speed...")
    bench = predictor.benchmark(num_runs=50, warmup=5)
    print(f"  Latency: {bench['mean_ms']:.1f} +/- {bench['std_ms']:.1f} ms")
    print(f"  FPS:     {bench['fps']:.1f}")

    if args.image:
        print(f"\nImage: {args.image}")
        print(f"Instruction: \"{args.instruction}\"")

        save_path = str(output_dir / f"result_{Path(args.image).stem}.jpg")
        result, vis = predictor.predict_and_visualize(args.image, args.instruction, save_path=save_path)

        print(f"\nPrediction:")
        print(f"  Center (x, y): ({result['x']:.4f}, {result['y']:.4f})")
        print(f"  Size (w, h):   ({result['w']:.4f}, {result['h']:.4f})")
        print(f"  Angle:         {result['theta']:.2f}deg")
        print(f"\nVisualization saved to {save_path}")

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = sorted(
            list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        )
        print(f"\nProcessing {len(image_files)} images...")

        import json
        results = []
        for img_path in image_files:
            save_path = str(output_dir / f"result_{img_path.stem}.jpg")
            result, _ = predictor.predict_and_visualize(str(img_path), args.instruction, save_path=save_path)
            results.append({"image": img_path.name, **result})
            print(f"  {img_path.name}: center=({result['x']:.3f}, {result['y']:.3f}), "
                  f"size=({result['w']:.3f}, {result['h']:.3f}), angle={result['theta']:.1f}deg")

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_dir}/results.json")
    else:
        parser.error("Either --image or --image_dir must be specified")


if __name__ == "__main__":
    main()
