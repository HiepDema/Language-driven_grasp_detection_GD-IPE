"""
Inference script for GraspCLIP model.

Run grasp prediction on a single image + text instruction.

Usage:
    python inference.py --checkpoint checkpoints/best.pt \
                        --image path/to/image.jpg \
                        --instruction "grasp the blue bottle"

    python inference.py --checkpoint checkpoints/best.pt \
                        --image_dir path/to/images/ \
                        --instruction "grasp the handle" \
                        --output_dir outputs/inference/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

from models.grasp_model import GraspCLIPModel
from utils.checkpoint import load_checkpoint
from utils.visualization import visualize_prediction


class GraspPredictor:
    """Inference wrapper for GraspCLIP model."""

    def __init__(
        self,
        checkpoint_path: str,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        grasp_head_hidden: int = 512,
        device: str = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = GraspCLIPModel(
            clip_model_name=clip_model_name,
            grasp_head_hidden=grasp_head_hidden,
        ).to(self.device)

        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    def model_info(self) -> dict:
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        clip_params = sum(p.numel() for p in self.model.clip.parameters())
        head_params = total - clip_params
        return {
            "total_params": total,
            "trainable_params": trainable,
            "clip_params": clip_params,
            "head_params": head_params,
            "total_M": total / 1e6,
            "clip_M": clip_params / 1e6,
            "head_M": head_params / 1e6,
        }

    @torch.no_grad()
    def benchmark(self, num_runs: int = 100, warmup: int = 10) -> dict:
        """Measure inference latency."""
        import time
        dummy_image = Image.fromarray(np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8))
        dummy_text = "grasp the object"

        inputs = self.processor(
            text=[dummy_text], images=[dummy_image],
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Warmup
        for _ in range(warmup):
            self.model(pixel_values, input_ids, attention_mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.model(pixel_values, input_ids, attention_mask)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times_ms = [t * 1000 for t in times]
        return {
            "mean_ms": np.mean(times_ms),
            "std_ms": np.std(times_ms),
            "min_ms": np.min(times_ms),
            "max_ms": np.max(times_ms),
            "fps": 1000.0 / np.mean(times_ms),
            "num_runs": num_runs,
            "device": str(self.device),
        }

    @torch.no_grad()
    def predict(self, image: Image.Image, instruction: str) -> dict:
        """
        Predict grasp pose for an image and instruction.

        Args:
            image: PIL Image (RGB)
            instruction: text grasp instruction

        Returns:
            dict with:
                - params: (x, y, w, h, theta) normalized
                - x, y: center coordinates [0, 1]
                - w, h: width, height [0, 1]
                - theta: angle in degrees
        """
        inputs = self.processor(
            text=[instruction],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        pixel_values = inputs["pixel_values"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        output = self.model(pixel_values, input_ids, attention_mask)
        params_deg = output["params_deg"][0].cpu().numpy()

        return {
            "params": params_deg,
            "x": float(params_deg[0]),
            "y": float(params_deg[1]),
            "w": float(params_deg[2]),
            "h": float(params_deg[3]),
            "theta": float(params_deg[4]),
        }

    def predict_and_visualize(
        self,
        image_path: str,
        instruction: str,
        save_path: str = None,
    ) -> tuple:
        """Predict and draw grasp on image. Returns (result_dict, visualization)."""
        image = Image.open(image_path).convert("RGB")
        result = self.predict(image, instruction)

        img_np = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR for cv2
        vis = visualize_prediction(
            img_np,
            pred_params=tuple(result["params"]),
            instruction=instruction,
            save_path=save_path,
        )

        return result, vis


def main():
    parser = argparse.ArgumentParser(description="GraspCLIP Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    predictor = GraspPredictor(
        checkpoint_path=args.checkpoint,
        clip_model_name=args.clip_model,
        device=args.device,
    )
    print(f"Model loaded on {predictor.device}")

    # Model info
    info = predictor.model_info()
    print(f"\nModel parameters:")
    print(f"  Total:  {info['total_M']:.2f}M")
    print(f"  CLIP:   {info['clip_M']:.2f}M")
    print(f"  Head:   {info['head_M']:.2f}M")

    # Benchmark
    print(f"\nBenchmarking inference speed...")
    bench = predictor.benchmark(num_runs=50, warmup=5)
    print(f"  Latency: {bench['mean_ms']:.1f} ± {bench['std_ms']:.1f} ms")
    print(f"  FPS:     {bench['fps']:.1f}")
    print(f"  Device:  {bench['device']}")

    if args.image:
        # Single image inference
        print(f"\nImage: {args.image}")
        print(f"Instruction: \"{args.instruction}\"")

        save_path = str(output_dir / f"result_{Path(args.image).stem}.jpg")
        result, vis = predictor.predict_and_visualize(
            args.image, args.instruction, save_path=save_path
        )

        print(f"\nPrediction:")
        print(f"  Center (x, y): ({result['x']:.4f}, {result['y']:.4f})")
        print(f"  Size (w, h):   ({result['w']:.4f}, {result['h']:.4f})")
        print(f"  Angle:         {result['theta']:.2f}°")
        print(f"\nVisualization saved to {save_path}")

    elif args.image_dir:
        # Batch inference
        image_dir = Path(args.image_dir)
        image_files = sorted(
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpeg"))
        )
        print(f"\nProcessing {len(image_files)} images...")

        results = []
        for img_path in image_files:
            save_path = str(output_dir / f"result_{img_path.stem}.jpg")
            result, _ = predictor.predict_and_visualize(
                str(img_path), args.instruction, save_path=save_path
            )
            results.append({"image": img_path.name, **result})
            print(f"  {img_path.name}: center=({result['x']:.3f}, {result['y']:.3f}), "
                  f"size=({result['w']:.3f}, {result['h']:.3f}), angle={result['theta']:.1f}°")

        # Save results summary
        import json
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_dir}/results.json")
    else:
        parser.error("Either --image or --image_dir must be specified")


if __name__ == "__main__":
    main()
