# Grasp-Anything: Language-Driven Grasp Pose Prediction

CLIP-based model for predicting 5-DoF grasp rectangles from RGB images and natural language instructions.

**Task:** Given an image and a grasping prompt (e.g., "grasp the blue bottle"), predict a rectangle grasp pose `{x, y, w, h, θ}` where:
- `(x, y)` — center point of the grasp rectangle
- `(w, h)` — width and height of the rectangle
- `θ` — rotation angle with respect to the image plane

## Project Structure

```
grasp_anything/
├── configs/
│   └── default.yaml          # Training configuration
├── models/
│   ├── __init__.py
│   ├── grasp_model.py        # GraspCLIP model (CLIP backbone + fusion + head)
│   └── grasp_head.py         # Grasp prediction head (MLP → x, y, w, h, θ)
├── utils/
│   ├── __init__.py
│   ├── losses.py             # Grasp loss (Smooth L1 + angular loss)
│   ├── metrics.py            # IoU, angle accuracy, GraspMetrics
│   ├── checkpoint.py         # Save/load/manage checkpoints
│   ├── visualization.py      # Draw grasp rectangles on images
│   └── label_parser.py       # Parse raw .pt labels → normalized 5-DoF
├── download_subset.py        # Download 30K subset from HuggingFace
├── dataloader.py             # PyTorch Dataset & DataLoader
├── train.py                  # Full training pipeline
├── eval.py                   # Evaluation with metrics & visualization
├── inference.py              # Single-image inference
├── test.py                   # Unit tests & sanity checks
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Download

Download a 30,000-sample subset from Grasp-Anything + Grasp-Anything-pp:

```bash
python download_subset.py --num_samples 30000 --data_dir ./data
```

To skip the large image download (~65 GB) and only get instructions + labels:
```bash
python download_subset.py --num_samples 30000 --data_dir ./data --skip_images
```

## Training

```bash
python train.py --config configs/default.yaml
```

Resume from checkpoint:
```bash
python train.py --config configs/default.yaml --resume checkpoints/latest.pt
```

## Evaluation

```bash
python eval.py --checkpoint checkpoints/best_epoch050_0.8500.pt --visualize
```

## Inference

Single image:
```bash
python inference.py --checkpoint checkpoints/best.pt \
                    --image path/to/image.jpg \
                    --instruction "grasp the blue bottle"
```

Batch inference:
```bash
python inference.py --checkpoint checkpoints/best.pt \
                    --image_dir path/to/images/ \
                    --instruction "grasp the handle" \
                    --output_dir outputs/inference/
```

## Tests

```bash
python test.py              # Run all tests
python test.py --test model # Test model forward pass only
```

## Model Architecture

```
Image  ──→ [CLIP ViT-B/16] ──→ image_features (512-d)
                                                        ├──→ [Concat + Fusion MLP] ──→ [Grasp Head] ──→ (x, y, w, h, θ)
Text   ──→ [CLIP Text Enc] ──→ text_features (512-d)
```

- **Backbone:** CLIP ViT-B/16 (pretrained, optionally frozen for first N epochs)
- **Fusion:** Concatenation + linear projection
- **Head:** MLP predicting position (sigmoid), size (sigmoid), and angle (atan2 of sin/cos)
- **Loss:** Smooth L1 for position/size + MSE on sin/cos for angle

## Datasets

- [Grasp-Anything](https://huggingface.co/datasets/airvlab/Grasp-Anything) — RGB images
- [Grasp-Anything-pp](https://huggingface.co/datasets/airvlab/Grasp-Anything-pp) — Grasp instructions + labels

## Configuration

Edit `configs/default.yaml` to change:
- Model architecture (CLIP variant, head size, dropout)
- Training hyperparameters (lr, batch size, epochs, scheduler)
- Loss weights (xy, wh, angle)
- Evaluation thresholds (IoU, angle)

## Citation

```bibtex
@article{vuong2023grasp,
  title={Grasp-Anything: Large-scale Grasp Dataset from Foundation Models},
  author={Vuong, An Dinh and Vu, Minh Nhat and Le, Baoru and Huang, Jie and Huynh, Binh and Vo, Thieu and Recently, Andreas and Vu, Ngan and Nguyen, Anh},
  journal={arXiv preprint arXiv:2309.09818},
  year={2023}
}
```
