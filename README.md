# Grasp-Anything: Language-Driven Grasp Pose Prediction

Multi-modal model for predicting 5-DoF grasp rectangles from RGB images and natural language instructions.

**Task:** Given an image and a grasping prompt (e.g., "grasp the blue bottle"), predict a rectangle grasp pose `{x, y, w, h, θ}` where:
- `(x, y)` — center point of the grasp rectangle (normalized [0, 1])
- `(w, h)` — width and height of the rectangle (normalized [0, 1])
- `θ` — rotation angle in degrees

## Project Structure

```
grasp_anything/
├── configs/
│   ├── default.yaml          # Full training configuration
│   └── quick_test.yaml       # Quick test (3 epochs)
├── models/
│   ├── __init__.py
│   ├── cnn.py                # CNN backbone (lightweight, depthwise separable)
│   ├── vit.py                # ViT backbone (patch-based transformer)
│   ├── nlp.py                # Text encoder (frozen BERT embeddings + attention pooling)
│   ├── grasp_detection.py    # Full grasp detection model (CNN + ViT + NLP + heads)
│   ├── grasp_model.py        # (legacy) CLIP-based model
│   └── grasp_head.py         # (legacy) Grasp prediction head
├── utils/
│   ├── __init__.py
│   ├── losses.py             # Multi-grasp loss (min-over-targets)
│   ├── metrics.py            # IoU, angle accuracy (match-any-GT)
│   ├── checkpoint.py         # Save/load/manage checkpoints
│   ├── visualization.py      # Draw grasp rectangles on images
│   └── label_parser.py       # Parse raw .pt labels → all grasps [N, 5]
├── download_subset.py        # Download 30K subset from HuggingFace
├── dataloader.py             # PyTorch Dataset & DataLoader (70/15/15 split)
├── train.py                  # Training with val + test evaluation
├── eval.py                   # Evaluation with metrics & visualization
├── inference.py              # Inference + speed benchmark
├── visualize.py              # Batch grid visualization
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

Full training:
```bash
python train.py --config configs/default.yaml
```

Quick test (validate pipeline works, ~5 min):
```bash
python train.py --config configs/quick_test.yaml
```

Resume from checkpoint:
```bash
python train.py --config configs/default.yaml --resume checkpoints/latest.pt
```

Training behavior:
- Every epoch: train + validate, print metrics for both
- Every N epochs: save periodic checkpoint (`epoch_005.pt`, `epoch_010.pt`, ...)
- When val accuracy improves: save best checkpoint + run test set immediately
- End of training: load best checkpoint, run final test, report results

## Data Split

| Split | Ratio | ~Samples (30K) | Purpose |
|-------|-------|----------------|---------|
| Train | 70%   | 21,000         | Model training |
| Val   | 15%   | 4,500          | Model selection, hyperparameter tuning |
| Test  | 15%   | 4,500          | Final evaluation, reported results |

## Evaluation

Evaluate on validation set:
```bash
python eval.py --checkpoint checkpoints/best.pt --split val --visualize
```

Evaluate on test set (final results):
```bash
python eval.py --checkpoint checkpoints/best.pt --split test --visualize
```

## Visualization

Generate a batch grid image showing predictions (red) vs ground truth (green) with instructions:

```bash
python visualize.py --checkpoint checkpoints/best.pt --split val --num_samples 16 --cols 4
python visualize.py --checkpoint checkpoints/best.pt --image_dir ./my_images \
                    --instruction "grasp the handle" --num_samples 8
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
Image (3x416x416) ──┬── CNN Backbone ──► A (d_model) ──┐
                    │                                   │ Cross Attention(A, C)
                    └── ViT Backbone ──► B (d_model)    │         │
                                              │         │         ▼
Text ──► BERT Embeddings ──┬── C (seq, 768) ──┘   E (d_model) ──► MLP ──► center (x, y)
         + Pos Encoding    │                            │
                           └── D (d_model)              │
                                     │                  │
                         B ──► MLP ──┤                  │
                                     + ──► F (d_model)  │
                         D ──► MLP ──┘        │         │
                                              ▼         │
                              F ──► MLP ──► sin(θ/2)    │
                                                        │
                                   [E, F] ──► MLP ──► size (w, h)
```

### Components

- **CNN Backbone** (`cnn.py`): Lightweight depthwise separable CNN with learnable positional encoding. ~410K params.
- **ViT Backbone** (`vit.py`): Vision Transformer with 169 patches (32x32), 4 layers, embed_dim=256. ~3.1M params.
- **Text Encoder** (`nlp.py`): Frozen BERT word embeddings + learnable positional encoding + attention pooling. ~1.1M trainable params.
- **Grasp Detection** (`grasp_detection.py`): Combines all backbones with cross-attention and prediction heads.

### Predictions

- `center (x, y)`: sigmoid → [0, 1]
- `size (w, h)`: sigmoid → [0, 1]
- `sin(θ/2)`: sigmoid → [0, 1] (θ in [0, 180deg])

### Loss Strategy

Min-over-targets: for each prediction, compute loss against all GT grasps and backprop only the minimum (best-matching GT).

### Evaluation Criteria

Prediction is correct if it matches ANY GT grasp: IoU >= 0.25 AND angle diff <= 30deg.

## Multi-Grasp Handling

Each sample may have multiple valid grasp poses. The pipeline handles this properly:

- **Training loss:** computes loss against each GT grasp, takes the minimum
- **Evaluation:** prediction is successful if it matches at least one GT grasp

## Data Format

**File naming convention:**
```
images/               abc123.jpg           <- 1 image
grasp_instructions/   abc123_1_1.pkl       <- multiple instructions per image
                      abc123_1_2.pkl
grasp_label_positive/ abc123_1_1.pt        <- 1 label per instruction
                      abc123_1_2.pt
```

Each sample ID (e.g., `abc123_1_1`) maps to one instruction + one label. The corresponding image is found by stripping the `_N_N` suffix.

**Label format:** `[N, 6]` tensor per sample
- Column 0: quality/confidence score (discarded)
- Columns 1-5: `(x, y, w, h, angle_deg)` in pixel coordinates

Parsed to: `[N, 5]` normalized `(x, y, w, h, theta_rad)` for training.

## Configuration

Edit `configs/default.yaml` to change:
- Model: `d_model`, `max_seq_len`, `bert_model`
- Training: lr, batch size, epochs, scheduler, warmup
- Loss weights: center, size, angle
- Data split ratios
- Evaluation thresholds (IoU, angle)
- Checkpoint saving frequency and top-k retention

## Datasets

- [Grasp-Anything](https://huggingface.co/datasets/airvlab/Grasp-Anything) — RGB images
- [Grasp-Anything-pp](https://huggingface.co/datasets/airvlab/Grasp-Anything-pp) — Grasp instructions + labels

## Citation

```bibtex
@article{vuong2023grasp,
  title={Grasp-Anything: Large-scale Grasp Dataset from Foundation Models},
  author={Vuong, An Dinh and Vu, Minh Nhat and Le, Baoru and Huang, Jie and Huynh, Binh and Vo, Thieu and Recently, Andreas and Vu, Ngan and Nguyen, Anh},
  journal={arXiv preprint arXiv:2309.09818},
  year={2023}
}
```
