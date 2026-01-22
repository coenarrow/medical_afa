# Goals

The intended modifications are to add more datasets to this existing repo, following the same general conventions.

# Setup

We have modifed the environment now to use uv as the package manager. 

After uv is installed, run the following from this directory:

```
uv venv -p 3.9
uv sync
source .venv/bin/activate
```

# Downloading Weights

From the project root, run the following to download the weights from hugging face.
The weights specified by the original repo aren't available anymore.

```
bash ./pretrained/download_pretrained.sh
```

# Usage

## LASC Dataset

The LASC medical imaging dataset is configured for binary classification (background + foreground).

**Dataset structure:**
- Location: `data/LASC/imageSlice/`
- Format: `.npy` files with naming pattern `{name}_{label}.npy`
- Split: Index-based via `slice_split=[4400, 4884]`
  - Training: 1892 samples (indices 4884+)
  - Validation: 484 samples (indices 4400-4884)

**Note:** LASC has no segmentation ground truth masks. Validation focuses on classification metrics (F1, accuracy). Segmentation/mIoU metrics are computed against dummy labels and are not meaningful.

### Training on CPU (Mac)

```bash
source .venv/bin/activate

torchrun --nproc_per_node=1 --master_port=29520 \
  scripts/dist_train_lasc.py \
  --config configs/lasc_attn_reg.yaml \
  --pooling gmp \
  --crop_size 320 \
  --work_dir work_dir_lasc \
  --backend gloo \
  --cpu
```

### Training on CUDA (GPU clusters)

```bash
source .venv/bin/activate

# Single GPU
torchrun --nproc_per_node=1 --master_port=29520 \
  scripts/dist_train_lasc.py \
  --config configs/lasc_attn_reg.yaml \
  --pooling gmp \
  --crop_size 320 \
  --work_dir work_dir_lasc

# Multi-GPU (e.g., 4 GPUs)
torchrun --nproc_per_node=4 --master_port=29520 \
  scripts/dist_train_lasc.py \
  --config configs/lasc_attn_reg.yaml \
  --pooling gmp \
  --crop_size 320 \
  --work_dir work_dir_lasc
```

### Monitoring Training

TensorBoard logs are saved to `work_dir_lasc/tb_logger/`:

```bash
tensorboard --logdir work_dir_lasc/tb_logger
```

### Configuration

Key parameters in `configs/lasc_attn_reg.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_classes` | 2 | Binary: background + foreground |
| `slice_split` | [4400, 4884] | Index-based train/val split |
| `max_iters` | 10000 | Total training iterations |
| `cam_iters` | 1000 | When affinity learning starts |
| `eval_iters` | 1000 | Validation frequency |
| `samples_per_gpu` | 4 | Batch size per GPU |

### Attention Maps (Automatic)

Attention maps are **automatically generated** after training completes. The training script:
1. Tracks the best checkpoint based on classification score (saved as `checkpoints/best.pth`)
2. Generates attention maps using the best checkpoint when training finishes

**Automatic outputs:**
- `{work_dir}/checkpoints/best.pth`: Best model checkpoint
- `{work_dir}/attention_maps/`: PNG heatmap visualizations
- `{work_dir}/cam_npy/`: NPY files with CAM arrays and prediction scores

### Generating Attention Maps (Standalone)

To regenerate attention maps manually or use a different checkpoint:

```bash
source .venv/bin/activate

python scripts/gen_attn.py \
  --config configs/lasc_attn_reg.yaml \
  --checkpoint work_dir_lasc/checkpoints/best.pth \
  --output_dir work_dir_lasc/attention_maps \
  --cam_npy_dir work_dir_lasc/cam_npy \
  --device cpu
```

**Options:**
- `--checkpoint`: Path to checkpoint (default: `{work_dir}/checkpoints/best.pth`)
- `--use_final`: Use final iteration checkpoint instead of best
- `--scales 1.0 0.5 1.5`: Multi-scale factors for CAM generation (default: [1.0, 0.5, 1.5])
- `--save_attn`: Also save raw transformer attention maps
- `--attn_npy_dir`: Directory for attention NPY files (used with `--save_attn`)
- `--device`: cpu, cuda, or mps

**Outputs:**
- `attention_maps/`: PNG heatmap visualizations overlaid on original images
- `cam_npy/`: NPY files containing raw CAM arrays and prediction scores

**Note:** The generic `gen_attn.py` script supports all datasets (LASC, COCO, VOC, and future BraTS/KiTS) by inferring the dataset type from the config's `root_dir` path.
