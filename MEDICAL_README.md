# Goals

The intended modifications are to add more datasets to this existing repo, following the same general conventions.

# Architecture

## Shared Utilities

Medical imaging datasets share common functionality through two modules:

### `utils/medical_utils.py`
Training utilities used by all medical imaging training scripts:
- `get_device(local_rank, force_cpu)` - Select best device (CUDA > MPS > CPU)
- `setup_seed(seed)` - Set random seeds for reproducibility
- `setup_logger(filename)` - Configure logging to file and console
- `cal_eta(time0, cur_iter, total_iter)` - Calculate elapsed time and ETA
- `get_down_size(ori_shape, stride)` - Calculate downsampled size after backbone
- `get_seg_loss(pred, label, ignore_index)` - Balanced segmentation loss
- `get_mask_by_radius(h, w, radius)` - Local affinity mask generation

### `datasets/medical_base.py`
Base classes and utilities for medical datasets:
- `robust_read_npy(file_path)` - Load .npy file and convert to RGB
- `load_slice_list(root_dir, split, slice_split)` - Index-based dataset splitting
- `load_cls_labels_from_filenames(name_list)` - Extract labels from filenames
- `MedicalSliceDataset` - Base dataset class
- `MedicalClsDataset` - Training dataset with augmentation
- `MedicalClsValDataset` - Validation dataset

Dataset classes (LASC, BraTS) inherit from these base classes, requiring only ~40 lines each instead of ~230 lines.

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

torchrun --nproc_per_node=1 --master_port=29520 scripts/dist_train_lasc.py --config configs/lasc_attn_reg.yaml --pooling gmp --crop_size 320 --work_dir work_dir_lasc


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

**Note:** The generic `gen_attn.py` script supports all datasets (LASC, COCO, VOC, BraTS) by inferring the dataset type from the config's `root_dir` path.

## BraTS Dataset

The BraTS (Brain Tumor Segmentation) dataset is configured for binary classification (background + tumor).

**Dataset structure:**
- Location: `data/BraTS/imageSlice/`
- Format: `.npy` files with naming pattern `BRATS_{id}_{slice}_{label}.npy`
- Split: Index-based via `slice_split=[15500, 27435]`
  - Training: Files with indices 27435+
  - Validation: Files with indices 15500-27435

**Note:** Like LASC, BraTS uses weakly-supervised training with only image-level labels. No segmentation ground truth masks are used during training.

### Training on CPU (Mac)

```bash
source .venv/bin/activate

torchrun --nproc_per_node=1 --master_port=29522 \
  scripts/dist_train_brats.py \
  --config configs/brats_attn_reg.yaml \
  --pooling gmp \
  --crop_size 320 \
  --work_dir work_dir_brats \
  --backend gloo \
  --cpu
```

### Training on CUDA (GPU clusters)

```bash
source .venv/bin/activate

# Single GPU
torchrun --nproc_per_node=1 --master_port=29522 \
  scripts/dist_train_brats.py \
  --config configs/brats_attn_reg.yaml \
  --pooling gmp \
  --crop_size 320 \
  --work_dir work_dir_brats

# Multi-GPU (e.g., 4 GPUs)
torchrun --nproc_per_node=4 --master_port=29522 \
  scripts/dist_train_brats.py \
  --config configs/brats_attn_reg.yaml \
  --pooling gmp \
  --crop_size 320 \
  --work_dir work_dir_brats
```

### Configuration

Key parameters in `configs/brats_attn_reg.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_classes` | 2 | Binary: background + tumor |
| `slice_split` | [15500, 27435] | Index-based train/val split |
| `max_iters` | 10000 | Total training iterations |
| `cam_iters` | 1000 | When affinity learning starts |
| `eval_iters` | 1000 | Validation frequency |
| `samples_per_gpu` | 4 | Batch size per GPU |

### Outputs

Same as LASC - attention maps are automatically generated after training:
- `{work_dir}/checkpoints/best.pth`: Best model checkpoint
- `{work_dir}/attention_maps/`: PNG heatmap visualizations
- `{work_dir}/cam_npy/`: NPY files with CAM arrays and prediction scores

## Adding New Medical Datasets

To add a new medical imaging dataset:

1. **Create dataset class** in `datasets/{dataset_name}.py`:
   ```python
   from .medical_base import MedicalClsDataset, MedicalClsValDataset

   DATASET_SLICE_SPLIT = [val_start, train_start]

   class DatasetClsDataset(MedicalClsDataset):
       def __init__(self, slice_split=DATASET_SLICE_SPLIT, **kwargs):
           super().__init__(slice_split=slice_split, **kwargs)

   class DatasetClsValDataset(MedicalClsValDataset):
       def __init__(self, slice_split=DATASET_SLICE_SPLIT, **kwargs):
           super().__init__(slice_split=slice_split, **kwargs)
   ```

2. **Create config** in `configs/{dataset}_attn_reg.yaml` (copy from LASC/BraTS and modify)

3. **Create training script** in `scripts/dist_train_{dataset}.py` (copy from LASC/BraTS and modify imports)

4. **Update `gen_attn.py`** to recognize the new dataset's root_dir pattern
