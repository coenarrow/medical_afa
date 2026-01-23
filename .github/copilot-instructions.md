# Copilot Instructions for Medical AFA

## Project Overview

**Medical AFA** extends the original AFA (Affinity from Attention, CVPR 2022) implementation for weakly-supervised semantic segmentation to medical imaging datasets. The core innovation uses transformer self-attention to learn semantic affinity for refining pseudo-labels with only image-level class labels during training.

**Current scope:** VOC, COCO, and medical datasets (LASC, BraTS, KITS)
**Key constraint:** Core model architecture in `wetr/` should remain unchanged unless explicitly required.

## Architecture Overview

### Core Components

```
wetr/model_attn_aff.py → WeTr model combines:
├── mix_transformer.py (SegFormer encoder: MIT-B1/B2/B3/B4)
├── segformer_head.py (multi-scale decoder)
├── PAR.py (Pixel-Adaptive Refinement: local affinity propagation)
└── attn_proj (attention-to-affinity projection: 16→1 channels)

Training pipeline:
1. Classification head: Image-level label → CAM generation
2. Segmentation head: Decode features → pseudo-label prediction
3. Affinity head: Learn semantic affinity from attention maps
4. CAM refinement: Propagate CAM using predicted affinity (after cam_iters)
5. Multi-task loss: classification + segmentation + affinity
```

### Data Flow

- **Input:** Image → Encoder (4,8,16,32 stride features)
- **Classification:** Last stride features → Global pooling → Classification logits
- **Segmentation:** Multi-scale decoder from all features → Segmentation logits
- **Affinity:** 16 attention heads → Conv(16→1) → Affinity prediction
- **Labels:** CAM-based pseudo-labels → Refined via affinity propagation (PAR)

## Adding Medical Datasets

Medical datasets follow a **standardized pattern** to minimize code duplication. Use `datasets/medical_base.py` utilities:

### Three-step addition:

1. **Create dataset class** (`datasets/{dataset}.py`):
   ```python
   from .medical_base import MedicalClsDataset, MedicalClsValDataset
   
   DATASET_SLICE_SPLIT = [val_start, train_start]  # Index-based split
   
   class DatasetClsDataset(MedicalClsDataset):
       def __init__(self, slice_split=DATASET_SLICE_SPLIT, **kwargs):
           super().__init__(slice_split=slice_split, **kwargs)
   
   class DatasetClsValDataset(MedicalClsValDataset):
       def __init__(self, slice_split=DATASET_SLICE_SPLIT, **kwargs):
           super().__init__(slice_split=slice_split, **kwargs)
   ```

2. **Create config** (`configs/{dataset}_attn_reg.yaml`):
   - Copy from `lasc_attn_reg.yaml` or `brats_attn_reg.yaml`
   - Update: `root_dir`, `slice_split`, `num_classes`, hyperparameters

3. **Create training script** (`scripts/dist_train_{dataset}.py`):
   - Copy from `dist_train_lasc.py`
   - Replace: dataset imports and work_dir path

**Why this pattern:** `datasets/medical_base.py` handles ~230 lines (file I/O, normalization, augmentation). Medical datasets inherit these, requiring only ~40 lines each.

## Critical Data Formats

### Medical Dataset File Structure
- **Location:** `data/{DATASET}/imageSlice/`
- **Format:** `.npy` files (NumPy arrays)
- **Naming:** `{PREFIX}_{id}_{slice}_{label}.npy` where `label` is `0` (background) or `1` (foreground)
- **Splitting:** Index-based via `slice_split=[val_start, train_start]`
  - Training: indices ≥ `train_start`
  - Validation: indices in range `[val_start, train_start)`

### Traditional Dataset File Structure
- **VOC:** `../VOCdevkit/VOC2012/` with `SegmentationClassAug/` (augmented annotations)
- **COCO:** VOC-style layout: `MSCOCO/JPEGImages/{train,val}/` and `SegmentationClass/{train,val}/`

## Key Configuration Parameters

YAML configs use OmegaConf. Critical parameters:

| Parameter | Scope | Purpose |
|-----------|-------|---------|
| `backbone.config` | Model | SegFormer variant (mit_b1, mit_b2, mit_b3, mit_b4) |
| `train.cam_iters` | Training | When affinity refinement starts (before: use initial CAM) |
| `train.max_iters` | Training | Total training iterations |
| `cam.bkg_score`, `high_thre`, `low_thre` | Pseudo-label | CAM thresholds for pseudo-label generation |
| `dataset.slice_split` | Data | Index-based train/val split for medical datasets |
| `dataset.num_classes` | Data | Usually 2 for medical (bg + foreground) or higher for multi-class |

## Training Workflows

### Medical Datasets (Recommended)
```bash
# CPU training (e.g., Mac)
torchrun --nproc_per_node=1 --master_port=29520 \
  scripts/dist_train_lasc.py \
  --config configs/lasc_attn_reg.yaml \
  --pooling gmp --crop_size 320 --work_dir work_dir_lasc \
  --backend gloo --cpu

# Single GPU
torchrun --nproc_per_node=1 --master_port=29520 \
  scripts/dist_train_lasc.py --config configs/lasc_attn_reg.yaml \
  --pooling gmp --crop_size 320 --work_dir work_dir_lasc

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 --master_port=29520 \
  scripts/dist_train_lasc.py --config configs/lasc_attn_reg.yaml \
  --pooling gmp --crop_size 320 --work_dir work_dir_lasc
```

### Traditional Datasets (VOC/COCO)
```bash
# VOC (2 GPUs)
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
  scripts/dist_train_voc.py --config configs/voc_attn_reg.yaml \
  --pooling gmp --crop_size 512 --work_dir work_dir_voc

# COCO (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29519 \
  scripts/dist_train_coco.py --config configs/coco_attn_reg.yaml \
  --pooling gmp --crop_size 512 --work_dir work_dir_coco
```

### Attention Map Generation

Attention maps are **auto-generated** after training completes. To regenerate or use different checkpoint:

```bash
python scripts/gen_attn.py \
  --config configs/lasc_attn_reg.yaml \
  --checkpoint work_dir_lasc/checkpoints/best.pth \
  --output_dir work_dir_lasc/attention_maps \
  --cam_npy_dir work_dir_lasc/cam_npy \
  --device cpu
```

Script auto-detects dataset type from config's `root_dir` path (supports all datasets).

## Loss Functions & Pseudo-Labels

### Multi-task Loss Components

Located in `utils/losses.py` and training scripts:
- **Classification loss:** Cross-entropy on global classification labels
- **Affinity loss:** `get_aff_loss()` - balanced positive/negative pixel pairs
- **Segmentation loss:** `get_seg_loss()` - binary CE on pseudo-labels (bg vs fg loss averaged)
- **Energy loss:** `get_energy_loss()` (optional) - energy minimization within crop regions

### Pseudo-Label Pipeline

1. **CAM generation:** Classification logits → Class Activation Maps
2. **Thresholding:** Apply `bkg_score`, `high_thre`, `low_thre` thresholds
3. **Affinity refinement (after cam_iters):** `propagte_aff_cam_with_bkg()` propagates CAM using predicted affinity
4. **Local refinement:** PAR module incorporates low-level appearance for consistency
5. **Labels for training:** Refined pseudo-labels supervise segmentation and affinity heads

## Important Utilities

### `utils/medical_utils.py`
- `get_device()` - Device selection (CUDA > MPS > CPU)
- `get_down_size()` - Calculate feature map size after backbone
- `get_seg_loss()` - Balanced segmentation loss for medical data
- `get_mask_by_radius()` - Local affinity mask for PAR module

### `utils/camutils.py`
- `multi_scale_cam()` - Multi-scale CAM generation
- `cams_to_affinity_label()` - Create affinity labels from CAM pairs
- `propagte_aff_cam_with_bkg()` - Refine CAM via affinity propagation
- `refine_cams_with_bkg_v2()` - Incorporate background information

### `datasets/medical_base.py`
- `robust_read_npy()` - Load .npy and convert to RGB
- `load_slice_list()` - Index-based train/val/test splitting
- `load_cls_labels_from_filenames()` - Parse labels from filenames
- Data augmentation and normalization already implemented

## Conventions & Patterns

1. **Port selection:** Use distinct `--master_port` values per experiment (29501=VOC, 29519=COCO, 29520=LASC/BraTS, 29522=KITS)
2. **Work directories:** `work_dir_{dataset}/` stores checkpoints, predictions, logs, tensorboard events
3. **Checkpoint naming:** `wetr_iter_{N}.pth` for iteration-based checkpoints; `best.pth` for best validation checkpoint
4. **Batch size:** `samples_per_gpu` in config; actual batch_size = `samples_per_gpu × num_gpus`
5. **Iterative training:** Medical datasets use iteration-based training (`max_iters`), not epochs
6. **No segmentation masks for medical training:** Medical datasets use only image-level labels; ground truth masks only used for validation metrics

## Environment & Dependencies

**Package manager:** `uv` (modern alternative to pip/conda)

```bash
# Setup
uv venv -p 3.9
uv sync
source .venv/bin/activate

# Download pre-trained weights
bash pretrained/download_pretrained.sh
```

**Key dependencies:** PyTorch, torchvision, OmegaConf, tensorboard, numpy, pillow, scikit-image

## Debugging & Monitoring

- **TensorBoard:** `tensorboard --logdir work_dir_{dataset}/tb_logger/`
- **Logs:** Saved to `{work_dir}/logs/` with detailed iteration info
- **Device check:** Script auto-detects best device; override with `--cpu` flag
- **Distributed setup:** Uses `torch.distributed` (DDP); backend selectable (default: nccl, gloo for CPU)
