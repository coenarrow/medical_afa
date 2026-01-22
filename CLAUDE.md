# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of the AFA (Affinity from Attention) repository - an implementation of the CVPR 2022 paper for end-to-end weakly-supervised semantic segmentation using only image-level labels. The original method uses transformer self-attention to learn semantic affinity for refining pseudo-labels. See `README.md` for details on the original implementation.

**Purpose of this fork:** Extend the original AFA implementation by testing these models on new datasets, particularly medical imaging datasets.

**Supported datasets:** PASCAL VOC 2012, MS COCO 2014, medical datasets (LASC - in development)

## Development Guidelines

**Do not modify the core model architecture.** The model in `wetr/` should remain unchanged unless explicitly required, and any modifications must have no substantial impact on the overall architecture.

**Follow existing conventions** when adding support for new datasets:
- Dataset loaders go in `./datasets/`
- Configuration files go in `./configs/`
- Training/testing scripts go in `./scripts/`

## Common Commands

### Training

```bash
# VOC dataset (distributed, 2 GPUs)
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
  scripts/dist_train_voc.py \
  --config configs/voc_attn_reg.yaml \
  --pooling gmp \
  --crop_size 512 \
  --work_dir work_dir_voc

# COCO dataset (distributed, 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29519 \
  scripts/dist_train_coco.py \
  --config configs/coco_attn_reg.yaml \
  --pooling gmp \
  --crop_size 512 \
  --work_dir work_dir_coco
```

### Testing

```bash
python scripts/test_msc_flip.py \
  --config configs/voc_attn_reg.yaml \
  --model_path ./work_dir_voc/checkpoints/wetr_iter_18000.pth \
  --work_dir results \
  --eval_set val
```

### Environment Setup

```bash
# Activate the virtual environment first
source ./.venv/bin/activate

uv sync
```

## Architecture

### Core Components

- **wetr/model_attn_aff.py** - Main WeTr model combining SegFormer encoder, decoder, classification head, and attention-to-affinity projection
- **wetr/mix_transformer.py** - SegFormer backbone (MIT-B1/B2/B3/B4 variants)
- **wetr/PAR.py** - Pixel-Adaptive Refinement module for local consistency using appearance-based affinity
- **utils/camutils.py** - CAM generation, affinity label creation, and pseudo-label refinement functions
- **utils/losses.py** - Loss functions: affinity loss, segmentation loss, energy loss

### Training Pipeline

1. Forward pass produces: classification logits, segmentation predictions, attention maps, affinity predictions
2. CAMs generated from classification outputs
3. After `cam_iters`: CAMs refined using predicted affinity via `propagte_aff_cam_with_bkg()`
4. Pseudo-labels computed with adaptive thresholds (`high_thre`, `low_thre`, `bkg_score`)
5. Multi-task loss: classification + segmentation + affinity + optional energy loss

### Configuration

YAML configs in `configs/` use OmegaConf. Key parameters:
- `backbone.config`: SegFormer variant (mit_b1, mit_b2, etc.)
- `train.cam_iters`: When affinity learning starts
- `cam.bkg_score`, `cam.high_thre`, `cam.low_thre`: Pseudo-label thresholds

### Dataset Structure

VOC expected at `../VOCdevkit/VOC2012/` with `SegmentationClassAug/` for augmented annotations.

COCO expected in VOC-style structure:
```
MSCOCO/
├── JPEGImages/{train,val}/
└── SegmentationClass/{train,val}/
```

### Pre-trained Weights

Download SegFormer ImageNet weights from [official repo](https://github.com/NVlabs/SegFormer) and place in `pretrained/` (e.g., `pretrained/mit_b1.pth`).

## Adding New Datasets

Follow existing patterns in `datasets/voc.py` and `datasets/coco.py`. Create:
1. Dataset class inheriting appropriate base
2. Config YAML in `configs/`
3. Training script in `scripts/` or modify existing
4. Split files in `datasets/{dataset_name}/`
