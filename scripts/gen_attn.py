"""
Generate attention maps and CAMs for any supported dataset.

This script loads a trained checkpoint and generates:
1. PNG visualizations: Heatmaps overlaid on original images
2. NPY files: Raw CAM/attention arrays for further processing

Supported datasets: LASC, COCO, VOC (+ future BraTS, KiTS)

Usage:
    python scripts/gen_attn.py \
        --config configs/lasc_attn_reg.yaml \
        --checkpoint work_dir_lasc/checkpoints/best.pth \
        --output_dir work_dir_lasc/attention_maps \
        --cam_npy_dir work_dir_lasc/cam_npy
"""

import argparse
import os
import sys
sys.path.append(".")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from wetr.model_attn_aff import WeTr
from utils.camutils import multi_scale_cam


def get_args_parser():
    parser = argparse.ArgumentParser('Generate attention maps for trained models')
    parser.add_argument('--config', required=True, type=str, help='Path to config file')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to checkpoint (default: {work_dir}/checkpoints/best.pth)')
    parser.add_argument('--use_final', action='store_true',
                        help='Use final checkpoint instead of best')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory for PNG visualizations (default: {work_dir}/attention_maps)')
    parser.add_argument('--cam_npy_dir', default=None, type=str,
                        help='Directory for NPY files (default: {work_dir}/cam_npy)')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pooling', default='gmp', type=str)
    parser.add_argument('--scales', nargs='+', type=float, default=None,
                        help='Multi-scale factors (default: from config or [1.0, 0.5, 1.5])')
    parser.add_argument('--device', default='cpu', type=str, help='Device (cpu, cuda, mps)')
    parser.add_argument('--save_attn', action='store_true',
                        help='Also save raw attention maps from transformer')
    parser.add_argument('--attn_npy_dir', default=None, type=str,
                        help='Directory for attention NPY files (default: {work_dir}/attn_npy)')
    return parser


def get_dataset(cfg, split='val'):
    """
    Factory function to create dataset based on config.

    Infers dataset type from config's root_dir path.
    Returns a dataset suitable for attention map generation.
    """
    root_dir = cfg.dataset.root_dir.lower()

    if 'lasc' in root_dir or 'brats' in root_dir or 'kits' in root_dir:
        # Medical imaging datasets (LASC, BraTS, KiTS)
        # These use index-based splitting and .npy files
        from datasets import lasc
        return lasc.LASCClsValDataset(
            root_dir=cfg.dataset.root_dir,
            split=split,
            stage='val',
            slice_split=cfg.dataset.slice_split,
            aug=False,
            ignore_index=cfg.dataset.ignore_index,
            num_classes=cfg.dataset.num_classes,
        )
    elif 'coco' in root_dir or 'mscoco' in root_dir:
        # COCO dataset
        from datasets import coco
        return coco.CocoSegDataset(
            root_dir=cfg.dataset.root_dir,
            name_list_dir=cfg.dataset.name_list_dir,
            split=split,
            stage='val',
            aug=False,
            ignore_index=cfg.dataset.ignore_index,
            num_classes=cfg.dataset.num_classes,
        )
    elif 'voc' in root_dir:
        # VOC dataset
        from datasets import voc
        return voc.VOC12SegDataset(
            root_dir=cfg.dataset.root_dir,
            name_list_dir=cfg.dataset.name_list_dir,
            split=split,
            stage='val',
            aug=False,
            ignore_index=cfg.dataset.ignore_index,
            num_classes=cfg.dataset.num_classes,
        )
    else:
        raise ValueError(f"Unknown dataset type. Cannot infer from root_dir: {cfg.dataset.root_dir}")


def show_cam_on_image(img, mask, save_path):
    """Overlay CAM heatmap on original image and save."""
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)


def denormalize_image(img_tensor):
    """Convert normalized tensor back to RGB image (0-255)."""
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    # img_tensor: (C, H, W)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    img = img * std + mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@torch.no_grad()
def generate_attention_maps(data_loader, model, device, output_dir, cam_npy_dir,
                            scales=[1.0, 0.5, 1.5], save_attn=False, attn_npy_dir=None):
    """
    Generate and save attention maps for all samples.

    Args:
        data_loader: DataLoader for the dataset
        model: Trained WeTr model
        device: torch device
        output_dir: Directory for PNG visualizations
        cam_npy_dir: Directory for CAM NPY files
        scales: Multi-scale factors for CAM generation
        save_attn: Whether to save raw attention maps
        attn_npy_dir: Directory for attention NPY files
    """
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(cam_npy_dir).mkdir(parents=True, exist_ok=True)
    if save_attn and attn_npy_dir:
        Path(attn_npy_dir).mkdir(parents=True, exist_ok=True)

    model.eval()

    for data in tqdm(data_loader, desc="Generating attention maps"):
        name, inputs, labels, cls_label = data

        inputs = inputs.to(device)
        cls_label = cls_label.to(device)
        b, c, h, w = inputs.shape

        # Get original image for visualization
        orig_image = denormalize_image(inputs[0])

        # Forward pass to get attention maps
        cls_out, segs, attns, attn_pred = model(inputs)

        # Generate multi-scale CAMs
        cams = multi_scale_cam(model, inputs, scales)

        # Also generate flipped version and average
        inputs_flip = torch.flip(inputs, dims=[-1])
        cams_flip = multi_scale_cam(model, inputs_flip, scales)
        cams_flip = torch.flip(cams_flip, dims=[-1])

        # Average original and flipped CAMs
        cams_avg = (cams + cams_flip) / 2

        # Resize CAMs to original image size
        cams_resized = F.interpolate(cams_avg, size=(h, w), mode='bilinear', align_corners=False)

        # Get prediction score (handle both binary and multi-class)
        if cls_out.shape[1] == 1:
            pred_score = torch.sigmoid(cls_out[0, 0]).item()
        else:
            pred_score = torch.sigmoid(cls_out[0]).max().item()

        # Process each sample in batch
        for b_idx in range(inputs.shape[0]):
            img_name = name[b_idx]

            # Get CAM for foreground class (index 0 in cams, which is class 1)
            # For multi-class, get the max CAM across all foreground classes
            if cams_resized.shape[1] == 1:
                cam = cams_resized[b_idx, 0].cpu().numpy()
            else:
                cam = cams_resized[b_idx].max(dim=0)[0].cpu().numpy()

            # Normalize CAM to [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-5:
                cam_norm = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam)

            # Save PNG visualization
            score_str = format(pred_score, '.3f')
            png_path = os.path.join(output_dir, f'{img_name}_{score_str}.png')
            show_cam_on_image(orig_image, cam_norm, png_path)

            # Save NPY file with CAM
            npy_path = os.path.join(cam_npy_dir, f'{img_name}.npy')
            np.save(npy_path, {
                'cam': cam_norm,
                'cam_raw': cam,
                'pred_score': pred_score,
                'cls_label': cls_label[b_idx].cpu().numpy()
            })

            # Optionally save raw attention maps from transformer
            if save_attn and attn_npy_dir:
                attn_npy_path = os.path.join(attn_npy_dir, f'{img_name}.npy')

                # Save the last attention map and affinity prediction
                attn_data = {
                    'attn_pred': attn_pred[b_idx].cpu().numpy(),
                    'num_attn_layers': len(attns),
                }
                # Save last few attention layers (to avoid huge files)
                for i, attn in enumerate(attns[-4:]):  # Last 4 layers
                    attn_data[f'attn_layer_{len(attns)-4+i}'] = attn[b_idx].cpu().numpy()

                np.save(attn_npy_path, attn_data)

    print(f"\nGeneration complete!")
    print(f"  PNG visualizations: {output_dir}")
    print(f"  CAM NPY files: {cam_npy_dir}")
    if save_attn and attn_npy_dir:
        print(f"  Attention NPY files: {attn_npy_dir}")


def main():
    args = get_args_parser().parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.use_final:
        # Find the final checkpoint (highest iteration number)
        ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir)
        ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('wetr_iter_') and f.endswith('.pth')]
        if not ckpts:
            raise ValueError(f"No checkpoints found in {ckpt_dir}")
        # Sort by iteration number and get the last one
        ckpts.sort(key=lambda x: int(x.replace('wetr_iter_', '').replace('.pth', '')))
        checkpoint_path = os.path.join(ckpt_dir, ckpts[-1])
    else:
        # Default to best.pth
        checkpoint_path = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, 'best.pth')

    # Determine output directories
    output_dir = args.output_dir or os.path.join(cfg.work_dir.dir, 'attention_maps')
    cam_npy_dir = args.cam_npy_dir or os.path.join(cfg.work_dir.dir, 'cam_npy')
    attn_npy_dir = args.attn_npy_dir or os.path.join(cfg.work_dir.dir, 'attn_npy')

    # Determine scales
    scales = args.scales
    if scales is None:
        scales = list(cfg.cam.scales) if hasattr(cfg.cam, 'scales') else [1.0, 0.5, 1.5]

    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Scales: {scales}")
    print(f"Output dir: {output_dir}")
    print(f"CAM NPY dir: {cam_npy_dir}")

    # Setup device
    device = torch.device(args.device)

    # Create dataset
    dataset = get_dataset(cfg, split='val')
    print(f"Dataset: {len(dataset)} samples")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False
    )

    # Create model
    model = WeTr(
        backbone=cfg.backbone.config,
        stride=cfg.backbone.stride,
        num_classes=cfg.dataset.num_classes,
        embedding_dim=256,
        pretrained=False,  # Don't load pretrained, we'll load checkpoint
        pooling=args.pooling,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle DDP wrapped checkpoints
    state_dict = checkpoint
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    # Generate attention maps
    generate_attention_maps(
        data_loader, model, device,
        output_dir=output_dir,
        cam_npy_dir=cam_npy_dir,
        scales=scales,
        save_attn=args.save_attn,
        attn_npy_dir=attn_npy_dir
    )


if __name__ == '__main__':
    main()
