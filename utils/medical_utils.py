"""
Medical imaging training utilities.

Shared utilities for training weakly-supervised segmentation models
on medical imaging datasets (LASC, BraTS, etc.).
"""

import datetime
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F


def get_device(local_rank, force_cpu=False):
    """Select best available device: CUDA > MPS > CPU"""
    if force_cpu:
        device = torch.device("cpu")
        print("Using CPU device (forced)")
    elif torch.cuda.is_available():
        device = torch.device(local_rank)
        torch.cuda.set_device(local_rank)
        print(f"Using CUDA device: {local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(filename='test.log'):
    """Configure logging to file and console."""
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    """Calculate elapsed time and ETA for training."""
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def get_down_size(ori_shape=(512, 512), stride=16):
    """Calculate downsampled size after backbone stride."""
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w


def get_seg_loss(pred, label, ignore_index=255):
    """Compute balanced foreground/background segmentation loss."""
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    """Generate local affinity mask based on radius."""
    hw = h * w
    mask = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1
    return mask
