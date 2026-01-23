"""
KITS (Kidney Tumor Segmentation) dataset.

Thin wrapper around the medical base classes with KITS-specific defaults.
"""
from .medical_base import (
    MedicalSliceDataset,
    MedicalClsDataset,
    MedicalClsValDataset,
    robust_read_npy,
    load_slice_list,
    load_cls_labels_from_filenames,
)

# Default slice split for KITS dataset
KITS_SLICE_SPLIT = [8494, 15054]


class KITSDataset(MedicalSliceDataset):
    """BraTS medical imaging dataset."""
    def __init__(self, root_dir=None, split='train', stage='train',
                 slice_split=KITS_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


class KITSClsDataset(MedicalClsDataset):
    """BraTS classification dataset for training."""
    def __init__(self, root_dir=None, split='train', stage='train',
                 slice_split=KITS_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


class KITSClsValDataset(MedicalClsValDataset):
    """KITS validation dataset."""
    def __init__(self, root_dir=None, split='val', stage='val',
                 slice_split=KITS_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


# Backwards compatibility aliases
load_kits_slice_list = load_slice_list
load_kits_cls_labels = load_cls_labels_from_filenames
