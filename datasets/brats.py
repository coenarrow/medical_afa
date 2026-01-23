"""
BraTS (Brain Tumor Segmentation) dataset.

Thin wrapper around the medical base classes with BraTS-specific defaults.
"""
from .medical_base import (
    MedicalSliceDataset,
    MedicalClsDataset,
    MedicalClsValDataset,
    robust_read_npy,
    load_slice_list,
    load_cls_labels_from_filenames,
)

# Default slice split for BraTS dataset
BRATS_SLICE_SPLIT = [15500, 27435]


class BRATSDataset(MedicalSliceDataset):
    """BraTS medical imaging dataset."""
    def __init__(self, root_dir=None, split='train', stage='train',
                 slice_split=BRATS_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


class BRATSClsDataset(MedicalClsDataset):
    """BraTS classification dataset for training."""
    def __init__(self, root_dir=None, split='train', stage='train',
                 slice_split=BRATS_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


class BRATSClsValDataset(MedicalClsValDataset):
    """BraTS validation dataset."""
    def __init__(self, root_dir=None, split='val', stage='val',
                 slice_split=BRATS_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


# Backwards compatibility aliases
load_brats_slice_list = load_slice_list
load_brats_cls_labels = load_cls_labels_from_filenames
