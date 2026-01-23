"""
LASC (Left Atrial Scar Classification) dataset.

Thin wrapper around the medical base classes with LASC-specific defaults.
"""
from .medical_base import (
    MedicalSliceDataset,
    MedicalClsDataset,
    MedicalClsValDataset,
    robust_read_npy,
    load_slice_list,
    load_cls_labels_from_filenames,
)

# Default slice split for LASC dataset
LASC_SLICE_SPLIT = [4400, 4884]


class LASCDataset(MedicalSliceDataset):
    """LASC medical imaging dataset."""
    def __init__(self, root_dir=None, split='train', stage='train',
                 slice_split=LASC_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


class LASCClsDataset(MedicalClsDataset):
    """LASC classification dataset for training."""
    def __init__(self, root_dir=None, split='train', stage='train',
                 slice_split=LASC_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


class LASCClsValDataset(MedicalClsValDataset):
    """LASC validation dataset."""
    def __init__(self, root_dir=None, split='val', stage='val',
                 slice_split=LASC_SLICE_SPLIT, **kwargs):
        super().__init__(root_dir, split, stage, slice_split, **kwargs)


# Backwards compatibility aliases
load_lasc_slice_list = load_slice_list
load_lasc_cls_labels = load_cls_labels_from_filenames
