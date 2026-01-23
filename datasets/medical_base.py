"""
Base classes for medical imaging datasets.

Provides common functionality for loading medical imaging slices
stored as .npy files with labels encoded in filenames.
"""

import os
import numpy as np
import PIL.Image
from torch.utils.data import Dataset
from . import transforms


def robust_read_npy(file_path):
    """
    Load .npy file and convert to RGB image array.
    Handles grayscale by stacking to 3 channels.
    """
    img_array = np.load(file_path)
    image = np.stack([img_array] * 3, axis=-1)
    return image


def load_slice_list(root_dir, split='train', slice_split=[0, 0]):
    """
    Load slice names based on split type using index-based splitting.

    Args:
        root_dir: Path to imageSlice directory containing .npy files
        split: 'train', 'val', or 'test'
        slice_split: [val_start, val_end] indices for splitting

    Returns:
        List of slice filenames (with .npy extension)
    """
    slice_names = sorted([f for f in os.listdir(root_dir) if f.endswith('.npy')])

    if split == 'train':
        return slice_names[slice_split[-1]:]  # After validation set
    elif split == 'val':
        return slice_names[slice_split[0]:slice_split[-1]]  # Validation range
    elif split == 'test':
        return slice_names[:slice_split[-1]]  # All up to validation end
    else:
        raise ValueError(f"Unknown split: {split}")


def load_cls_labels_from_filenames(name_list):
    """
    Extract classification labels from filenames.
    Filename format: {prefix}_{id}_{slice}_{label}.npy where label is 0 or 1

    Returns:
        dict: {slice_name_without_ext: one_hot_label_array}
    """
    cls_labels = {}
    for name in name_list:
        # Parse label from filename: PREFIX_XXX_YY_Z.npy -> Z is label
        label = int(name.split('_')[-1].split('.')[0])
        name_key = name.replace('.npy', '')
        # One-hot for single foreground class (binary: 0 or 1)
        cls_labels[name_key] = np.array([label], dtype=np.uint8)
    return cls_labels


class MedicalSliceDataset(Dataset):
    """
    Base dataset class for medical imaging datasets.
    Loads .npy files and extracts labels from filenames.
    Supports in-memory caching to avoid repeated disk I/O.
    """
    def __init__(
        self,
        root_dir=None,
        split='train',
        stage='train',
        slice_split=[0, 0],
        use_cache=True,
        **kwargs
    ):
        super().__init__()
        self.root_dir = root_dir
        self.stage = stage
        self.slice_split = slice_split
        self.name_list = load_slice_list(root_dir, split, slice_split)
        # In-memory cache for raw images to avoid repeated disk I/O
        self._cache = {} if use_cache else None

    def __len__(self):
        return len(self.name_list)

    def _load_image(self, idx):
        """Load image from cache or disk."""
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        slice_name = self.name_list[idx]
        slice_path = os.path.join(self.root_dir, slice_name)
        image = robust_read_npy(slice_path)
        _img_name = slice_name.replace('.npy', '')

        if self._cache is not None:
            self._cache[idx] = (_img_name, image)

        return _img_name, image

    def __getitem__(self, idx):
        _img_name, image = self._load_image(idx)

        # Medical datasets typically have no segmentation labels
        if self.stage == "train" or self.stage == "val":
            label = np.zeros(image.shape[:2], dtype=np.uint8)
        else:  # test
            label = image[:, :, 0]

        return _img_name, image, label


class MedicalClsDataset(MedicalSliceDataset):
    """
    Medical classification dataset for training.
    Returns image-level classification labels extracted from filenames.
    """
    def __init__(self,
                 root_dir=None,
                 split='train',
                 stage='train',
                 slice_split=[0, 0],
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=2,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, split, stage, slice_split)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        # Build classification labels from filenames
        self.label_list = load_cls_labels_from_filenames(self.name_list)

    def __len__(self):
        return len(self.name_list)

    def _transforms(self, image):
        img_box = None
        if self.aug:
            if self.rescale_range:
                image = transforms.random_scaling(
                    image,
                    scale_range=self.rescale_range)

            if self.img_fliplr:
                image = transforms.random_fliplr(image)

            if self.crop_size:
                image, img_box = transforms.random_crop(
                    image,
                    crop_size=self.crop_size,
                    mean_rgb=[0, 0, 0],
                    ignore_index=self.ignore_index)

        image = transforms.normalize_img(image)
        # to CHW
        image = np.transpose(image, (2, 0, 1))

        return image, img_box

    def __getitem__(self, idx):
        # Use cached image loading from base class
        img_name, image = self._load_image(idx)

        # Apply transforms (includes random augmentation, so not cached)
        image, img_box = self._transforms(image=image.copy())

        cls_label = self.label_list[img_name]

        if self.aug:
            return img_name, image, cls_label, img_box
        else:
            return img_name, image, cls_label


class MedicalClsValDataset(MedicalSliceDataset):
    """
    Medical validation dataset.

    Returns dummy segmentation labels (all zeros) for compatibility with
    the training loop. Segmentation metrics will NOT be meaningful.
    Focus validation on classification metrics.
    """
    def __init__(self,
                 root_dir=None,
                 split='val',
                 stage='val',
                 slice_split=[0, 0],
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 num_classes=2,
                 **kwargs):

        super().__init__(root_dir, split, stage, slice_split)

        self.aug = aug
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        # Build classification labels from filenames
        self.label_list = load_cls_labels_from_filenames(self.name_list)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # Use cached image loading from base class
        img_name, image = self._load_image(idx)

        # Normalize without augmentation for validation
        image = transforms.normalize_img(image)
        # to CHW
        image = np.transpose(image, (2, 0, 1))

        # Dummy segmentation label (all background/zeros)
        h, w = image.shape[1], image.shape[2]
        label = np.zeros((h, w), dtype=np.int16)

        cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label
