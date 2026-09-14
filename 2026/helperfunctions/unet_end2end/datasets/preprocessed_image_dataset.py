"""
Contains an implementation of a dataset for preprocessed images. It expects the images
to be stored as quantized uint16 NumPy arrays. It also loads masks and reference values.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from .utils import dequantize_16_bit


class PreprocessedImageDataset(Dataset):
    """
    This is the dataset class that we used to load the preprocessed images
    (already cropped or padded to the input size of 2360 x 1272 pixels)
    """
    def __init__(
        self,
        reference_values,
        images_path,
        quant_biases_path,
        quant_scales_path,
        masks_path,
        csv_path,
        split,
        split_column,
        transform,
    ):
        self.reference_values = reference_values
        self.csv_path = Path(csv_path)
        self.split = split
        self.split_column = split_column
        if images_path is not None:
            self.images_path = Path(images_path)
            try:
                self.quant_scales_path = Path(quant_scales_path)
                self.quant_biases_path = Path(quant_biases_path)
            except TypeError:
                self.quant_scales_path = None
                self.quant_biases_path = None
        else:
            self.images_path = None
        if masks_path is not None:
            self.masks_path = Path(masks_path)
        else:
            self.masks_path = None
        self.transform = transform

        self.df = self._load_csv()
        if self.images_path is not None:
            self.image_file_names = self._load_file_names(self.images_path, "npy")
            if self.quant_scales_path is not None:
                self.bias_file_names = self._load_file_names(
                    self.quant_biases_path, "npy"
                )
                self.scale_file_names = self._load_file_names(
                    self.quant_scales_path, "npy"
                )
            else:
                self.bias_file_names = None
                self.scale_file_names = None
        if self.masks_path is not None:
            self.mask_file_names = self._load_file_names(self.masks_path, "npy")
            self.mask_files = self._load_mask_files()

    def __len__(self):
        if self.images_path is not None:
            return len(self.image_file_names)
        elif self.masks_path is not None:
            return len(self.mask_file_names)
        else:
            raise ValueError("Either images_path or masks_path must be provided.")

    def _load_csv(self):
        df = pd.read_csv(self.csv_path)
        # Use the split column to filter the dataframe. self.split is a list of integers
        df = df[df[self.split_column].isin(self.split)]
        df.set_index("meat_id", inplace=True)
        return df

    def _load_file_names(self, path: Path, file_extension: str):
        file_names = []
        for f in sorted(list(path.glob(f"*.{file_extension}"))):
            meat_id = int(f.stem.split("-")[0])
            if meat_id not in self.df.index.values:
                continue
            file_names.append(f)
        return file_names

    def _load_mask_files(self):
        # Masks are around 2 MB each, so we can load them all at once
        mask_files = {}
        for f in self.mask_file_names:
            key = f
            mask = np.load(f).astype(np.float32)
            mask_files[key] = mask
        return mask_files

    def __getitem__(self, idx):
        if self.images_path is not None:
            file_id = self.image_file_names[idx].stem
            meat_id = int(file_id.split("-")[0])
            img = np.load(self.image_file_names[idx])
            if self.quant_scales_path is not None:
                bias = np.load(self.bias_file_names[idx])
                scale = np.load(self.scale_file_names[idx])
                img = dequantize_16_bit(img, bias, scale)
            img = tv_tensors.Image(img)
        else:
            img = None

        if self.masks_path is not None:
            file_id = self.mask_file_names[idx].stem
            meat_id = int(file_id.split("-")[0])
            mask = self.mask_files[self.mask_file_names[idx]]
            mask = tv_tensors.Mask(mask)
        else:
            mask = None
        return_tuple = ()
        if img is not None:
            return_tuple += (img,)
        if mask is not None:
            return_tuple += (mask,)
        if self.transform is not None:
            if len(return_tuple) == 1:
                return_tuple = (self.transform(*return_tuple),)
            else:
                return_tuple = self.transform(*return_tuple)

        if self.reference_values is not None:
            refs = self.df.loc[meat_id, self.reference_values].values
            refs = torch.tensor(refs, dtype=torch.float32)
            if self.masks_path is not None:
                # Extract mask from the return tuple
                mask = return_tuple[-1]
                # Couple the reference values with the mask
                mask_refs_tuple = ((mask, refs),)
                return_tuple = return_tuple[:-1] + mask_refs_tuple
            else:
                return_tuple += (refs,)
        return_tuple += (file_id,)
        return return_tuple
