"""
Utility functions for image processing in hyperspectral datasets.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from typing import Tuple, Union

import numpy as np
import spectral as spy
import torch
import torch.nn.functional as F


def pad_with_average_spectrum(
    img: Union[spy.io.bilfile.BilFile, np.ndarray],
    output_height: int,
    output_width: int,
    is_mask: bool,
):
    if isinstance(img, spy.io.bilfile.BilFile):
        img = img.load()
    h, w = img.shape[:2]
    if h >= output_height and w >= output_width:
        return img
    if is_mask:
        img = np.expand_dims(img, axis=-1)
    left_column = img[:, 0:1, :]
    right_column = img[:, -1:, :]
    if is_mask:
        avg_spectrum = np.zeros(img.shape[-1], dtype=img.dtype)
        # print(f"Padding mask with avg_spectrum")
    else:
        # print(f"Padding image with avg_spectrum")
        avg_spectrum_left_column = np.mean(left_column, axis=(0, 1))
        avg_spectrum_right_column = np.mean(right_column, axis=(0, 1))
        avg_spectrum = (avg_spectrum_left_column + avg_spectrum_right_column) / 2

    if h < output_height:
        num_top_pad_rows = (output_height - img.shape[0]) // 2
        num_bottom_pad_rows = num_top_pad_rows + (output_height - img.shape[0]) % 2
    else:
        num_top_pad_rows = 0
        num_bottom_pad_rows = 0
        output_height = h

    top_pad = np.tile(avg_spectrum[None, None, :], (num_top_pad_rows, img.shape[1], 1))
    bottom_pad = np.tile(
        avg_spectrum[None, None, :], (num_bottom_pad_rows, img.shape[1], 1)
    )

    if w < output_width:
        num_left_pad_columns = (output_width - img.shape[1]) // 2
        num_right_pad_columns = num_left_pad_columns + (output_width - img.shape[1]) % 2

    left_pad = np.tile(
        avg_spectrum[None, None, :], (output_height, num_left_pad_columns, 1)
    )
    right_pad = np.tile(
        avg_spectrum[None, None, :], (output_height, num_right_pad_columns, 1)
    )

    img = np.concatenate([top_pad, img, bottom_pad], axis=0)
    img = np.concatenate([left_pad, img, right_pad], axis=1)
    if is_mask:
        img = img.squeeze(axis=-1)
    return img


def pad_with_random_spectrum(
    img: Union[spy.io.bilfile.BilFile, np.ndarray], output_height: int, is_mask: bool
):
    if isinstance(img, spy.io.bilfile.BilFile):
        img = img.load()
    if img.shape[0] >= output_height:
        return img
    if is_mask:
        img = np.expand_dims(img, axis=-1)
    left_column = img[:, 0:1, :]
    right_column = img[:, -1:, :]
    if is_mask:
        zero_spectrum = np.zeros(img.shape[-1], dtype=img.dtype)
        print(f"Padding mask with zero spectrum")
    else:
        print(f"Padding image with random spectrum")
        # stack the left and right columns
        stacked_columns = np.concatenate([left_column, right_column], axis=0).squeeze(
            axis=1
        )

    num_top_pad_rows = (output_height - img.shape[0]) // 2
    num_bottom_pad_rows = num_top_pad_rows + (output_height - img.shape[0]) % 2

    if is_mask:
        top_pad = np.tile(
            zero_spectrum[None, None, :], (num_top_pad_rows, img.shape[1], 1)
        )
        bottom_pad = np.tile(
            zero_spectrum[None, None, :], (num_bottom_pad_rows, img.shape[1], 1)
        )

    else:
        random_indices = np.random.randint(
            0,
            stacked_columns.shape[0],
            size=(num_top_pad_rows + num_bottom_pad_rows, img.shape[1]),
        )
        top_pad = stacked_columns[random_indices[:num_top_pad_rows]]
        bottom_pad = stacked_columns[random_indices[num_top_pad_rows:]]

    img = np.concatenate([top_pad, img, bottom_pad], axis=0)
    if is_mask:
        img = img.squeeze(axis=-1)
    return img


def bilinear_interpolation(
    img: torch.Tensor,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    return F.interpolate(
        img[None, ...],
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )


def discard_less_than_750nm_and_bin(img):
    assert img.shape[0] == 300
    img = img[176:, ...]
    # bin by taking the average of every two bands
    img = img.reshape(img.shape[0] // 2, 2, img.shape[1], img.shape[2]).mean(axis=1)
    return img


def discard_less_than_750nm(img):
    assert img.shape[0] == 300
    return img[177:, ...]


def convert_mask_to_8bit(mask):
    mask = mask.astype(np.uint8)
    return mask


def quantize_16_bit(arr):
    bias = arr.min()
    max_uint16 = 65535
    scale = max_uint16 / (arr.max() - bias)
    uint_16_arr = np.round((arr - bias) * scale)
    assert uint_16_arr.min() == 0
    try:
        assert uint_16_arr.max() == max_uint16
    except AssertionError:
        print(f"Max value is {uint_16_arr.max()}")
    uint_16_arr = uint_16_arr.astype(np.uint16)
    return uint_16_arr, bias, scale


def dequantize_16_bit(arr, bias, scale):
    assert arr.dtype == np.uint16
    assert bias.dtype == np.float32
    assert scale.dtype == np.float32
    return (arr / scale) + bias


def compute_crop_central_coordinates(
    crop_height: int,
    crop_width: int,
    central_crop_height: int,
    central_crop_width: int,
) -> Tuple[int, int, int, int]:
    crop_start_height_coordinate = (crop_height - central_crop_height) // 2
    crop_end_height_coordinate = crop_start_height_coordinate + central_crop_height
    crop_start_width_coordinate = (crop_width - central_crop_width) // 2
    crop_end_width_coordinate = crop_start_width_coordinate + central_crop_width
    return (
        crop_start_height_coordinate,
        crop_end_height_coordinate,
        crop_start_width_coordinate,
        crop_end_width_coordinate,
    )


def mask_image(
    img: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    img = img * mask
    return img


def load_image_crop_overlap_tile(
    img: Union[spy.io.bilfile.BilFile, np.ndarray],
    crop_start_height_coordinate: int,
    crop_end_height_coordinate: int,
    crop_start_width_coordinate: int,
    crop_end_width_coordinate: int,
    crop_expanded_height: int,
    crop_expanded_width: int,
) -> torch.Tensor:
    img_height = img.shape[0]
    img_width = img.shape[1]

    crop_width = crop_end_width_coordinate - crop_start_width_coordinate
    crop_height = crop_end_height_coordinate - crop_start_height_coordinate

    # How much the crop is expanded on each side (i.e., the extra pixels on each side of the original crop)
    expanded_left_size = (crop_expanded_width - crop_width) // 2
    expanded_right_size = expanded_left_size + (crop_expanded_width - crop_width) % 2
    expanded_top_size = (crop_expanded_height - crop_height) // 2
    expanded_bottom_size = expanded_top_size + (crop_expanded_height - crop_height) % 2

    # Get as much of the expanded crop as possible without going out of bounds
    calculated_crop_start_height_coordinate = (
        crop_start_height_coordinate - expanded_top_size
    )
    calculated_crop_end_height_coordinate = (
        crop_end_height_coordinate + expanded_bottom_size
    )
    calculated_crop_start_width_coordinate = (
        crop_start_width_coordinate - expanded_left_size
    )
    calculated_crop_end_width_coordinate = (
        crop_end_width_coordinate + expanded_right_size
    )
    actual_crop_start_height_coordinate = max(
        0, calculated_crop_start_height_coordinate
    )
    actual_crop_end_height_coordinate = min(
        img_height, calculated_crop_end_height_coordinate
    )
    actual_crop_start_width_coordinate = max(0, calculated_crop_start_width_coordinate)
    actual_crop_end_width_coordinate = min(
        img_width, calculated_crop_end_width_coordinate
    )

    # Get the expanded crop
    expanded_crop = img[
        actual_crop_start_height_coordinate:actual_crop_end_height_coordinate,
        actual_crop_start_width_coordinate:actual_crop_end_width_coordinate,
    ]

    if isinstance(expanded_crop, np.ndarray):
        expanded_crop = np.asarray(expanded_crop, dtype=np.float32)
        if len(expanded_crop.shape) == 2:
            expanded_crop = expanded_crop[None, ...]
        elif len(expanded_crop.shape) == 3:
            expanded_crop = np.moveaxis(expanded_crop, -1, 0)
        expanded_crop = torch.tensor(expanded_crop)

    # Check if the expanded crop needs mirroring
    top_mirror_size = (
        actual_crop_start_height_coordinate - calculated_crop_start_height_coordinate
    )
    bottom_mirror_size = (
        calculated_crop_end_height_coordinate - actual_crop_end_height_coordinate
    )
    left_mirror_size = (
        actual_crop_start_width_coordinate - calculated_crop_start_width_coordinate
    )
    right_mirror_size = (
        calculated_crop_end_width_coordinate - actual_crop_end_width_coordinate
    )

    # Mirror the expanded crop if necessary
    if (
        top_mirror_size > 0
        or bottom_mirror_size > 0
        or left_mirror_size > 0
        or right_mirror_size > 0
    ):
        expanded_crop = F.pad(
            expanded_crop,
            (left_mirror_size, right_mirror_size, top_mirror_size, bottom_mirror_size),
            mode="reflect",
        )

    return expanded_crop
