"""
Contains an implementation of a dataloader for preprocessed images.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from torch.utils.data import DataLoader

from ..datasets.preprocessed_image_dataset import PreprocessedImageDataset


def get_preprocessed_image_dataloader(
    reference_values,
    images_path,
    quant_biases_path,
    quant_scales_path,
    masks_path,
    csv_path,
    split,
    transform,
    batch_size,
    shuffle,
):
    dataset = PreprocessedImageDataset(
        reference_values=reference_values,
        images_path=images_path,
        quant_biases_path=quant_biases_path,
        quant_scales_path=quant_scales_path,
        masks_path=masks_path,
        csv_path=csv_path,
        split=split,
        split_column="split",
        transform=transform,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
    return dataloader
