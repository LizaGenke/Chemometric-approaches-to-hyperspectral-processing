"""
Contains an implementation of the U-Net architecture.
U-Net paper by Ronneberger et al. (2015): https://arxiv.org/abs/1505.04597

This implementation is based on the original U-Net architecture, with options for
normalization (batch normalization or layer normalization), bilinear upsampling,
and padding in the convolution layers.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_channels: int, out_channels: int, bias: bool, pad: bool) -> nn.Conv2d:
    """
    Applies a convolution with a 3x3 kernel.
    """
    if pad:
        padding = 1
    else:
        padding = "valid"
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=padding,
        bias=bias,
    )
    return layer


def conv_block(
    in_channels: int,
    out_channels: int,
    non_linearity: nn.Module,
    normalization: None | str,
    bias: bool,
    pad: bool,
) -> nn.Sequential:
    """
    A block of two convolutional layers, each followed by a non-linearity
    and optionally a normalization layer.

    In the U-Net architecture illustration in the U-Net paper,
    this corresponds to two blue arrows.
    """
    layers = []
    for _ in range(2):
        layers.append(
            conv3x3(
                in_channels=in_channels, out_channels=out_channels, bias=bias, pad=pad
            )
        )
        layers.append(non_linearity)
        layers.append(
            get_norm_layer(normalization=normalization, in_channels=out_channels)
        )
        in_channels = out_channels
    return nn.Sequential(*layers)


def batch_norm(in_channels: int) -> nn.Sequential:
    """
    Apply Batch Normalization over the channel dimension.
    Batch Normalization paper by Ioffe and Szegedy (2015): https://arxiv.org/abs/1502.03167
    """
    return nn.BatchNorm2d(in_channels, momentum=0.01)


class Permute(nn.Module):
    """
    Permute the dimensions of a tensor.
    """

    def __init__(self, dims: Iterable[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(map(str, self.dims))})'


def layer_norm(in_channels: int) -> nn.Sequential:
    """
    Apply Layer Normalization over the channel dimension.
    Layer Normalization paper by Ba et al. (2016): https://arxiv.org/abs/1607.06450
    """
    layers = [
        # (B, C, H, W) -> (B, H, W, C)
        Permute((0, 2, 3, 1)),
        # LayerNorm expects the last dimension to be the feature dimension
        # (we want the normalized shape to be (C,))
        nn.LayerNorm(in_channels),
        # (B, H, W, C) -> (B, C, H, W)
        Permute((0, 3, 1, 2)),
    ]
    return nn.Sequential(*layers)


def get_norm_layer(normalization: None | str, in_channels: int) -> nn.Module:
    """
    Get the normalization layer based on the specified type.
    Either 'bn' for batch normalization, 'ln' for layer normalization,
    or None for no normalization layer.
    """
    if normalization == "bn":
        return batch_norm(in_channels)
    if normalization == "ln":
        return layer_norm(in_channels)
    return nn.Identity()


def copy_and_crop(large: torch.Tensor, small: torch.Tensor) -> torch.Tensor:
    """
    Implementation of a copy-and-crop block in the U-Net architecture.
    Copy the large image and crop it to the size of the small image.
    The large image is cropped in the middle, and then the two images are
    concatenated along the channel dimension.

    In the U-Net architecture illustration in the U-Net paper,
    this corresponds to a gray arrow.
    """
    large_height, large_width = large.shape[-2:]
    small_height, small_width = small.shape[-2:]
    start_x = (large_height - small_height) // 2
    start_y = (large_width - small_width) // 2
    cropped_large = large[
        ..., start_x : start_x + small_height, start_y : start_y + small_width
    ]
    return torch.cat([cropped_large, small], dim=-3)


class ContractionBlock(nn.Module):
    """
    Implementation of a contraction block in the U-Net architecture.
    This block consists of a max pooling layer followed by a convolution block.

    In the U-Net architecture illustration in the U-Net paper, this corresponds to
    one red arrow followed by the subsequent two blue arrows.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity: nn.Module,
        nonormalization: None | str,
        bias: bool,
        pad: bool,
    ):
        super().__init__()
        self.max_pool = self._max_pool()
        self.conv_block = conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            non_linearity=non_linearity,
            normalization=nonormalization,
            bias=bias,
            pad=pad,
        )

    def _max_pool(self) -> nn.MaxPool2d:
        layer = nn.MaxPool2d(kernel_size=2, stride=2)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.max_pool(x)
        x = self.conv_block(x)
        return x


class Upsample(nn.Module):
    """
    Implementation of an upsampling block in the U-Net architecture.
    This block consists of either a transposed convolution or bilinear upsampling,
    followed by a convolution block.

    In the U-Net architecture illustration in the U-Net paper, this corresponds to
    one green arrow.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity,
        normalization: None | str,
        bias: bool,
        bilinear: bool,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.non_linearity = non_linearity
        self.normalization = normalization
        self.bias = bias
        self.bilinear = bilinear
        self.up = self._upsample(in_channels, out_channels)

    def _upsample(self, in_channels: int, out_channels: int) -> nn.Sequential:
        if self.bilinear:
            up = self._up_bilinear(in_channels, out_channels)
        else:
            up = self._up_trans_conv2x2(in_channels, out_channels)
        return up

    def _up_trans_conv2x2(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=self.bias
            ),
            self.non_linearity,
        ]
        layers.append(get_norm_layer(self.normalization, out_channels))
        return nn.Sequential(*layers)

    def _up_bilinear(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = [
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=True),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            self.non_linearity,
        ]
        layers.append(get_norm_layer(self.normalization, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class ExpansionBlock(nn.Module):
    """
    Implementation of an expansion block in the U-Net architecture.
    This block consists of an upsampling block followed by a copy-and-crop block and
    a convolution block.

    In the U-Net architecture illustration in the U-Net paper, this corresponds to
    one green arrow followed by a gray arrow and then two blue arrows.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        non_linearity: nn.Module,
        normalization: None | str,
        bias: bool,
        bilinear: bool,
        pad: bool,
    ):
        super().__init__()
        self.pad = pad
        self.upsample = Upsample(
            in_channels=in_channels,
            out_channels=out_channels,
            non_linearity=non_linearity,
            normalization=normalization,
            bias=bias,
            bilinear=bilinear,
        )
        self.conv_block = self.conv_block = conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            non_linearity=non_linearity,
            normalization=normalization,
            bias=bias,
            pad=pad,
        )

    def forward(self, large: torch.Tensor, small: torch.Tensor) -> torch.Tensor:
        x = self.upsample(small)
        if self.pad:
            diff_h = large.shape[-2] - x.shape[-2]
            diff_w = large.shape[-1] - x.shape[-1]
            pad_left = diff_w // 2
            pad_right = diff_w - pad_left
            pad_top = diff_h // 2
            pad_bottom = diff_h - pad_top
            x = F.pad(
                x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
            )
        x = copy_and_crop(large, x)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """
    in_channels: Number of input channels
    out_channels: Number of output channels
    normalization: None | str, if 'bn' use batch normalization, if 'ln' use layer normalization
    bilinear: bool, if True use bilinear upsampling, if False use transposed convolution
    pad: bool, if True use padding in the convolution layers, preserving the input size
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pad: bool,
        bilinear: bool,
        normalization: None | str,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad = pad
        self.bilinear = bilinear
        self.normalization = normalization
        if self.normalization not in [None, "bn", "ln"]:
            raise ValueError(
                "Normalization must be None, 'bn' for batch normalization,"
                "or 'ln' for layer normalization"
            )
        # Whether to use bias in the convolution layers
        # If normalization is used, bias is already included in the normalization layer
        self.bias_conv = normalization is None
        self.non_linearity = nn.ReLU(inplace=True)
        self.intermediate_channels = [64 * 2**i for i in range(5)]
        self.first_convs = conv_block(
            in_channels=in_channels,
            out_channels=self.intermediate_channels[0],
            non_linearity=self.non_linearity,
            normalization=self.normalization,
            bias=self.bias_conv,
            pad=self.pad,
        )
        self.last_conv = nn.Conv2d(
            self.intermediate_channels[0], out_channels, kernel_size=1
        )
        self.contraction1 = self._get_contraction_block(
            in_channels=self.intermediate_channels[0],
            out_channels=self.intermediate_channels[1],
        )
        self.contraction2 = self._get_contraction_block(
            in_channels=self.intermediate_channels[1],
            out_channels=self.intermediate_channels[2],
        )
        self.contraction3 = self._get_contraction_block(
            in_channels=self.intermediate_channels[2],
            out_channels=self.intermediate_channels[3],
        )
        self.contraction4 = self._get_contraction_block(
            in_channels=self.intermediate_channels[3],
            out_channels=self.intermediate_channels[4],
        )
        self.expansion4 = self._get_expansion_block(
            in_channels=self.intermediate_channels[4],
            out_channels=self.intermediate_channels[3],
        )
        self.expansion3 = self._get_expansion_block(
            in_channels=self.intermediate_channels[3],
            out_channels=self.intermediate_channels[2],
        )
        self.expansion2 = self._get_expansion_block(
            in_channels=self.intermediate_channels[2],
            out_channels=self.intermediate_channels[1],
        )
        self.expansion1 = self._get_expansion_block(
            in_channels=self.intermediate_channels[1],
            out_channels=self.intermediate_channels[0],
        )

        # Init weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_contraction_block(
        self, in_channels: int, out_channels: int
    ) -> ContractionBlock:
        return ContractionBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            non_linearity=self.non_linearity,
            nonormalization=self.normalization,
            bias=self.bias_conv,
            pad=self.pad,
        )

    def _get_expansion_block(
        self, in_channels: int, out_channels: int
    ) -> ExpansionBlock:
        return ExpansionBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            non_linearity=self.non_linearity,
            normalization=self.normalization,
            bias=self.bias_conv,
            bilinear=self.bilinear,
            pad=self.pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.first_convs(x)
        x2 = self.contraction1(x1)
        x3 = self.contraction2(x2)
        x4 = self.contraction3(x3)
        x5 = self.contraction4(x4)
        x = self.expansion4(x4, x5)
        x = self.expansion3(x3, x)
        x = self.expansion2(x2, x)
        x = self.expansion1(x1, x)
        x = self.last_conv(x)
        return x