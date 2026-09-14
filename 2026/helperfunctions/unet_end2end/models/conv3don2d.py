"""
Contains an implementation of a Module that performs 3D convolution on an image.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch
import torch.nn as nn

from .unet import Permute


class Conv3Don2D(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, normalization):
        super(Conv3Don2D, self).__init__()
        bias = normalization is None
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=(1, 2, 2),
            padding="valid",
            bias=bias,
        )
        self.num_filters = num_filters
        self.normalization = normalization
        self.norm_layer = self._get_norm_layer()
        self.out_channels = num_filters * (in_channels - kernel_size[0] + 1)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_norm_layer(self):
        if self.normalization == "bn":
            return nn.BatchNorm3d(self.num_filters, momentum=0.01)
        if self.normalization == "ln":
            return self._get_layer_norm()
        return nn.Identity()

    def _get_layer_norm(self):
        layers = [
            Permute((0, 2, 3, 4, 1)),
            nn.LayerNorm(self.num_filters),
            Permute((0, 4, 1, 2, 3)),
        ]
        return nn.Sequential(*layers)

    def _reshape3Dto4D(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, 1, x.size(-3), x.size(-2), x.size(-1))

    def _reshape4Dto3D(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self.out_channels, x.size(-2), x.size(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape3Dto4D(x)
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self._reshape4Dto3D(x)
        return x


if __name__ == "__main__":
    conv3d_model = Conv3Don2D(
        in_channels=124,
        num_filters=1,
        kernel_size=(7, 2, 2),
        normalization=None,
    )
    print(conv3d_model)

    num_params = sum(p.numel() for p in conv3d_model.parameters())
    print(f"Number of parameters: {num_params}")

    num_trainable_params = sum(
        p.numel() for p in conv3d_model.parameters() if p.requires_grad
    )
    print(f"Number of trainable parameters: {num_trainable_params}")
    pass
