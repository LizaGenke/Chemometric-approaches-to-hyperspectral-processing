"""
Contains an implementation of the total loss function, which combines multiple losses:
- Out of bounds loss
- Smoothness loss
- L2 regularization loss
- Mean squared error loss

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from typing import Tuple

import torch

from .out_ouf_bounds_losses import OutOfBoundsLoss
from .regression_losses import MSELoss
from .regularization_losses import L2RegularizationLoss
from .smoothness_losses import SmoothnessLoss


def get_total_loss(
    device,
    average_batch_size,
    out_of_bounds_lambd,
    smoothness_lambd,
    l2_lambd,
    mse_lambd,
    return_individual_losses,
):
    def inner():
        return TotalLoss(
            device=device,
            average_batch_size=average_batch_size,
            out_of_bounds_lambd=out_of_bounds_lambd,
            smoothness_lambd=smoothness_lambd,
            l2_lambd=l2_lambd,
            mse_lambd=mse_lambd,
            return_individual_losses=return_individual_losses,
        )

    return inner


class TotalLoss:
    def __init__(
        self,
        device: torch.device,
        average_batch_size: float | None,
        out_of_bounds_lambd: float,
        smoothness_lambd: float,
        l2_lambd: float,
        mse_lambd: float,
        return_individual_losses: bool,
    ) -> None:
        self.oobl = OutOfBoundsLoss(
            lambd=out_of_bounds_lambd,
            avg_batch_size=average_batch_size,
            device=device,
        )
        self.sl = SmoothnessLoss(
            lambd=smoothness_lambd,
            avg_batch_size=average_batch_size,
            device=device,
        )
        self.l2l = L2RegularizationLoss(
            lambd=l2_lambd,
            bias=False,
            device=device,
        )
        self.msel = MSELoss(
            lambd=mse_lambd,
            avg_batch_size=average_batch_size,
            device=device,
        )
        self.return_individual_losses = return_individual_losses
        self.all_losses = [self.oobl, self.sl, self.l2l, self.msel]
        self._init_values()

    def __call__(
        self,
        model: torch.nn.Module,
        y_true: Tuple[torch.Tensor, torch.Tensor],
        y_pred: torch.Tensor,
    ):
        mask, target = y_true
        masked_pred = y_pred * mask
        self.oobl(masked_pred)
        self.sl(mask, y_pred)
        self.l2l(model)
        self.msel(mask, target, masked_pred)

    def _init_values(self):
        pass

    def compute(self):
        for loss in self.all_losses:
            loss.compute()
        if self.return_individual_losses:
            self.value = [loss.value for loss in self.all_losses]
            self.name = [loss.__name__ for loss in self.all_losses]
        else:
            self.value = torch.sum(
                torch.stack([loss.value for loss in self.all_losses])
            )

    def reset(self):
        for loss in self.all_losses:
            loss.reset()
        self._init_values()
