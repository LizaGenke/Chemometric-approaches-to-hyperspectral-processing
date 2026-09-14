"""
Contains an implementation of the out of bounds loss function.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch


def get_out_of_bounds_loss(
    lambd: float, avg_batch_size: None | float, device: torch.device
):
    def inner():
        return OutOfBoundsLoss(
            lambd=lambd, avg_batch_size=avg_batch_size, device=device
        )

    return inner


class OutOfBoundsLoss:
    def __init__(
        self, lambd: float, avg_batch_size: None | float, device: torch.device
    ) -> None:
        self.__name__ = "Out of Bounds"
        self.lambd = torch.tensor(lambd, dtype=torch.float32, device=device)
        self.avg_batch_size = avg_batch_size
        self.device = device
        self._init_values()

    def _init_values(self):
        self.below_0_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        self.above_100_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        self.samples_seen = torch.tensor(0.0, dtype=torch.int32, device=self.device)

    def __call__(self, masked_pred: torch.Tensor):
        # Compute the average of the values below 0 and above 100, inside the mask
        below_0_loss = torch.sum(torch.relu(-masked_pred) ** 2)
        above_100_loss = torch.sum(torch.relu(masked_pred - 100) ** 2)
        self.below_0_loss = self.below_0_loss + below_0_loss
        self.above_100_loss = self.above_100_loss + above_100_loss
        self.samples_seen = self.samples_seen + masked_pred.shape[0]

    def compute(self):
        if self.avg_batch_size is None:
            smoothness_loss = (
                self.below_0_loss + self.above_100_loss
            ) / self.samples_seen.to(torch.float32)
        else:
            smoothness_loss = (
                self.below_0_loss + self.above_100_loss
            ) / self.avg_batch_size
        self.value = smoothness_loss * self.lambd

    def reset(self):
        self._init_values()
