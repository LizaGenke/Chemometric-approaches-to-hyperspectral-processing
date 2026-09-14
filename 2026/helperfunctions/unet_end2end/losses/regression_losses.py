"""
Contains an implementation of the mean squared error (MSE) loss function.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch


def get_mse_loss(lambd: float, avg_batch_size: None | float, device: torch.device):
    def inner():
        return MSELoss(lambd=lambd, avg_batch_size=avg_batch_size, device=device)

    return inner


class MSELoss:
    def __init__(
        self, lambd: float, avg_batch_size: None | float, device: torch.device
    ) -> None:
        self.__name__ = "MSE"
        self.lambd = torch.tensor(lambd, device=device)
        self.avg_batch_size = avg_batch_size
        self.device = device
        self._init_values()

    def _init_values(self):
        self.sum_squared_error = torch.tensor(
            0.0, device=self.device, requires_grad=True
        )
        self.samples_seen = torch.tensor(0, dtype=torch.int32, device=self.device)

    def __call__(
        self,
        mask: torch.Tensor,
        target: torch.Tensor,
        masked_pred: torch.Tensor,
    ):
        spatial_dims = tuple(range(2, masked_pred.dim()))
        average_masked_pred = masked_pred.sum(dim=spatial_dims) / mask.sum(
            dim=spatial_dims
        )
        sum_squared_error = torch.sum((target - average_masked_pred) ** 2)
        self.sum_squared_error = self.sum_squared_error + sum_squared_error
        self.samples_seen = self.samples_seen + target.shape[0]

    def compute(self):
        if self.avg_batch_size is None:
            mse = self.sum_squared_error / self.samples_seen.to(torch.float32)
        else:
            mse = self.sum_squared_error / self.avg_batch_size
        self.value = mse * self.lambd

    def reset(self):
        self._init_values()
