"""
Contains an implementation of the smoothness loss function.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch


def get_smoothness_loss(
    lambd: float, avg_batch_size: None | float, device: torch.device
):
    def inner():
        return SmoothnessLoss(lambd=lambd, avg_batch_size=avg_batch_size, device=device)

    return inner


class SmoothnessLoss:
    def __init__(
        self, lambd: float, avg_batch_size: None | float, device: torch.device
    ) -> None:
        self.__name__ = "Smoothness"
        self.lambd = torch.tensor(lambd, dtype=torch.float32, device=device)
        self.avg_batch_size = avg_batch_size
        self.device = device
        self._init_values()

    def _init_values(self):
        self.sum_smoothness_loss = torch.tensor(
            0.0, dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.samples_seen = torch.tensor(0, dtype=torch.int32, device=self.device)

    def __call__(
        self,
        mask: torch.Tensor,  # Shape (batch_size, 1, height, width)
        y_pred: torch.Tensor,  # Shape (batch_size, 1, height, width)
    ):
        mask = mask[
            ..., :-1, :-1
        ]  # Remove the last row and column to match the derivative shape
        non_batch_dims = tuple(range(1, y_pred.dim()))
        # Compute the gradient of the prediction in the x and y directions
        dx = torch.diff(y_pred, dim=-2)
        dx = dx[..., :, :-1]  # Remove the last column to match the mask shape
        dy = torch.diff(y_pred, dim=-1)
        dy = dy[..., :-1, :]  # Remove the last row to match the mask shape

        # Compute and apply the mask for the derivatives
        dx = dx * mask
        dy = dy * mask
        mask_size = torch.sum(mask, dim=non_batch_dims)  # Shape (batch_size,)

        # Compute the smoothness loss per batch sample
        dx_loss = torch.sum(dx**2, dim=non_batch_dims)  # Shape (batch_size,)
        dy_loss = torch.sum(dy**2, dim=non_batch_dims)  # Shape (batch_size,)

        # Compute the total smoothness per mask size
        smoothness_loss = (dx_loss + dy_loss) / mask_size  # Shape (batch_size,)
        # Sum across all dimensions
        smoothness_loss = torch.sum(smoothness_loss)  # Shape ()
        self.sum_smoothness_loss = self.sum_smoothness_loss + smoothness_loss
        self.samples_seen = self.samples_seen + y_pred.shape[0]

    def compute(self):
        if self.avg_batch_size is None:
            value = self.sum_smoothness_loss / self.samples_seen.to(torch.float32)
        else:
            value = self.sum_smoothness_loss / self.avg_batch_size
        self.value = value * self.lambd

    def reset(self):
        self._init_values()
