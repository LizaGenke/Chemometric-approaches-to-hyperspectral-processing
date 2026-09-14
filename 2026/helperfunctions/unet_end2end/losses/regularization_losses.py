"""
Contains an implementation of the L2 regularization loss.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm


def get_l2_loss(lambd: float, bias: bool, device: torch.device):
    def inner():
        return L2RegularizationLoss(lambd=lambd, bias=bias, device=device)

    return inner


class L2RegularizationLoss:
    def __init__(self, lambd: float, bias: bool, device: torch.device) -> None:
        self.__name__ = "L2 Regularization"
        self.lambd = torch.tensor(lambd, dtype=torch.float32, device=device)
        self.bias = bias
        self.device = device
        self._init_values()

    def _init_values(self):
        self.l2_loss = torch.tensor(
            0.0, dtype=torch.float32, device=self.device, requires_grad=True
        )

    def __call__(self, model: torch.nn.Module):
        all_params = []
        # Acumulate the L2 loss for each parameter except bias terms and beta in batch normalization
        for module_name, module in model.named_modules():
            if isinstance(module, (LayerNorm, BatchNorm1d, BatchNorm2d, BatchNorm3d)):
                continue
            for param_name, param in module.named_parameters(recurse=False):
                if self.bias:
                    all_params.append(param.flatten())
                else:
                    if "beta" not in [module_name, param_name] and "bias" not in [
                        module_name,
                        param_name,
                    ]:
                        all_params.append(param.flatten())
        self.l2_loss = torch.sum(torch.concatenate(all_params) ** 2)

    def compute(self):
        self.value = self.l2_loss * self.lambd

    def reset(self):
        self._init_values()
