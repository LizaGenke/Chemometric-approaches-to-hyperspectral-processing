"""
Contains a function to return the Adam optimizer.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch


def Adam(model: torch.nn.Module, lr: float):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    return optim
