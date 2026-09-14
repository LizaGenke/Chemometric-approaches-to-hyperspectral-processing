"""
Contains a function returning a learning rate scheduler.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau


def reduce_lr_on_plateau(optimizer, patience, lr_multiplier, min_lr):
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=patience - 1,
        factor=lr_multiplier,
        min_lr=min_lr,
    )
    return scheduler
