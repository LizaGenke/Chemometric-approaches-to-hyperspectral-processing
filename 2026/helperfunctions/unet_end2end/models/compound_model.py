"""
Contains an implementation of a Module that sequentially applies multiple models.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch.nn as nn


class CompoundModel(nn.Module):
    def __init__(self, *models):
        super(CompoundModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        for model in self.models:
            x = model(x)
        return x
