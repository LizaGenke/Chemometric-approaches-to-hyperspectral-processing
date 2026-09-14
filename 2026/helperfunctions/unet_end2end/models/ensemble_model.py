"""
Contains an implementation of a Module that averages the outputs of multiple models.
This can be used to construct an ensemble of models.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, *models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models])
        return torch.mean(outputs, axis=0)  # Average the outputs of all models
