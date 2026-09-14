"""
Contains an implementation of a the EarlyStopping callback. It monitors a metric and
stops training if the metric does not improve for a specified number of epochs (patience).

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import torch


class EarlyStopping:
    def __init__(self, patience: int, mode: str) -> None:
        self.__name__ = "EarlyStopping"
        self.patience = patience
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self._init_values()

    def _init_values(self):
        best_metric = torch.tensor(
            float("inf") if self.mode == "min" else float("-inf"),
            dtype=torch.float32,
            requires_grad=False,
        )
        self._set_best_metric(best_metric)
        self.counter = 0

    def _set_best_metric(self, metric: torch.Tensor):
        self.best_metric = metric

    def _should_stop(self):
        return self.counter >= self.patience

    def __call__(self, metric: torch.Tensor):
        if self.mode == "min":
            if metric < self.best_metric:
                self._set_best_metric(metric)
                self.counter = 0
            else:
                self.counter += 1
        else:
            if metric > self.best_metric:
                self._set_best_metric(metric)
                self.counter = 0
            else:
                self.counter += 1
        return self._should_stop()

    def update_best_metric(self, metric: torch.Tensor):
        if self.mode == "min":
            if metric < self.best_metric:
                self._set_best_metric(metric)
        else:
            if metric > self.best_metric:
                self._set_best_metric(metric)
