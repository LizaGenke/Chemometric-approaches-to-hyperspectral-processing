"""
Contains an implementation of a ModelCheckpoint class that saves and loads the best
model weights and associated optimizer state based on a specified metric.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from pathlib import Path

import torch


class ModelCheckpoint:
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        mode: str,
        save_dir: Path,
    ) -> None:
        self.model = model
        self.optim = optim
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.save_dir / "best_weights.pt"
        self.optim_path = self.save_dir / "best_weights_optim_state.pt"
        self.best_metric = torch.tensor(
            float("inf") if self.mode == "min" else float("-inf"),
            dtype=torch.float32,
            requires_grad=False,
        )

    def save_weights(self):
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(self.optim.state_dict(), self.optim_path)

    def save_best_weights(self, metric: torch.Tensor):
        if (self.mode == "min" and metric < self.best_metric) or (
            self.mode == "max" and metric > self.best_metric
        ):
            print(f"New best metric found {self.best_metric} -> {metric}.")
            self.best_metric = metric
            self.save_weights()
            return True
        return False

    def load_best_weights(self, keep_current_lr: bool = True):
        print(f"Loading best weights from {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path, weights_only=False))
        # Restore the state of the optimizer but keep the current learning rate
        print(f"Loading best optimizer state from {self.optim_path}")
        if keep_current_lr:
            current_lr = self.optim.param_groups[0]["lr"]
        self.optim.load_state_dict(torch.load(self.optim_path, weights_only=False))
        if keep_current_lr:
            print(f"Restoring current learning rate: {current_lr}")
            self.optim.param_groups[0]["lr"] = current_lr
