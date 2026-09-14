"""
Contains an implementation of a Pipeline class to train and evaluate a PyTorch model.
It combines a model, data loaders, loss functions, metrics, optimizer, scheduler,
callbacks, and other components to train the model in a structured way.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..callbacks.early_stopping import EarlyStopping
from ..callbacks.model_checkpoint import ModelCheckpoint


class Pipeline:
    """
    A class to train a PyTorch model using a training loop.

    Paramters
    ---------

    model: torch.nn.Module
        A PyTorch model.

    train_loader: torch.utils.data.DataLoader
        A PyTorch DataLoader for the training set.

    val_loader: torch.utils.data.DataLoader
        A PyTorch DataLoader for the validation set.

    test_loader: torch.utils.data.DataLoader
        A PyTorch DataLoader for the test set.

    train_eval_loader: torch.utils.data.DataLoader
        A PyTorch DataLoader for the training set to evaluate the model after training.
        This can be the same as the train_loader. But it can also be different if, for
        example, the train_loader uses a sampler that shuffles the data or randomly
        flips the images, while the train_eval_loader does not shuffle the data and
        does not flip the images.

    loss_fn: Callable[
            [torch.nn.Module, Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
            torch.Tensor,
        ]
        A PyTorch loss function.
        The function should take the model outputs and the targets, and return a scalar
        tensor.

    metric_fn: Callable[
            [],
            Callable[
                [torch.nn.Module, Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
        ]
        A function that returns a metric function. The metric function is evaluated and
        its results are logged and may be used for model selection but the results are
        not differentiated and hence not used for training.

    metric_decision_idx: int
        The index of the metric to use for model selection. This is the metric that
        the early stopping callback and the learning rate scheduler will use to decide
        whether to stop training or to reduce the learning rate.

    optimizer: torch.optim.Optimizer
        A PyTorch optimizer.

    scheduler: torch.optim.lr_scheduler._LRScheduler
        A PyTorch learning rate scheduler.

    early_stopping: EarlyStopping
        A callback to stop the training loop based on a metric.

    model_checkpoint: ModelCheckpoint
        Exposes a save_best_weights method to save the best model weights based on a metric.
        Exposes a load_best_weights method to load the best model weights.

    num_burn_in_epochs: int
        The number of burn-in epochs to run before applying early stopping and before
        applying the learning rate scheduler.

    save_dir: Path
        A directory to save the best model weights, metrics, and logs.

    device: str, optional, default="cuda" if torch.cuda.is_available() else "cpu"
        The device to run the model on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        train_eval_loader: torch.utils.data.DataLoader,
        loss_fn: Callable[
            [torch.nn.Module, Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
            torch.Tensor,
        ],
        metric_fn: Callable[
            [],
            Callable[
                [torch.nn.Module, Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
        ],
        metric_decision_idx: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        early_stopping: EarlyStopping,
        model_checkpoint: ModelCheckpoint,
        num_burn_in_epochs: int,
        save_dir: Path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_eval_loader = train_eval_loader
        self.loss_fn = loss_fn
        self.metric_decision_idx = metric_decision_idx

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.num_burn_in_epochs = num_burn_in_epochs
        self.save_dir = save_dir
        self.device = device
        self.train_metrics = metric_fn()
        self.val_metrics = metric_fn()
        self.test_metrics = metric_fn()

    def _init_logs(self):
        self.logs = {
            "lr": [],
            "train": {m.__name__: [] for m in self.train_metrics.all_losses},
            "val": {m.__name__: [] for m in self.val_metrics.all_losses},
            "test": {m.__name__: [] for m in self.test_metrics.all_losses},
        }

    def _update_logs(
        self,
        lr,
    ):
        self.logs["lr"].append(lr)
        for (
            train_metric,
            val_metric,
            test_metric,
            train_metric_name,
            val_metric_name,
            test_metric_name,
        ) in zip(
            self.train_metrics.value,
            self.val_metrics.value,
            self.test_metrics.value,
            self.train_metrics.name,
            self.val_metrics.name,
            self.test_metrics.name,
        ):
            self.logs["train"][train_metric_name].append(train_metric.cpu().numpy())
            self.logs["val"][val_metric_name].append(val_metric.cpu().numpy())
            self.logs["test"][test_metric_name].append(test_metric.cpu().numpy())

    def _save_logs(self):
        # Flatten the nested dictionaries
        flattened_logs = {}
        for key, value in self.logs.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened_logs[f"{key} {sub_key}"] = sub_value
            else:
                flattened_logs[key] = value

        # Create DataFrame from the flattened dictionary
        logs_df = pd.DataFrame(flattened_logs)

        # Add epoch column
        logs_df["epoch"] = logs_df.index + 1

        # Save to CSV
        logs_df.to_csv(self.save_dir / "logs.csv", index=False)

    def _compute_metrics(self):
        with torch.no_grad():
            for metric in [self.train_metrics, self.val_metrics, self.test_metrics]:
                metric.compute()
                metric.reset()

    def _send_item_to_device(self, item):
        return item.to(self.device)

    def _send_items_to_device(self, items):
        # targets may be nested in a tuple or list. There may be multiple layers of nesting.
        if isinstance(items, torch.Tensor):
            return self._send_item_to_device(items)
        else:
            return tuple(
                (
                    self._send_item_to_device(target)
                    if isinstance(target, torch.Tensor)
                    else self._send_items_to_device(target)
                )
                for target in items
            )

    def _evaluate(self, loader, metrics):
        self.model.eval()
        with torch.no_grad():
            for inputs, targets, file_ids in tqdm(
                loader, desc="Evaluating...", total=len(loader)
            ):
                inputs = self._send_items_to_device(inputs)
                targets = self._send_items_to_device(targets)
                outputs = self.model(inputs)
                metrics(self.model, targets, outputs)

    def _save_items(self, file_ids, items, dataset_save_dir):
        # if targets are nested in a tuple or list, save each target separately
        if isinstance(items, torch.Tensor):
            items = items.cpu().numpy()
            for file_id, item in zip(file_ids, items):
                np.save(dataset_save_dir / f"{file_id}.npy", item)
        else:
            for j, target in enumerate(items):
                target = target.cpu().numpy()
                for file_id, item in zip(file_ids, target):
                    np.save(dataset_save_dir / f"{file_id}_{j}.npy", item)

    def _predict(self):
        self.model.eval()
        with torch.no_grad():
            for dataset, loader in zip(
                ["train", "val", "test"],
                [self.train_eval_loader, self.val_loader, self.test_loader],
            ):
                dataset_targets_save_dir = self.save_dir / dataset / "targets"
                dataset_targets_save_dir.mkdir(parents=True, exist_ok=True)
                dataset_predictions_save_dir = self.save_dir / dataset / "predictions"
                dataset_predictions_save_dir.mkdir(parents=True, exist_ok=True)
                for inputs, targets, file_ids in tqdm(
                    loader, desc="Predicting...", total=len(loader)
                ):
                    inputs = self._send_items_to_device(inputs)
                    targets = self._send_items_to_device(targets)
                    output = self.model(inputs)
                    self._save_items(file_ids, targets, dataset_targets_save_dir)
                    self._save_items(file_ids, output, dataset_predictions_save_dir)

    def _train_one_epoch(self):
        self.model.train()
        for inputs, targets, _file_ids in tqdm(
            self.train_loader, desc="Training epoch...", total=len(self.train_loader)
        ):
            inputs = self._send_items_to_device(inputs)
            targets = self._send_items_to_device(targets)
            self.optimizer.zero_grad()

            preds = self.model(inputs)
            self.loss_fn(self.model, targets, preds)
            self.loss_fn.compute()
            loss = self.loss_fn.value
            loss.backward()
            self.optimizer.step()
            self.loss_fn.reset()

    def _evaluate_all(self, train, val, test):
        if train:
            self._evaluate(self.train_eval_loader, self.train_metrics)
        if val:
            self._evaluate(
                self.val_loader,
                self.val_metrics,
            )
        if test:
            self._evaluate(self.test_loader, self.test_metrics)

    def _fprint_metric(self, metric):
        return f"{metric.__name__}: {metric.value}"

    def _print_newest_metrics(self):
        for metric in self.train_metrics.all_losses:
            print(f"Train {self._fprint_metric(metric)}")
        for metric in self.val_metrics.all_losses:
            print(f"Val {self._fprint_metric(metric)}")
        for metric in self.test_metrics.all_losses:
            print(f"Test {self._fprint_metric(metric)}")

    def _finalize(self):
        print("Finalizing training...")
        self.model_checkpoint.load_best_weights()
        # Final bias correction and evaluation of the model
        self._evaluate_all(train=True, val=True, test=True)
        self._compute_metrics()
        self._update_logs(-1)
        self._print_newest_metrics()
        # Saving of logs
        self._save_logs()
        # Saving of predictions
        self._predict()
        # Saving of the final model weights after bias has possibly been corrected
        self.model_checkpoint.save_weights()

    def train(self, max_epochs):
        if self.scheduler.mode == "min":
            best_burn_in_decision_metric_value = torch.tensor(
                float("inf"), dtype=torch.float32, device=self.device
            )
        else:
            best_burn_in_decision_metric_value = torch.tensor(
                float("-inf"), dtype=torch.float32, device=self.device
            )
        self._init_logs()
        self.model.to(self.device)
        self.model_checkpoint.save_weights()
        # Run until early stopping callback stops it or max_epochs is reached
        for epoch in tqdm(
            range(1, max_epochs + 1), desc="Training all epochs...", total=max_epochs
        ):
            print(f"Epoch {epoch}:")
            # Train one epoch
            self._train_one_epoch()

            # Evaluate the model on the train, val, and test sets
            self._evaluate_all(train=True, val=True, test=False)

            # Finalize metrics
            self._compute_metrics()

            # Get the learning rate for the current epoch
            cur_lr = self.scheduler.get_last_lr()[0]

            # Update logs
            self._update_logs(cur_lr)

            decision_metric_value = self.val_metrics.value[self.metric_decision_idx]
            if epoch <= self.num_burn_in_epochs:
                if self.scheduler.mode == "min":
                    is_best = decision_metric_value < best_burn_in_decision_metric_value
                else:
                    is_best = decision_metric_value > best_burn_in_decision_metric_value
                if is_best:
                    best_burn_in_decision_metric_value = decision_metric_value

            # Save the best model weights based on the specified metric
            self.model_checkpoint.save_best_weights(decision_metric_value)

            self._print_newest_metrics()
            print(f"Learning Rate: {cur_lr}")

            if epoch == self.num_burn_in_epochs:
                print("Burn-in epochs completed.")
                self.early_stopping.update_best_metric(
                    best_burn_in_decision_metric_value
                )
                self.scheduler.step(best_burn_in_decision_metric_value)

            if epoch > self.num_burn_in_epochs:
                # Check if early stopping should be called
                if self.early_stopping(decision_metric_value):
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Check if the learning rate scheduler has reduced the learning rate
                self.scheduler.step(decision_metric_value)
                new_lr = self.scheduler.get_last_lr()[0]

                # Restore best weights if the learning rate scheduler has reduced the learning rate
                if new_lr < cur_lr:
                    print(
                        f"Learning rate reduced from {cur_lr} to {new_lr}. Restoring best weights."
                    )
                    # Print a random weight to sanity check that it is indeed updated
                    self.model_checkpoint.load_best_weights()

        # Finalize
        self._finalize()
