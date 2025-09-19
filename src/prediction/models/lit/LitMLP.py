import lightning as L
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from ..BasePredictor import BaseLitPredictor
from ..utils.activation import get_activation
from ..utils.evaluation import compute_all_metrics
from ..utils.optim import Lookahead


class LitMLP(BaseLitPredictor):

    def __init__(self, args):
        super().__init__(args)

        if args.task != "classification":
            raise ValueError(f"Task {args.task} is not supported for MLP model")

        self.model = _LitMLP(args)

    @classmethod
    def define_default_params(cls):
        params_arch = {
            "activation": "tanh",
            "hidden_layer_list": [100, 100, 10],
            "dropout_rate": 0,
            "batch_normalization": True,
        }

        params_optim = {
            "lr": 3e-3,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        hidden_dim = trial.suggest_int("hidden_dim", 10, 100)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        params_arch = {
            "activation": trial.suggest_categorical("activation", ["tanh", "relu", "l_relu", "sigmoid", "none"]),
            "hidden_layer_list": n_layers * [hidden_dim],
            "dropout_rate": trial.suggest_float("dropout_rate", 0, 0.5),
            "batch_normalization": trial.suggest_categorical("batch_normalization", [True, False]),
        }

        params_optim = {
            "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {
            "activation": "tanh",
            "hidden_layer_list": [100, 100, 10],
            "dropout_rate": 0,
            "batch_normalization": True,
        }

        params_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {
            "activation": "tanh",
            "hidden_layer_list": [100, 100, 10],
            "dropout_rate": 0,
            "batch_normalization": True,
        }

        param_optim = {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }

        return {
            "architecture": params_arch,
            "optimization": param_optim,
        }


class _LitMLP(L.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = MLP(
            input_dim=args.full_num_features_processed,
            output_dim=args.full_num_classes_processed,
            **args.model_params["architecture"],
        )

        self.lr = args.model_params["optimization"]["lr"]
        self.weight_decay = args.model_params["optimization"]["weight_decay"]

        self.training_step_output_list = []
        self.validation_step_output_list = []
        self.test_step_output_list = []

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.model.predict(x)

    def predict_proba(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.model.predict_proba(x)

    def forward(self, x):
        return self.model(x)

    def step(self, x, y_true):
        # Compute probabilities and predictions
        y_hat = self.forward(x)
        y_pred = y_hat if self.args.full_num_classes_processed == 1 else torch.argmax(y_hat, dim=1)

        # Compute losses
        loss_dict = self.compute_loss(y_hat, y_true)

        return {
            "total_loss": loss_dict["total_loss"],
            "loss_dict": {k: v.detach().cpu().numpy() for k, v in loss_dict.items()},
            "y_true": y_true.detach().cpu().numpy(),
            "y_pred": y_pred.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
        }

    def training_step(self, batch):
        # Load data from batch
        x, y_true = batch

        # Forward pass
        step_dict = self.step(x, y_true)

        # Log the process
        self.training_step_output_list.append(step_dict)
        self.log_losses([step_dict], key="train_metrics")

        return step_dict["total_loss"]

    def on_train_epoch_end(self):
        # log the training loss every epoch
        self.log_losses(self.training_step_output_list, key="train_metrics")
        # log the training metrics at most once per epoch
        self.log_epoch_metrics(self.training_step_output_list, key="train_metrics")

        # Clear the list of training step outputs for memory efficiency
        self.training_step_output_list.clear()

    def validation_step(self, batch):
        # Load data from batch
        x, y_true = batch

        # Forward pass
        step_dict = self.step(x, y_true)

        # Log the process
        self.validation_step_output_list.append(step_dict)

    def on_validation_epoch_end(self):
        # Log the validation loss and metrics every epoch
        self.log_losses(self.validation_step_output_list, key="valid_metrics")
        self.log_epoch_metrics(self.validation_step_output_list, key="valid_metrics")

        # Clear the list of validation step outputs for memory efficiency
        self.validation_step_output_list.clear()

    def test_step(self, batch):
        # Load data from batch
        x, y_true = batch

        # Forward pass
        step_dict = self.step(x, y_true)

        # Log the process
        self.test_step_output_list.append(step_dict)

        return step_dict["total_loss"]

    def on_test_epoch_end(self):
        # Log the validation loss and metrics every epoch
        self.log_losses(self.test_step_output_list, key=self.log_test_key)
        self.log_epoch_metrics(self.test_step_output_list, key=self.log_test_key)

        # Log the predictions
        self.log_predictions(self.test_step_output_list, key=self.log_test_key)

        # Clear the list of test step outputs for memory efficiency
        self.test_step_output_list.clear()

    def compute_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor):
        losses = self.compute_mlp_loss(y_hat, y_true)

        return losses

    def compute_mlp_loss(self, y_hat, y_true):
        losses = {}
        losses["total_loss"] = torch.zeros(1, device=self.device)
        losses["mse_loss"] = torch.zeros(1, device=self.device)
        losses["cross_entropy_loss"] = torch.zeros(1, device=self.device)

        # compute loss for prediction
        if self.args.full_num_classes_processed == 1:
            losses["mse_loss"] = F.mse_loss(input=y_hat.squeeze(-1), target=y_true)
        else:
            losses["cross_entropy_loss"] = F.cross_entropy(
                input=y_hat,
                target=y_true,
                weight=torch.tensor(self.args.train_class_weight_list, dtype=torch.float32, device=self.device),
            )

        losses["total_loss"] = losses["mse_loss"] + losses["cross_entropy_loss"]

        return losses

    def log_losses(self, step_dict_list, key, dataloader_name=""):
        loss_dict_agg = {}
        for loss_name in step_dict_list[0]["loss_dict"].keys():
            loss_dict_agg[loss_name] = np.mean([step_dict["loss_dict"][loss_name] for step_dict in step_dict_list])

        for loss_name, loss_value in loss_dict_agg.items():
            self.log(f"{key}/{loss_name}{dataloader_name}", loss_value)

    def log_epoch_metrics(self, step_dict_list, key, dataloader_name=""):
        if self.args.full_num_classes_processed == 1:
            return

        y_true = np.concatenate([step_dict["y_true"] for step_dict in step_dict_list])[:, np.newaxis]
        y_pred = np.concatenate([step_dict["y_pred"] for step_dict in step_dict_list], axis=0)
        y_hat = np.concatenate([step_dict["y_hat"] for step_dict in step_dict_list], axis=0)

        metric_dict = compute_all_metrics(self.args, y_true, y_pred, y_hat)

        for metric_name, metric_value in metric_dict.items():
            self.log(f"{key}/{metric_name}{dataloader_name}", metric_value)

    def log_predictions(self, output_list, key):
        # Save prediction probabilities and ground truth
        y_true_list = [output["y_true"] for output in output_list]
        y_true_all = np.concatenate(y_true_list, axis=0)[:, np.newaxis]
        y_hat_list = [output["y_hat"] for output in output_list]
        y_hat_all = np.concatenate(y_hat_list, axis=0)
        if self.args.full_num_classes_processed > 1:
            y_hat_all = scipy.special.softmax(y_hat_all, axis=1)

        y_hat_and_true = np.concatenate([y_hat_all, y_true_all], axis=1)
        y_hat_and_true = pd.DataFrame(y_hat_and_true)
        y_hat_and_true = wandb.Table(dataframe=y_hat_and_true)
        wandb.log({f"{key}_y_hat_and_true": y_hat_and_true})

    def configure_optimizers(self):
        params = self.parameters()

        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay, betas=[0.9, 0.98])
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)

        if self.args.enable_lookahead_optimizer:
            optimizer = Lookahead(optimizer, la_steps=5, la_alpha=0.5)

        if self.args.lr_scheduler == "none":
            return optimizer
        else:
            if self.args.lr_scheduler == "plateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=30, verbose=True
                )
            elif self.args.lr_scheduler == "cosine_warm_restart":
                # Usually the model trains in 1000 epochs. The paper "Snapshot ensembles: train 1, get M for free"
                # 	splits the scheduler for 6 periods. We split into 6 periods as well.
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.args.cosine_warm_restart_t_0,
                    eta_min=self.args.cosine_warm_restart_eta_min,
                    verbose=True,
                )
            elif self.args.lr_scheduler == "linear":
                # Warm up to base lr for stable training
                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.5,
                    end_factor=1.0,
                    # steps needed to schedule lr
                    total_iters=2 * self.args.log_every_n_steps,
                )
            elif self.args.lr_scheduler == "lambda":

                def scheduler(epoch):
                    if epoch < 500:
                        return 0.995**epoch
                    else:
                        return 0.1

                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler)
            else:
                raise Exception()

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": f"valid_metrics/{self.args.metric_model_selection}",
                    "interval": "step",
                    "name": "lr_scheduler",
                },
            }


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str,
        hidden_layer_list: list,
        batch_normalization: bool = True,
        dropout_rate: float = 0,
    ) -> None:
        """MLP for classification or regression

        Args:
            output_dim (int): number of nodes for the output layer of the prediction net, 1 (regression) or 2 (classification)
            activation (str): activation function of the prediction net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            hidden_layer_list (list): number of nodes for each hidden layer for the prediction net, example: [200,200]
        """
        super().__init__()

        self.num_classes = output_dim
        self.act = get_activation(activation)
        full_layer_list = [input_dim, *hidden_layer_list]
        self.fn = nn.Sequential()
        for i in range(len(full_layer_list) - 1):
            self.fn.add_module("fn{}".format(i), nn.Linear(full_layer_list[i], full_layer_list[i + 1]))
            self.fn.add_module("act{}".format(i), self.act)
            # use BN after activation has better performance
            if batch_normalization:
                self.fn.add_module("bn{}".format(i), nn.BatchNorm1d(full_layer_list[i + 1]))
            if dropout_rate > 0:
                self.fn.add_module("dropout{}".format(i), nn.Dropout(dropout_rate))

        self.head = nn.Sequential()
        self.head.add_module("head", nn.Linear(full_layer_list[-1], output_dim))

        # when using cross-entropy loss in pytorch, we do not need to use softmax.
        # self.head.add_module('softmax', nn.Softmax(-1))

    def forward(self, x):
        x_emb = self.fn(x)
        x = self.head(x_emb)

        return x

    def predict(self, x):
        y_hat = self.forward(x)
        y_pred = y_hat if self.num_classes == 1 else torch.argmax(y_hat, dim=1)
        return y_pred.detach().cpu().numpy()

    def predict_proba(self, x):
        y_hat = self.forward(x)
        if self.num_classes > 1:
            y_hat = F.softmax(y_hat, dim=1)
        return y_hat.detach().cpu().numpy()
