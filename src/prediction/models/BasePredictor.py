from abc import abstractmethod

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar, Timer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

import wandb
from src.common import LOG_DIR, WANDB_PROJECT
from src.common.model.BaseModel import BaseModel
from src.common.runtime.log.TerminalIO import TerminalIO


class BasePredictor(BaseModel):

    def __init__(self, args):
        super().__init__(args)

    @abstractmethod
    def predict(self, X: np.ndarray):
        """Predicts the labels of the test dataset (X).

        Args:
            X (np.ndarray): The test dataset. Default to be on CPU.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def predict_proba(self, X: np.ndarray):
        """Predicts the probabilities of the test dataset (X).

        Args:
            X (np.adarray): The test dataset. Default to be on CPU.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def feature_selection(self, X=None):
        """Selects the features.

        Args:
            X (np.ndarray, optional): The test dataset. Defaults to None.
        """
        raise NotImplementedError("This method has to be implemented by the sub class")


class BaseSklearnPredictor(BasePredictor):

    def __init__(self, args):
        super().__init__(args)

    def fit(self, data_module):
        self.model.fit(data_module.X_train, data_module.y_train)

    def predict(self, X):
        y_pred = self.model.predict(X)

        return y_pred

    def predict_proba(self, X):
        if self.args.task == "classification":
            y_hat = self.model.predict_proba(X)
        else:
            y_hat = None

        return y_hat


class BaseLitPredictor(BasePredictor):

    def __init__(self, args):
        super().__init__(args)

    def fit(self, data_module):
        trainer = self.create_lit_trainer()

        self.train_lit_model(data_module, trainer)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def create_lit_trainer(self):
        # ===== Prepare callbacks =====
        callbacks = []
        # === Stop single run after 2 hours ===
        timer_callback = Timer(duration="00:02:00:00")
        callbacks.append(timer_callback)
        # === Set up training metric ===
        mode_metric = "max" if self.args.metric_model_selection == "balanced_accuracy" else "min"
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{LOG_DIR}/{WANDB_PROJECT}/{wandb.run.id}/",
            monitor=f"valid_metrics/{self.args.metric_model_selection}",
            mode=mode_metric,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        # === Terminal style ===
        callbacks.append(RichProgressBar())
        # === Add callback functions for training ===
        if self.args.patience_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=f"valid_metrics/{self.args.metric_model_selection}",
                    mode=mode_metric,
                    patience=self.args.patience_early_stopping,
                )
            )
        # === Only monitor when wandb is enabled ===
        if not self.args.disable_wandb:
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        # ===== Set up trainer =====
        trainer = L.Trainer(
            # Training
            max_steps=self.args.max_steps,
            gradient_clip_val=self.args.gradient_clip_val,
            # logging
            logger=self.args.wandb_logger,
            log_every_n_steps=self.args.log_every_n_steps,
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            callbacks=callbacks,
            # miscellaneous
            accelerator=self.args.accelerator,
            detect_anomaly=self.args.debugging,
            deterministic=self.args.deterministic,
            # used for debugging, but it may crash when validation is not performed before showing results
            # fast_dev_run=True,
        )

        return trainer

    def train_lit_model(self, data_module, trainer):
        # === Train ===
        trainer.fit(self.model, data_module)

        # === Load the best model for evaluation ===
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        model_checkpoint = torch.load(checkpoint_path)
        weights = model_checkpoint["state_dict"]

        TerminalIO.print("Loading weights into model from {}.".format(checkpoint_path), color=TerminalIO.OKGREEN)
        missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
        self.model.to(self.args.device)
        self.model.eval()

        TerminalIO.print("Missing keys:", color=TerminalIO.WARNING)
        TerminalIO.print(missing_keys, color=TerminalIO.WARNING)

        TerminalIO.print("Unexpected keys:", color=TerminalIO.WARNING)
        TerminalIO.print(unexpected_keys, color=TerminalIO.WARNING)
