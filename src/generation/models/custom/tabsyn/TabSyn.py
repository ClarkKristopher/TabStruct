import os

import wandb
from src.common import LOG_DIR, WANDB_PROJECT
from src.common.runtime.log.WandbHelper import WandbHelper

from ...BaseGenerator import BaseMixedGenerator
from .official.main import run_with_processed_data as tabsyn_main
from .official.sample import sample_without_recover_data as tabsyn_sample
from .official.vae.main import run_with_processed_data as vae_main


class TabSyn(BaseMixedGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression"]:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

        self.vae_params = self.args.model_params["vae"]
        self.tabsyn_params = self.args.model_params["tabsyn"]

        self.patience_for_generation = 5

    def _fit(self, data_module):
        log_dir = os.path.join(LOG_DIR, WANDB_PROJECT, wandb.run.id)

        # === Prepare the data according to feature type ===
        data_dict = self.prepare_data(data_module)

        # === Fit VAE first ===
        self.vae_log = vae_main(
            X_train_num=data_dict["X_train_num"],
            X_train_cat=data_dict["X_train_cat"],
            X_test_num=data_dict["X_valid_num"],
            X_test_cat=data_dict["X_valid_cat"],
            log_dir=log_dir,
            device=self.args.device,
            **self.vae_params,
        )

        # === Fit TabSyn ===
        tabsyn_main(
            log_dir=log_dir,
            device=self.args.device,
            **self.tabsyn_params,
        )

    def _model_generate(self):
        if self.args.saved_checkpoint_path is not None:
            run_id = WandbHelper.parse_run_id_from_path(self.args.saved_checkpoint_path)
        else:
            run_id = wandb.run.id
        log_dir = os.path.join(LOG_DIR, WANDB_PROJECT, run_id)
        synthetic_data_dict = tabsyn_sample(
            log_dir=log_dir,
            device=self.args.device,
            d_numerical=self.vae_log["d_numerical"],
            categories=self.vae_log["categories"],
            num_layers=self.vae_params["num_layers"],
            d_token=self.vae_params["d_token"],
            n_head=self.vae_params["n_head"],
            factor=self.vae_params["factor"],
        )

        return synthetic_data_dict

    @classmethod
    def define_default_params(cls):
        params = {
            "vae": {
                "num_epochs": 500,
                "max_beta": 1e-2,
                "min_beta": 1e-5,
                "lambd": 0.7,
                "num_layers": 2,
                "d_token": 4,
                "n_head": 1,
                "factor": 32,
                "lr": 1e-3,
                "wd": 0,
            },
            "tabsyn": {
                "num_epochs": 500,
                "lr": 1e-3,
                "wd": 0,
            },
        }

        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "vae": {
                "num_epochs": trial.suggest_int("num_epochs", 100, 1000),
                "max_beta": trial.suggest_float("max_beta", 1e-3, 1e-2, log=True),
                "min_beta": trial.suggest_float("min_beta", 1e-5, 1e-4, log=True),
                "lambd": trial.suggest_float("lambd", 0.1, 1.0),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "d_token": trial.suggest_int("d_token", 1, 8),
                "n_head": trial.suggest_int("n_head", 1, 4),
                "factor": trial.suggest_int("factor", 1, 64),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "wd": trial.suggest_float("wd", 0, 1e-2, log=True),
            },
            "tabsyn": {
                "num_epochs": trial.suggest_int("num_epochs", 100, 500),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "wd": trial.suggest_float("wd", 0, 1e-2, log=True),
            },
        }

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "vae": {
                "num_epochs": 1,
                "max_beta": 1e-2,
                "min_beta": 1e-5,
                "lambd": 0.7,
                "num_layers": 2,
                "d_token": 4,
                "n_head": 1,
                "factor": 32,
                "lr": 1e-3,
                "wd": 0,
            },
            "tabsyn": {
                "num_epochs": 1,
                "lr": 1e-3,
                "wd": 0,
            },
        }

        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "vae": {
                "num_epochs": 1,
                "max_beta": 1e-2,
                "min_beta": 1e-5,
                "lambd": 0.7,
                "num_layers": 2,
                "d_token": 4,
                "n_head": 1,
                "factor": 32,
                "lr": 1e-3,
                "wd": 0,
            },
            "tabsyn": {
                "num_epochs": 1,
                "lr": 1e-3,
                "wd": 0,
            },
        }

        return params
