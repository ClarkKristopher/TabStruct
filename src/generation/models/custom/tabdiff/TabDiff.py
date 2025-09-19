import copy
import os

import wandb
from src.common import LOG_DIR, WANDB_PROJECT

from ...BaseGenerator import BaseMixedGenerator
from .official.main import run_with_processed_data as tabdiff_main


class TabDiff(BaseMixedGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression"]:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

        self.tabdiff_config = self.args.model_params
        self.tabdiff_config["dataset"] = self.args.dataset
        self.tabdiff_log_dict = None

        self.patience_for_generation = 5

    def _fit(self, data_module):
        log_dir = os.path.join(LOG_DIR, WANDB_PROJECT, wandb.run.id)

        # === Prepare the data according to feature type ===
        data_dict = self.prepare_data(data_module)

        # === Fit TabDiff ===
        if self.tabdiff_log_dict is None:
            self.tabdiff_log_dict = tabdiff_main(
                X_train_num=data_dict["X_train_num"],
                X_train_cat=data_dict["X_train_cat"],
                log_dir=log_dir,
                device=self.args.device,
                config_dict=self.tabdiff_config,
                ckpt_path=None,
            )
        else:
            # Note: we cannot use `self.tabdiff_config["train"]["main"]` as it was used in the previous training
            self.tabdiff_log_dict["trainer"].update_training_config(
                self.define_params(reg_test=self.args.reg_test, dev="dev" in self.args.tags)["train"]["main"]
            )
            self.tabdiff_log_dict["trainer"].run_loop()

    def _model_generate(self):
        synthetic_data_dict = self.tabdiff_log_dict["trainer"].sample_without_recover_data(
            num_samples=self.args.train_num_samples_processed
        )

        return synthetic_data_dict

    @classmethod
    def define_default_params(cls):
        params = copy.deepcopy(template_config_dict)
        params["train"]["main"]["steps"] = 300

        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = copy.deepcopy(template_config_dict)
        params["train"]["main"]["steps"] = trial.suggest_int("steps", 4000, 10000)

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = copy.deepcopy(template_config_dict)

        return params

    @classmethod
    def _define_test_params(cls):
        params = copy.deepcopy(template_config_dict)
        params["train"]["main"]["steps"] = 5

        return params


template_config_dict = {
    "data": {
        "dequant_dist": "none",
        "int_dequant_factor": 0,
    },
    "unimodmlp_params": {
        "bias": True,
        "d_token": 4,
        "dim_t": 1024,
        "factor": 32,
        "n_head": 1,
        "num_layers": 2,
        "use_mlp": True,
    },
    "diffusion_params": {
        "cat_scheduler": "log_linear",
        "noise_dist": "uniform_t",
        "num_timesteps": 50,
        "scheduler": "power_mean",
        "non_learnable_schedule": False,
        "sampler_params": {
            "second_order_correction": True,
            "stochastic_sampler": True,
        },
        "edm_params": {
            "net_conditioning": "sigma",
            "precond": True,
            "sigma_data": 1.0,
        },
        "noise_dist_params": {"P_mean": -1.2, "P_std": 1.2},
        "noise_schedule_params": {
            "eps_max": 0.001,
            "eps_min": 1e-05,
            "k_init": -6.0,
            "k_offset": 1.0,
            "rho": 7,
            "rho_init": 7.0,
            "rho_offset": 5.0,
            "sigma_max": 80,
            "sigma_min": 0.002,
        },
    },
    "train": {
        "main": {
            "batch_size": 4096,
            "c_lambda": 1.0,
            "check_val_every": 30,
            "closs_weight_schedule": "anneal",
            "d_lambda": 1.0,
            "ema_decay": 0.997,
            "factor": 0.9,
            "lr": 0.001,
            "lr_scheduler": "reduce_lr_on_plateau",
            "reduce_lr_patience": 50,
            "steps": 300,
            "weight_decay": 0,
        }
    },
    "sample": {
        "batch_size": 10000,
    },
}
