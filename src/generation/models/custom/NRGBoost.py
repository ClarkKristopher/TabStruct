import numpy as np
import pandas as pd
from nrgboost import Dataset, NRGBooster

from ..BaseGenerator import BaseGenerator


class NRGBoost(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression"]:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

        self.model = None
        self.params = self.args.model_params

    def _fit(self, data_module):
        # === Save the data ===
        if self.args.task == "classification":
            self.X_train = data_module.X_train
            self.y_train = data_module.y_train
        elif self.args.task == "regression":
            # The samples are sorted by class id (0->real data, 1->dummy data)
            self.X_train = np.concatenate([data_module.X_train, data_module.y_train.reshape(-1, 1)], axis=1)
            self.y_train = np.zeros(self.X_train.shape[0], dtype=np.int64)

    def _generate(self, class2synthetic_samples):
        # === Prepare the data and generatation configurations ===
        X_train = self.X_train
        y_train = self.y_train

        if self.args.task == "regression":
            class2synthetic_samples = {
                0: class2synthetic_samples["real"],
                1: class2synthetic_samples["dummy"],
            }

        synthetic_data_dict = {}
        for class_id, num_synthetic_samples in class2synthetic_samples.items():
            if num_synthetic_samples == 0:
                continue
            X_train_class = X_train[y_train == class_id]
            train_df = pd.DataFrame(X_train_class)
            train_ds = Dataset(train_df)

            model = NRGBooster.fit(train_ds, self.params, seed=self.args.seed)

            samples_df = model.sample(num_samples=num_synthetic_samples).to_numpy()
            synthetic_data_dict[class_id] = samples_df

        # === Format the synthetic data for return ===
        X_syn = np.concatenate(list(synthetic_data_dict.values()))
        if self.args.task == "classification":
            y_syn = np.concatenate(
                [
                    [class_id] * num_synthetic_samples
                    for class_id, num_synthetic_samples in class2synthetic_samples.items()
                ]
            ).reshape(-1)
        elif self.args.task == "regression":
            X_syn = X_syn[:, :-1]
            y_syn = X_syn[:, -1]
        else:
            raise ValueError(f"Task {self.args.task} is not supported by {self.args.model}")

        return {
            "X_syn": X_syn,
            "y_syn": y_syn,
        }

    @classmethod
    def define_default_params(cls):
        params = {
            "num_trees": 1,
            "shrinkage": 0.15,
            "max_leaves": 256,
            "max_ratio_in_leaf": 2,
            "num_model_samples": 80_000,
            "p_refresh": 0.1,
            "num_chains": 16,
            "burn_in": 100,
        }

        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {
            "num_trees": trial.suggest_int("num_trees", 100, 500),
        }

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "num_trees": 1,
            "shrinkage": 0.15,
            "max_leaves": 256,
            "max_ratio_in_leaf": 2,
            "num_model_samples": 80_000,
            "p_refresh": 0.1,
            "num_chains": 16,
            "burn_in": 100,
        }

        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "num_trees": 1,
            "shrinkage": 0.15,
            "max_leaves": 256,
            "max_ratio_in_leaf": 2,
            "num_model_samples": 80_000,
            "p_refresh": 0.1,
            "num_chains": 16,
            "burn_in": 100,
        }

        return params
