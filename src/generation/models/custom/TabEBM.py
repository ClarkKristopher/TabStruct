import os
import random
from collections import defaultdict

import numpy as np
import scipy
import torch
from tabpfn import TabPFNClassifier

from src.common.runtime.error.ManualStopError import ManualStopError

from ..BaseGenerator import BaseGenerator


def to_numpy(X):
    match type(X):
        case np.ndarray:
            return X
        case torch.Tensor:
            return X.detach().cpu().numpy()
        case None:
            return None
        case _:
            raise ValueError("X must be either a np.ndarray or a torch.Tensor")


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================================================================
#                               TabEBM CLASS
# ========================================================================
class TabEBM(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.task not in ["classification", "regression"]:
            raise ValueError(f"Task {args.task} is not supported by {args.model}")

        if args.full_num_features_processed > 500:
            raise ManualStopError("TabEBM does not support more than 500 features.")

        self.model = TabEBMOfficial()

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
        num_synthetic_samples_max = max(class2synthetic_samples.values())

        # === Generate the synthetic data ===
        current_synthetic_samples = 0
        synthetic_data_dict = {}
        while current_synthetic_samples < num_synthetic_samples_max:
            # If num_synthetic_samples_max is too large, we limit the number of samples to generate per iteration
            num_samples_to_generate = num_synthetic_samples_max - current_synthetic_samples
            num_samples_to_generate = min(num_samples_to_generate, 1000)
            synthetic_data_temp = self.model.generate(
                X=X_train,
                y=y_train,
                num_samples=num_samples_to_generate,
                sgld_steps=self.args.model_params["sgld_steps"],
            )
            for class_id in synthetic_data_temp.keys():
                new_samples = synthetic_data_temp[class_id][:num_samples_to_generate]
                if class_id not in synthetic_data_dict:
                    synthetic_data_dict[class_id] = new_samples
                else:
                    synthetic_data_dict[class_id] = np.concatenate([synthetic_data_dict[class_id], new_samples], axis=0)
            current_synthetic_samples += num_samples_to_generate

        # === Format the synthetic data for return ===
        if self.args.task == "classification":
            X_syn = np.concatenate(
                [
                    synthetic_data_dict[f"class_{class_id}"][
                        np.random.choice(synthetic_data_dict[f"class_{class_id}"].shape[0], num_synthetic_samples)
                    ]
                    for class_id, num_synthetic_samples in class2synthetic_samples.items()
                ],
                axis=0,
            )
            y_syn = np.concatenate(
                [
                    [class_id] * num_synthetic_samples
                    for class_id, num_synthetic_samples in class2synthetic_samples.items()
                ]
            ).reshape(-1)
        elif self.args.task == "regression":
            X_syn = synthetic_data_dict["class_0"]
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
            "sgld_steps": 100,
        }

        return params

    @classmethod
    def _define_optuna_params(cls, trial):
        params = {}

        return params

    @classmethod
    def _define_single_run_params(cls):
        params = {
            "sgld_steps": 200,
        }

        return params

    @classmethod
    def _define_test_params(cls):
        params = {
            "sgld_steps": 1,
        }

        return params


class TabEBMOfficial:
    def __init__(self, tabpfn=None):
        if tabpfn is not None:
            self.tabpfn = tabpfn
        else:
            self.tabpfn = TabPFNClassifier(
                fit_mode="fit_no_preprocessors",
                n_estimators=1,
            )

    @staticmethod
    def compute_energy(
        logits,  # Model's logit (unnormalized) for each class (shape = (num_samples, num_classes))
        return_unnormalized_prob=False,  # Whether to compute the unnormalized probability p(x) instead of the energy
    ):
        # Compute the proposed TabEBM class-specific energy E_c(x) = -log(exp(f^c(\x)[0]) + exp(f^c(x)[1]))
        if type(logits) is torch.Tensor:
            # === Assert the logits are unnormlized (and not probabilities)
            assert (logits.sum(dim=1) - 1).abs().max() > 1e-5, "Logits must be unnormalized"

            energy = -torch.logsumexp(logits, dim=1)
            if return_unnormalized_prob:
                return torch.exp(-energy)
            else:
                return energy
        elif type(logits) is np.ndarray:
            # === Assert the logits are unnormlized (and not probabilities)
            assert (logits.sum(axis=1) - 1).max() > 1e-5, "Logits must be unnormalized)"

            energy = -1 * scipy.special.logsumexp(logits, axis=1)
            if return_unnormalized_prob:
                return np.exp(-energy)
            else:
                return energy
        else:
            raise ValueError("Logits must be either a torch.Tensor or a np.ndarray")

    @staticmethod
    def add_surrogate_negative_samples(
        X,  # The data (expected to be standardized to have zero mean and unit variance)
        distance_negative_class=5,  # The distance of the surrogate "negative samples" from the data
    ):
        """
        Create the surrogate negative samples for TabEBM's proposed surrogate task (for each class)
        """
        if X.shape[1] == 2:
            true_negatives = [
                [-distance_negative_class, -distance_negative_class],
                [distance_negative_class, distance_negative_class],
                [-distance_negative_class, distance_negative_class],
                [distance_negative_class, -distance_negative_class],
            ]
        else:
            # === Generate "true negative" samples ===
            true_negatives = set()
            while len(true_negatives) < 4:
                point = np.random.choice([-distance_negative_class, distance_negative_class], X.shape[1])
                point = tuple(point)
                if point not in true_negatives:
                    true_negatives.add(point)
                    true_negatives.add(tuple(-np.array(point)))
            true_negatives = list(true_negatives)
        num_true_negatives = len(true_negatives)

        if type(X) is np.ndarray:
            X_ebm = np.array(true_negatives)
            X_ebm = np.concatenate([X, X_ebm], axis=0)
            y_ebm = np.concatenate([np.zeros(X.shape[0]), np.ones(num_true_negatives)], axis=0)
            return X_ebm, y_ebm
        elif type(X) is torch.Tensor:
            X_ebm = torch.tensor(true_negatives).float().to(X.device)
            X_ebm = torch.cat([X, X_ebm], dim=0)
            y_ebm = torch.cat([torch.zeros(X.shape[0]), torch.ones(num_true_negatives)], dim=0).long().to(X.device)
            return X_ebm, y_ebm
        else:
            raise ValueError("X must be either a np.ndarray or a torch.Tensor")

    def generate(
        self,
        X,
        y,  # The data must have been processed using TabEBMOfficial.add_surrogate_negative_samples()
        num_samples,  # Number of samples to generate (per class)
        starting_point_noise_std=0.01,  # SGLD noise standard deviation to initialise the starting points
        sgld_step_size=0.1,  # SGLD step size
        sgld_noise_std=0.01,  # SGLD noise standard deviation
        sgld_steps=200,  # Number of SGLD steps
        distance_negative_class=5,  # Distance of the "negative samples" created to have a different class
        seed=42,
    ):
        res = self._sampling_internal(
            X=X,
            y=y,
            num_samples=num_samples,
            starting_point_noise_std=starting_point_noise_std,
            sgld_step_size=sgld_step_size,
            sgld_noise_std=sgld_noise_std,
            sgld_steps=sgld_steps,
            distance_negative_class=distance_negative_class,
            seed=seed,
        )

        augmented_data = defaultdict(list)
        for target_class in range(len(np.unique(to_numpy(y)))):
            augmented_data[f"class_{int(target_class)}"] = res[f"class_{int(target_class)}"]["sampling_paths"]

        return augmented_data

    def _sampling_internal(
        self,
        X,
        y,  # The data must have been processed using add_surrogate_negative_samples()
        num_samples,  # number of samples to generate
        starting_point_noise_std=0.01,  # Noise std to compute the starting points for the sampling
        sgld_step_size=0.1,
        sgld_noise_std=0.01,
        sgld_steps=200,
        distance_negative_class=5,  # Distance of the "negative samples" created to have a different class
        seed=42,
        return_trajectory=False,  # If False, then return only save the last point of the sampling path
        return_energy_values=False,  # If True, then return the energy values of the samples
        return_gradients_energy_surface=False,  # If True, then return the gradients of the energy surface as part of the final output
        debug=False,  # if True, print debug information
    ):
        if debug:
            print("Inside TabEBM sampling")
            print(f"sgld_step_size = {sgld_step_size}")
            print(f"sgld_noise_std = {sgld_noise_std}")
            print(f"sgld_steps = {sgld_steps}")
            print(f"distance_negative_class = {distance_negative_class}")
            print(f"starting_point_noise_std = {starting_point_noise_std}")

        if return_gradients_energy_surface:
            assert (
                return_trajectory
            ), "If return_gradients_energy_surface is True, then return_trajectory must be True to get the trajectory of the gradients"

        # === Sampling for each class ===
        synthetic_data_per_class = defaultdict(list)
        for target_class in np.unique(to_numpy(y)):
            X_one_class = X[y == target_class][:1000, :]  # Limit the number of samples to 1024
            X_ebm, y_ebm = TabEBMOfficial.add_surrogate_negative_samples(
                X_one_class, distance_negative_class=distance_negative_class
            )

            X_ebm = torch.from_numpy(X_ebm).float()
            y_ebm = torch.from_numpy(y_ebm).long()
            self.tabpfn.fit(X_ebm, y_ebm)

            # ======= CREATE THE STARTING POINTS FOR RUNNING SGLD =======
            seed_everything(seed)
            X_one_class = X_ebm[y_ebm == 0]  # The convention is that the target class is always 0
            x = X_one_class[
                np.random.choice(len(X_one_class), size=num_samples)
            ]  # Select random samples from the training set
            x += starting_point_noise_std * np.random.randn(*x.shape)  # Add noise to the starting points
            x.requires_grad = True
            x.to(self.tabpfn.device_)

            gradients_energy_surface = []
            energy_values = []
            if return_trajectory:
                sgld_sampling_paths = [x.detach().cpu().numpy()]

            seed_everything(seed)
            noise = torch.randn(sgld_steps, *x.shape)
            noise.to(self.tabpfn.device_)
            for t in range(sgld_steps):
                if x.grad is not None:
                    x.grad.zero_()

                # === Compute the class-specific energy ===
                logits = self.tabpfn.predict_logits_with_grad(x)
                energy = TabEBMOfficial.compute_energy(logits)
                total_energy = energy.sum() / len(x)
                total_energy.backward()

                if debug:
                    print(
                        f"Step {t} has energy {total_energy.item():.3f} with gradient norm {x.grad.norm().item():.4f}"
                    )

                if return_gradients_energy_surface and return_trajectory:
                    gradients_energy_surface.append(x.grad.detach().cpu().numpy())
                if return_energy_values and return_trajectory:
                    energy_values.append(energy.detach().cpu().numpy())

                x = x.detach() - sgld_step_size * x.grad + sgld_noise_std * noise[t]
                x.requires_grad = True

                if return_trajectory:
                    sgld_sampling_paths.append(x.detach().cpu().numpy())

            if return_trajectory:
                res = {
                    "sampling_paths": np.array(sgld_sampling_paths).transpose(1, 0, 2)
                }  # shape = (num_samples, num_steps, num_features)

                if return_gradients_energy_surface:
                    res["gradients_energy_surface"] = np.array(gradients_energy_surface).transpose(
                        1, 0, 2
                    )  # shape = (num_samples, num_steps, num_features)
                if return_energy_values:
                    res["energy_values"] = np.array(energy_values).transpose(1, 0)  # shape = (num_samples, num_steps)
                return res
            else:
                res = {"sampling_paths": x.detach().cpu().numpy()}  # shape = (num_samples, num_features)

            synthetic_data_per_class[f"class_{int(target_class)}"] = res

        return synthetic_data_per_class
