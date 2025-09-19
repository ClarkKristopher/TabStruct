from synthcity.plugins.generic.plugin_great import GReaTPlugin

from src.common.runtime.error.ManualStopError import ManualStopError

from ..BaseGenerator import BaseSynthcityConditionalGenerator


class GReaT(BaseSynthcityConditionalGenerator):

    def __init__(self, args):
        super().__init__(args)

        self.model = GReaTPlugin(
            # Architecture
            # Optimization
            n_iter=args.model_params["optimization"]["n_iter"],
            # Misc.
            sampling_patience=1,
            strict=True,
        )

    @classmethod
    def define_default_params(cls):
        params_arch = {}

        params_optim = {
            "n_iter": 5,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_optuna_params(cls, trial):
        params_arch = {}

        params_optim = {
            "n_iter": trial.suggest_int("n_iter", 50, 500),
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_single_run_params(cls):
        params_arch = {}

        params_optim = {
            "n_iter": 5,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }

    @classmethod
    def _define_test_params(cls):
        params_arch = {}

        params_optim = {
            "n_iter": 1,
        }

        return {
            "architecture": params_arch,
            "optimization": params_optim,
        }
