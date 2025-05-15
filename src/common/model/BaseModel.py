from abc import abstractmethod

import optuna


class BaseModel:
    """Basic interface for all models.

    All implemented models should inherit from this base class to provide a common interface.
    At least they have to extend the init method defining the model and the define_trial_parameters method
    specifying the hyperparameters.

    """

    # ================================================================
    # =                                                              =
    # =                       Initialisation                         =
    # =                                                              =
    # ================================================================
    def __init__(self, args):
        """Initializes the model. Within the sub class, the __init__() method needs to include:
        - Sanity check of the arguments (e.g., TabPFN does not support regression)
        - The definition of the model architecture (self.model)
        - The case-by-case definition of the model parameters (self.params, e.g., XGBoost's device setup)

        Args:
            args (argparse.Namespace): The arguments for the experiment.
        """
        self.args = args
        self.params = args.model_params

        # Model definition has to be implemented by the concrete model
        self.model = None

    # ================================================================
    # =                                                              =
    # =                       Top-level APIs                         =
    # =                                                              =
    # ================================================================
    def get_metadata(self):
        return {
            "name": self.__class__.__name__,
            "params": self.params,
        }

    @classmethod
    def define_params(cls, reg_test, trial=None, dev=False):
        if trial is not None:
            return cls._define_optuna_params(trial)
        elif reg_test:
            return cls._define_test_params()
        elif not dev:
            return cls.define_default_params()
        else:
            return cls._define_single_run_params()

    # ================================================================
    # =                                                              =
    # =              Utils to implement in sub class                 =
    # =                                                              =
    # ================================================================
    @classmethod
    def define_default_params(cls):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _define_optuna_params(cls, trial: optuna.Trial):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _define_single_run_params(cls):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    @abstractmethod
    def _define_test_params(cls):
        raise NotImplementedError("This method has to be implemented by the sub class")

    @abstractmethod
    def fit(self, data_module):
        raise NotImplementedError("This method has to be implemented by the sub class")
