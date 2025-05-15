from __future__ import annotations

import os
import random
import warnings
from copy import deepcopy
from typing import Optional

import cloudpickle
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from sklearn.model_selection import train_test_split

from .. import LOCAL_DATA_PATH, dataset_with_dag
from ..utils.loading import load_tabular_dataset
from .stype import stype


class TabularDataset:
    # ================================================================
    # =                                                              =
    # =                    Construct instances                       =
    # =                                                              =
    # ================================================================
    def __init__(
        self,
        dataset_name: str,
        task_type: str,
        target_col: Optional[str] = None,
        metafeature_dict: Optional[dict] = None,
        data_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialise the TabularDataset.

        Args:
            dataset_name (str): Name of the dataset.
            task_type (str): Type of task (classification or regression).
            target_col (Optional[str], optional): Name of the target column. Defaults to None.
            metafeature_dict (Optional[dict], optional): Dictionary containing the metafeatures. Defaults to None.
            data_df (Optional[pd.DataFrame], optional): DataFrame containing the features and target. Defaults to None.
        """
        # ===== Sanity check for arguments =====
        if task_type not in ["classification", "regression"]:
            raise ValueError(f"Task type {task_type} not recognised. Must be either 'classification' or 'regression'.")
        if target_col is None:
            warnings.warn(
                "Target column not provided. The last column is considered as the target and will be renamed to 'target'.",
                UserWarning,
            )
        if metafeature_dict is None:
            metafeature_dict = {}

        # ===== Set the basic properties of the dataset =====
        self._dataset_name = dataset_name
        self._task_type = task_type
        # Rename the target column to self._target_col
        if "__" in dataset_name:
            self._target_col = self._dataset_name.split("__")[-1]
        else:
            self._target_col = "target" if target_col is None else target_col

        # ===== Load the dataset =====
        if data_df is None:
            dataset = load_tabular_dataset(self._dataset_name, self._target_col, metafeature_dict)
            data_source = dataset["data_source"]
            data_id = dataset["data_id"]
            data_dict = dataset["data_dict"]
            # === Create a pandas DataFrame from the dataset ===
            data_df = data_dict["X"]
            data_df[self._target_col] = data_dict["y"]
        else:
            data_source = "local"
            data_id = None
            # === Rename the target column to self._target_col ===
            original_target = target_col if target_col is not None else data_df.columns[-1]
            data_df = data_df.rename(columns={original_target: self._target_col})
            # === Ensure the target column is the last column ===
            if self._target_col != data_df.columns[-1]:
                column_list = list(data_df.columns)
                column_list.remove(self._target_col)
                column_list.append(self._target_col)
                data_df = data_df[column_list]
        self._data_source = data_source
        self._data_id = data_id
        # All column names are strings
        data_df.columns = data_df.columns.astype(str)

        # ===== Update the DataFrame =====
        self._data_df = data_df

        # ===== Analyse the metafeatures =====
        # Metafeatures are those consistent after subsampling/splitting
        self._metafeature_dict = self._parse_metafeatures(metafeature_dict)

        # ===== Update the dataset properties after parsing metafeatures =====
        data_df = self._data_df
        # All categorical features have string values (except NaN)
        for categorical_col in self._categorical_feature_list:
            nan_mask = self.data_df[categorical_col].isna()
            data_df[categorical_col] = self.data_df[categorical_col].astype(str)
            # pd.NA is not compatible with sklean transformations for data preprocessing
            data_df.loc[nan_mask, categorical_col] = np.nan
        # Before converting to tensor, the target column is always string for classification tasks
        if self._task_type == "classification" and not self._is_tensor:
            data_df[self._target_col] = data_df[self._target_col].astype(str)
        # Update the DataFrame again
        self._data_df = data_df

        # ===== Parse the structure information of the dataset =====
        self._structure_info_dict = self._parse_structure_info()
        self._dag_edges = self._structure_info_dict["dag_edges"]
        self._dependency_dict = self._structure_info_dict["dependency_dict"]

        # ===== Log the properties of the dataset =====
        self._update_data_df_properties()

    def _update_data_df_properties(self) -> None:
        """Parse the properties of the dataset."""
        self._num_samples = self.X_df.shape[0]
        self._num_features = self.X_df.shape[1]
        if self._task_type == "classification":
            self._num_classes = len(self.data_df[self._target_col].unique())
            self._class2samples = self.data_df[self._target_col].value_counts().to_dict()
            self._class2distribution = self.data_df[self._target_col].value_counts(normalize=True).to_dict()
        else:
            self._num_classes = None
            self._class2samples = None
            self._class2distribution = None

    def _parse_metafeatures(self, metafeature_dict: dict) -> dict:
        """Parse the metafeatures of the dataset.

        Args:
            metafeature_dict (dict): Provided dictionary from the user, which contains the metafeatures.

        Returns:
            dict: Dictionary containing the parsed full metafeatures.
        """
        # ===== Parse the metafeatures =====
        self._is_tensor = metafeature_dict.get("is_tensor", False)
        self._col2type = metafeature_dict.get("col2type", self._infer_column_types())
        self._data_source = metafeature_dict.get("source", self._data_source)
        self._data_id = metafeature_dict.get("id", self._data_id)
        # When building a dataset with give column names (e.g., from a DataFrame), the column names are consistent
        self._data_df.columns = list(self._col2type.keys())

        self._numerical_feature_list = [
            col for col, type in self.col2type.items() if (col != self._target_col and type == stype.numerical)
        ]
        self._categorical_feature_list = [
            col for col, type in self.col2type.items() if (col != self._target_col and type == stype.categorical)
        ]
        self._num_numerical_features = len(self._numerical_feature_list)
        self._num_categorical_features = len(self._categorical_feature_list)

        return {
            "is_tensor": self._is_tensor,
            "col2type": self._col2type,
            "source": self._data_source,
            "id": self._data_id,
        }

    def _parse_structure_info(self) -> pd.DataFrame:
        """Parse the structure information of the dataset.

        Returns:
            pd.DataFrame: DataFrame containing the directed acyclic graph (DAG) edges.
        """
        # ===== Parse the structure information =====
        dag_edges = None
        dependency_dict = None
        if "__" in self.dataset_name:
            dataset_name = self.dataset_name.split("__")[0]
        else:
            dataset_name = self.dataset_name

        if dataset_name in dataset_with_dag:
            # === Load the structure information of the dataset ===
            dataset_name_lower = dataset_name.lower()
            dag_edges = pd.read_csv(
                os.path.join(LOCAL_DATA_PATH, "structure", dataset_name_lower, f"{dataset_name_lower}_edges.csv"),
                dtype=str,
            )
            dependency_path = os.path.join(
                LOCAL_DATA_PATH, "structure", dataset_name_lower, f"{dataset_name_lower}_dependency.pkl"
            )
            if os.path.exists(dependency_path):
                with open(os.path.join(dependency_path), "rb") as f:
                    dependency_dict = cloudpickle.load(f)

        return {
            "dag_edges": dag_edges,
            "dependency_dict": dependency_dict,
        }

    # ================================================================
    # =                                                              =
    # =                   Sample dataset                          =
    # =                                                              =
    # ================================================================
    def subsample(
        self,
        subsample_mode: str,
        subsample_size: Optional[int | float] = None,
        subsample_indices: Optional[list[int]] = None,
        customised_class2distribution: Optional[dict] = None,
        random_state: Optional[int] = 42,
    ) -> dict:
        """Subsample the dataset.

        Args:
            num_samples (int): Number of samples to subsample.
            subsample_mode (str): Mode of subsampling (random, stratified, uniform, fixed).
            subsample_size (Optional[int | float], optional): Number of samples to subsample. If float, it represents the proportion of samples to subsample. Only applicable for unfixed subsampling.
            subsample_indices (Optional[list[int]], optional): Indices of the samples to subsample. Only applicable for fixed subsampling.
            random_state (int): Random state for reproducibility. When random_state is set, larger subsample_size will be superset of smaller sample size.

        Returns:
            dict: Dictionary containing the subsampled dataset and the indices of the subsampled data **in original dataset**.
        """
        # ===== Sanity check before subsampling =====
        # === Check if subsample size or indices are provided ===
        if subsample_mode in ["random", "stratified", "uniform"]:
            if subsample_size is None:
                raise ValueError("Subsample size must be provided for unfixed subsampling.")
            if subsample_indices is not None:
                warnings.warn("Subsample indices are provided but will be ignored.", UserWarning)
        if subsample_mode in ["customised_ratio"]:
            if customised_class2distribution is None:
                raise ValueError("Customised class distributions must be provided for customised ratio subsampling.")
            if subsample_size is None:
                raise ValueError("Subsample size must be provided for customised ratio subsampling.")
        if subsample_mode in ["fixed"]:
            if subsample_indices is None:
                raise ValueError("Subsample indices must be provided for fixed subsampling.")
            if subsample_size is not None:
                warnings.warn("Subsample size is provided but will be ignored.", UserWarning)
            subsample_size = len(subsample_indices)
        # === Check if the sampling size is valid ===
        if isinstance(subsample_size, float):
            subsample_size = int(subsample_size * self._num_samples)
        if subsample_size > self._num_samples:
            raise ValueError(
                f"Number of subsamples ({subsample_size}) is greater than the number of samples ({self._num_samples}) in the dataset."
            )
        if subsample_mode == "stratified" and subsample_size < self._num_classes:
            raise ValueError(
                f"Number of subsamples ({subsample_size}) is less than the number of classes ({self._num_classes}) when using stratified subsampling."
            )

        # ===== Random sampling =====
        if subsample_mode == "random":
            data_df = self.data_df.sample(n=subsample_size, random_state=random_state)
        # ===== Stratified sampling =====
        elif subsample_mode in ["stratified", "customised_ratio"]:
            # === Compute the number of samples per class ===
            class2distribution = (
                self._class2distribution if subsample_mode == "stratified" else customised_class2distribution
            )
            class2num_subsamples = {
                class_label: max(1, int(subsample_size * class_distribution))
                for class_label, class_distribution in class2distribution.items()
            }

            # === Align the number of samples per class ===
            i = 0
            while subsample_size != sum(class2num_subsamples.values()) and i < 1000:
                # Randomly select a class for more or fewer samples (The sampling is reproducible because the seed is fixed, different to np.random.shuffle)
                class_label = random.choice(list(class2num_subsamples.keys()))
                if subsample_size > sum(class2num_subsamples.values()):
                    # Increase the number of samples for the selected class only if it is less than the total number of samples
                    class2num_subsamples[class_label] = min(
                        class2num_subsamples[class_label] + 1, self._class2samples[class_label]
                    )
                else:
                    # Decrease the number of samples for the selected class only if it is more than 1
                    class2num_subsamples[class_label] = max(class2num_subsamples[class_label] - 1, 1)
                i += 1

            if subsample_size != sum(class2num_subsamples.values()):
                raise ValueError("The number of samples per class cannot be aligned.")

            # === Subsample the dataset ===
            # When random_state is set, larger subsample_size will be superset of smaller sample size
            data_df = self.data_df.groupby(self._target_col, observed=True, group_keys=False).apply(
                lambda x: x.sample(n=class2num_subsamples[x[self._target_col].iloc[0]], random_state=random_state)
            )
        # ===== Uniform sampling =====
        elif subsample_mode == "uniform":
            # === Check if the number of samples is divisible by the number of classes ===
            if subsample_size % self._num_classes != 0:
                raise ValueError(
                    f"Number of samples {subsample_size} is not divisible by the number of classes {self._num_classes}."
                )

            # === Check if the number of samples per class is less than the minimum number of samples per class ===
            num_samples_per_class = subsample_size // self._num_classes
            if num_samples_per_class > min(self._class2samples.values()):
                raise ValueError(
                    f"Number of samples per class ({num_samples_per_class}) is greater than the minimum number of samples per class ({min(self._class2samples.values())})."
                )

            # === Subsample the dataset ===
            # When random_state is set, larger subsample_size will be superset of smaller sample size
            data_df = self.data_df.groupby(self._target_col, observed=True, group_keys=False).apply(
                lambda x: x.sample(n=num_samples_per_class, random_state=random_state)
            )
        # ===== Fixed sampling =====
        elif subsample_mode == "fixed":
            data_df = self.data_df.loc[subsample_indices]
        # ===== Error =====
        else:
            raise NotImplementedError(f"Subsample mode {subsample_mode} not recognised.")

        # ===== Format the subsampled data =====
        # Record the indices of the subsampled data (the indices of the original full dataset)
        subsample_indices = data_df.index.tolist()

        return {
            "dataset_subsample": TabularDataset(
                dataset_name=self._dataset_name,
                task_type=self._task_type,
                target_col=self._target_col,
                metafeature_dict=self._metafeature_dict,
                data_df=data_df,
            ),
            "subsample_indices": subsample_indices,
        }

    def sample(
        self,
        sample_mode: str,
        sample_size: Optional[int | float] = None,
        sample_indices: Optional[list[int]] = None,
        customised_class2distribution: Optional[dict] = None,
        random_state: Optional[int] = 42,
    ) -> dict:
        """Sample the dataset.

        Args:
            sample_mode (str): Mode of sampling (random, stratified, uniform, fixed).
            sample_size (Optional[int | float], optional): Number of samples to sample. If float, it represents the proportion of samples to sample. Only applicable for unfixed sampling.
            sample_indices (Optional[list[int]], optional): Indices of the samples to sample. Only applicable for fixed sampling.
            customised_class2distribution (Optional[dict], optional): Customised class distributions. Only applicable for customised ratio sampling.
            random_state (int): Random state for reproducibility. When random_state is set, larger sample_size will be superset of smaller sample size.

        Returns:
            dict: Dictionary containing the sampled dataset and the indices of the sampled data **in original dataset**.
        """
        # ===== Sanity check before subsampling =====
        # === Check if sample size or indices are provided ===
        if sample_mode in ["random", "stratified", "uniform"]:
            if sample_size is None:
                raise ValueError("Sample size must be provided for unfixed ampling.")
            if sample_indices is not None:
                warnings.warn("Sample indices are provided but will be ignored.", UserWarning)
        if sample_mode in ["customised_ratio"]:
            if customised_class2distribution is None:
                raise ValueError("Customised class distributions must be provided for customised ratio sampling.")
            if sample_size is None:
                raise ValueError("Sample size must be provided for customised ratio sampling.")
        if sample_mode in ["fixed"]:
            if sample_indices is None:
                raise ValueError("Sample indices must be provided for fixed sampling.")
            if sample_size is not None:
                warnings.warn("Sample size is provided but will be ignored.", UserWarning)
            sample_size = len(sample_indices)
        # === Check if the sampling size is valid ===
        if isinstance(sample_size, float):
            sample_size = int(sample_size * self._num_samples)
        if sample_mode == "stratified" and sample_size < self._num_classes:
            raise ValueError(
                f"Number of samples ({sample_size}) is less than the number of classes ({self._num_classes}) when using stratified subsampling."
            )

        # ===== Random sampling =====
        if sample_mode == "random":
            data_df = self.data_df.sample(n=sample_size, random_state=random_state, replace=True)
        # ===== Stratified sampling =====
        elif sample_mode in ["stratified", "customised_ratio"]:
            # === Compute the number of samples per class ===
            class2distribution = (
                self._class2distribution if sample_mode == "stratified" else customised_class2distribution
            )
            class2num_samples = {
                class_label: max(1, int(sample_size * class_distribution))
                for class_label, class_distribution in class2distribution.items()
            }

            # === Align the number of samples per class ===
            i = 0
            while sample_size != sum(class2num_samples.values()) and i < 1000:
                # Randomly select a class for more or fewer samples (The sampling is reproducible because the seed is fixed, different to np.random.shuffle)
                class_label = random.choice(list(class2num_samples.keys()))
                if sample_size > sum(class2num_samples.values()):
                    # Increase the number of samples for the selected class only if it is less than the total number of samples
                    class2num_samples[class_label] = min(
                        class2num_samples[class_label] + 1, self._class2samples[class_label]
                    )
                else:
                    # Decrease the number of samples for the selected class only if it is more than 1
                    class2num_samples[class_label] = max(class2num_samples[class_label] - 1, 1)
                i += 1

            if sample_size != sum(class2num_samples.values()):
                raise ValueError("The number of samples per class cannot be aligned.")

            # === Sample the dataset ===
            # When random_state is set, larger sample_size will be superset of smaller sample size
            data_df = self.data_df.groupby(self._target_col, observed=True, group_keys=False).apply(
                lambda x: x.sample(
                    n=class2num_samples[x[self._target_col].iloc[0]], random_state=random_state, replace=True
                )
            )
        # ===== Uniform sampling =====
        elif sample_mode == "uniform":
            # === Check if the number of samples is divisible by the number of classes ===
            if sample_size % self._num_classes != 0:
                raise ValueError(
                    f"Number of samples {sample_size} is not divisible by the number of classes {self._num_classes}."
                )

            # === Check if the number of samples per class is less than the minimum number of samples per class ===
            num_samples_per_class = sample_size // self._num_classes
            if num_samples_per_class > min(self._class2samples.values()):
                raise ValueError(
                    f"Number of samples per class ({num_samples_per_class}) is greater than the minimum number of samples per class ({min(self._class2samples.values())})."
                )

            # === Sample the dataset ===
            # When random_state is set, larger sample_size will be superset of smaller sample size
            data_df = self.data_df.groupby(self._target_col, observed=True, group_keys=False).apply(
                lambda x: x.sample(n=num_samples_per_class, random_state=random_state, replace=True)
            )
        # ===== Fixed sampling =====
        elif sample_mode == "fixed":
            data_df = self.data_df.loc[sample_indices]
        # ===== Error =====
        else:
            raise NotImplementedError(f"Sample mode {sample_mode} not recognised.")

        # ===== Format the sampled data =====
        # Record the indices of the sampled data (the indices of the original full dataset)
        sample_indices = data_df.index.tolist()

        return {
            "dataset_sampled": TabularDataset(
                dataset_name=self._dataset_name,
                task_type=self._task_type,
                target_col=self._target_col,
                metafeature_dict=self._metafeature_dict,
                data_df=data_df,
            ),
            "sample_indices": sample_indices,
        }

    # ================================================================
    # =                                                              =
    # =                     Split dataset                            =
    # =                                                              =
    # ================================================================
    def split(
        self,
        split_mode: str,
        train_size: Optional[float | int] = None,
        test_size: Optional[float | int] = None,
        indices_train: Optional[list[int]] = None,
        indices_test: Optional[list[int]] = None,
        random_state: Optional[int] = 42,
    ) -> dict:
        """Split the dataset into training and testing sets.

        Args:
            split_mode (str): Mode of splitting (random, stratified, fixed).
            train_size (Optional[int | float], optional): Same as the train_size parameter in sklearn's train_test_split.
            test_size (Optional[int | float], optional): Same as the test_size parameter in sklearn's train_test_split.
            indices_train (Optional[ist[int]], optional): Indices of the training set (The index is consistent with the original dataset). Only applicable for fixed split.
            indices_test (Optional[list[int]], optional): Indices of the testing set (The index is consistent with the original dataset). Only applicable for fixed split.
            random_state (Optional[int], optional): Random state for reproducibility.

        Returns:
            dict: Dictionary containing the training and testing sets and the indices of the training and testing sets **in original dataset**.
        """
        # ===== Sanity check before splitting =====
        if split_mode in ["random", "stratified"]:
            if train_size is None and test_size is None:
                raise ValueError("Either train size or test size must be provided for unfixed split.")
        elif split_mode == "fixed":
            if indices_train is None or indices_test is None:
                raise ValueError("Train and test indices must be provided for fixed split.")
            if not set(indices_train).isdisjoint(set(indices_test)):
                raise ValueError("Train and test indices must be disjoint.")
            if not set(indices_train).issubset(set(self.data_indices)):
                raise ValueError("Train indices must be a subset of the dataset indices.")
            if not set(indices_test).issubset(set(self.data_indices)):
                raise ValueError("Test indices must be a subset of the dataset indices.")

        # ===== Split the dataset without preset indices =====
        if split_mode in ["random", "stratified"]:
            X = self.data_df.drop(self._target_col, axis=1)
            y = self.data_df[self._target_col]
            # IMPORTANT: stratify does not gurantee all classes are icluded in both train and test set
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=train_size,
                test_size=test_size,
                random_state=random_state,
                stratify=y if split_mode == "stratified" else None,
            )
            train_df = X_train
            train_df[self._target_col] = y_train
            test_df = X_test
            test_df[self._target_col] = y_test

            # === Check if all classes are included in both train and test set ===
            if split_mode == "stratified":
                classes_full = set(y.unique())
                classes_in_train = set(y_train.unique())
                classes_in_test = set(y_test.unique())
                if classes_full != classes_in_train:
                    classes_diff = classes_full - classes_in_train
                    raise ValueError(f"Training set does not include all classes, missing: {classes_diff}.")
                if classes_full != classes_in_test:
                    classes_diff = classes_full - classes_in_test
                    raise ValueError(f"Test set does not include all classes, missing: {classes_diff}.")
        # ===== Split the dataset with preset indices =====
        elif split_mode == "fixed":
            train_df = self.data_df.loc[indices_train]
            test_df = self.data_df.loc[indices_test]

        # ===== Format the data splits =====
        # Record the indices of the sampled data (the indices of the original full dataset)
        indices_train = train_df.index.tolist()
        indices_test = test_df.index.tolist()

        return {
            "train_set": TabularDataset(
                dataset_name=self._dataset_name,
                task_type=self._task_type,
                metafeature_dict=self._metafeature_dict,
                data_df=train_df,
                target_col=self._target_col,
            ),
            "test_set": TabularDataset(
                dataset_name=self._dataset_name,
                task_type=self._task_type,
                metafeature_dict=self._metafeature_dict,
                data_df=test_df,
                target_col=self._target_col,
            ),
            "indices_train": indices_train,
            "indices_test": indices_test,
        }

    # ================================================================
    # =                                                              =
    # =                    Helper functions                          =
    # =                                                              =
    # ================================================================
    def _infer_column_types(self) -> dict:
        """Infer the column types of the dataset.

        Returns:
            dict: Dictionary containing the inferred column types.
        """
        col2type = {}
        for col in self.X_df.columns:
            # ptypes.is_numeric_dtype treats boolean as numerical, which is not desired
            if ptypes.is_numeric_dtype(self.X_df[col]) and ptypes.infer_dtype(self.X_df[col]) != "boolean":
                col2type[col] = stype.numerical
            else:
                col2type[col] = stype.categorical

        if self.task_type == "classification":
            col2type[self.target_col] = stype.categorical
        else:
            col2type[self.target_col] = stype.numerical

        return col2type

    def drop_low_sample_class(self, min_sample_per_class: int | None) -> None:
        """Drop classes with fewer samples than the threshold (excluded).

        Args:
            min_sample_per_class (int | None): Threshold for the minimum number of samples.
        """
        if min_sample_per_class is None:
            return
        for class_id, num_samples in self._class2samples.items():
            if num_samples < min_sample_per_class:
                self.drop_class(class_id)

    def drop_class(self, class_id: str | None) -> None:
        """Drop a class from the dataset.

        Args:
            class_id (str | None): ID of the class to drop.
        """
        # ===== Sanity check before dropping class =====
        if class_id is None:
            return
        if self.is_tensor:
            raise ValueError("Cannot drop class after converting the dataset to tensor.")
        if self._task_type != "classification":
            raise ValueError("Cannot drop class for regression task.")
        if class_id not in self.data_df[self._target_col].unique():
            raise ValueError(f"Class {class_id} not found in the dataset.")

        # ===== Drop the class =====
        data_df = self.data_df[self.data_df[self._target_col] != class_id]
        self.data_df = data_df

    @staticmethod
    def drop_constant_feature(dataset: TabularDataset) -> TabularDataset:
        """Drop constant features from the dataset."""
        # === Drop constant features ===
        # As `nunique()` does not count nan by default, we also drop the columns with a single value + nan
        drop_col_list = dataset.X_df.columns[dataset.X_df.nunique() == 1].tolist()
        X_df = dataset.X_df.drop(drop_col_list, axis=1)
        metafeature_dict = deepcopy(dataset.metafeature_dict)
        for col in drop_col_list:
            metafeature_dict["col2type"].pop(col, None)

        # === For clearance, return a new TabularDataset instance ===
        return TabularDataset(
            dataset_name=dataset.dataset_name,
            task_type=dataset.task_type,
            target_col=dataset.target_col,
            metafeature_dict=metafeature_dict,
            data_df=pd.concat([X_df, dataset.y_s], axis=1),
        )

    # ================================================================
    # =                                                              =
    # =                  Dataset properties                          =
    # =                                                              =
    # ================================================================
    # ===================== Read-only properties =====================
    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset.

        Returns:
            str: Name of the dataset.
        """
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name: str) -> None:
        """Set the name of the dataset.

        Args:
            dataset_name (str): Name of the dataset.
        """
        self._dataset_name = dataset_name

    @property
    def task_type(self) -> str:
        """Return the type of task.

        Returns:
            str: Type of task (classification or regression).
        """
        return self._task_type

    @property
    def target_col(self) -> str:
        """Return the name of the target column.

        Returns:
            str: Name of the target column.
        """
        return self._target_col

    @property
    def X_df(self) -> pd.DataFrame:
        """Return the DataFrame containing the features.

        Returns:
            pd.DataFrame: DataFrame containing the features.
        """
        return self.data_df.drop(self._target_col, axis=1)

    @property
    def y_s(self) -> pd.Series:
        """Return the Series containing the target.

        Returns:
            pd.Series: Series containing the target.
        """
        return self.data_df[self._target_col]

    @property
    def data_indices(self) -> list:
        """Return the indices of the samples in the dataset.

        Returns:
            list: Indices of the samples in the dataset.
        """
        return self.data_df.index.tolist()

    @property
    def num_samples(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self._num_samples

    @property
    def num_features(self) -> int:
        """Return the number of features in the dataset.

        Returns:
            int: Number of features in the dataset.
        """
        return self._num_features

    @property
    def num_classes(self) -> int | None:
        """Return the number of classes in the dataset.

        Returns:
            int | None: Number of classes in the dataset.
        """
        return self._num_classes

    @property
    def class_list(self) -> list | None:
        """Return the list of classes in the dataset.

        Returns:
            list | None: List of classes in the dataset.
        """
        return list(self._class2distribution.keys()) if self._class2distribution is not None else None

    @property
    def class2samples(self) -> dict | None:
        """Return the numbers per class in the dataset.

        Returns:
            dict | None: Class-wise sample count in the dataset.
        """
        return self._class2samples

    @property
    def class2distribution(self) -> dict | None:
        """Return the class distribution in the dataset.

        Returns:
            dict | None: Class distribution in the dataset.
        """
        return self._class2distribution

    @property
    def col2type(self) -> dict:
        """Return the dictionary containing the column types.

        Returns:
            dict: Dictionary containing the column types.
        """
        return self._col2type

    @property
    def metafeature_dict(self) -> dict:
        """Return the dictionary containing the metafeatures.

        Returns:
            dict: Dictionary containing the metafeatures.
        """
        return self._metafeature_dict

    @property
    def numerical_feature_list(self) -> list:
        """Return the list of numerical features.

        Returns:
            list: List of numerical features.
        """
        return self._numerical_feature_list

    @property
    def categorical_feature_list(self) -> list:
        """Return the list of categorical features.

        Returns:
            list: List of categorical features.
        """
        return self._categorical_feature_list

    @property
    def dag_edges(self) -> pd.DataFrame:
        """Return the DataFrame containing the directed acyclic graph (DAG) edges.

        Returns:
            pd.DataFrame: DataFrame containing the directed acyclic graph (DAG) edges.
        """
        return self._dag_edges

    @property
    def dependency_dict(self) -> dict:
        """Return the dictionary containing the dependencies.

        Returns:
            dict: Dictionary containing the dependencies.
        """
        return self._dependency_dict

    # ===================== Read-write properties =====================
    @property
    def data_df(self) -> pd.DataFrame:
        """Return the DataFrame containing the features and target.

        Returns:
            pd.DataFrame: DataFrame containing the features and target.
        """
        return self._data_df

    @data_df.setter
    def data_df(self, data_df: pd.DataFrame) -> None:
        """Set the DataFrame containing the features and target.

        Args:
            data_df (pd.DataFrame): DataFrame containing the features and target.

        Raises:
            ValueError: If the column names are not the same as the original DataFrame.
        """
        # ===== Sanity check for the DataFrame =====
        # === Column names and order must be the same as the original DataFrame ===
        if not data_df.columns.equals(self.data_df.columns):
            print(data_df.columns)
            print(self.data_df.columns)
            raise ValueError("Column names must be the same as the original DataFrame.")

        # ===== Update the DataFrame =====
        self._data_df = data_df

        # ===== Analyses the DataFrame-specific properties =====
        self._update_data_df_properties()

    @property
    def is_tensor(self) -> bool:
        """Return whether the dataframe has been materialised into tensor.

        Returns:
            bool: Whether the dataframe is tensor.
        """
        return self._is_tensor

    @is_tensor.setter
    def is_tensor(self, is_tensor: bool) -> None:
        """Set whether the dataframe has been materialised into tensor.

        Args:
            is_tensor (bool): Whether the dataframe is tensor.
        """
        self._is_tensor = is_tensor
        self._metafeature_dict["is_tensor"] = is_tensor

    # ================================================================
    # =                                                              =
    # =                     Present dataset                          =
    # =                                                              =
    # ================================================================
    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self._num_samples

    def __repr__(self) -> str:
        """Return the string representation of the TabularDataset.

        Returns:
            str: String representation of the TabularDataset.
        """
        return f"TabularDataset(dataset_name={self.dataset_name}, task_type={self.task_type}, target_col={self.target_col}, is_tensor={self.is_tensor})"

    def __str__(self) -> str:
        """Return the printing representation of the TabularDataset.

        Returns:
            str: Printing representation of the TabularDataset.
        """
        data_info = "============================DATA INFO============================\n"
        data_info += f"Dataset: {self._dataset_name}\n"
        data_info += f"Task type: {self._task_type}\n"
        data_info += f"Status (is_tensor): {self._is_tensor}\n"
        data_info += f"Number of samples: {self._num_samples}\n"
        data_info += f"Number of features: {self._num_features} (Numerical: {self._num_numerical_features}, Categorical: {self._num_categorical_features})\n"
        data_info += f"Number of classes: {self._num_classes}\n"
        data_info += f"Class distribution: {self._class2distribution}\n"
        data_info += "================================================================="
        return data_info

    @property
    def info_df(self) -> pd.DataFrame:
        """Return the DataFrame containing the dataset information.

        Returns:
            pd.DataFrame: DataFrame containing the dataset information.
        """
        dataset = self.dataset_name
        num_samples = self.num_samples
        num_features = self.num_features
        num_numerical = len(self.numerical_feature_list)
        num_categorical = len(self.categorical_feature_list)
        num_classes = self.num_classes
        samples_per_class = list(self.class2samples.values())

        return pd.DataFrame(
            {
                "Dataset": dataset,
                "Source": self._data_source,
                "ID": self._data_id,
                "# Samples ($N$)": f"{num_samples:,}",
                "# Features ($D$)": f"{num_features:,}",
                "$N/D$": f"{num_samples/num_features: .2f}",
                "# Numerical": f"{num_numerical:,}",
                "# Categorical": f"{num_categorical:,}",
                "# Classes": num_classes if self.task_type == "classification" else "N/A",
                "# Samples per class (Min)": f"{min(samples_per_class):,}",
                "# Samples per class (Max)": f"{max(samples_per_class):,}",
                "# Samples per class": str(samples_per_class) if self.task_type == "classification" else "N/A",
            },
            index=[0],
        )
