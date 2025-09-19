import numpy as np
import pandas as pd
from camel.data.transform import CategoryTransform, SimpleImputeTransform
from sdmetrics.reports.single_table import QualityReport
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from synthcity.metrics.eval_causality import StructuralHammingDistance
from synthcity.metrics.eval_privacy import DeltaPresence, IdentifiabilityScore
from synthcity.metrics.eval_sanity import (CloseValuesProbability, CommonRowsProportion, DataMismatchScore,
                                           DistantValuesProbability, NearestSyntheticNeighborDistance)
from synthcity.metrics.eval_statistical import (AlphaPrecision, ChiSquaredTest, InverseKLDivergence,
                                                JensenShannonDistance, KolmogorovSmirnovTest, PRDCScore,
                                                WassersteinDistance)
from synthcity.metrics.eval_structure import CITest, UtilityPerFeature
from synthcity.plugins.core.dataloader import GenericDataLoader
from tqdm import tqdm

import wandb
from src.common.data.DataHelper import DataHelper
from src.common.runtime.error.ManualStopError import ManualStopError


# ================================================================
# =                                                              =
# =                   Data operations                            =
# =                                                              =
# ================================================================
def compute_all_metrics(args, data_module, generation_dict) -> dict:
    metric_dict = {}

    pbar = tqdm(["train", "valid", "test"])
    for split in pbar:
        pbar.set_description(f"Evaluating on {split} split")
        temp_metric_dict = {}

        # === Prepare the data ===
        # Recover the original data for real samples
        # Note that imputation is not allowed for evaluating synthetic data with SDV, and thus we cannot use `recover_original_data`
        # Instead, we directly use the loaded synthetic data without any imputation
        original_data_dict = DataHelper.recover_original_data(
            args,
            getattr(data_module, f"X_{split}"),
            getattr(data_module, f"y_{split}"),
        )
        # Use real training samples when evaluating the "real" generator
        if args.model == "real" and split == "train":
            generation_dict = {
                "X_syn": getattr(data_module, "X_train"),
                "y_syn": getattr(data_module, "y_train"),
                "X_syn_original": original_data_dict["X_original"],
                "y_syn_original": original_data_dict["y_original"],
            }

        # === Compute SDV metrics ===
        temp_metric_dict |= compute_sdv_metrics(
            args,
            original_data_dict["X_original"],
            original_data_dict["y_original"],
            generation_dict["X_syn_original"],
            generation_dict["y_syn_original"],
        )

        # Compute Synthcity metrics
        temp_metric_dict |= compute_synthcity_metrics(
            args,
            getattr(data_module, f"X_{split}"),
            getattr(data_module, f"y_{split}"),
            generation_dict["X_syn"],
            generation_dict["y_syn"],
            dependency_dict=data_module.dependency_dict,
            split=split,
        )

        # Log metrics
        for metric, value in temp_metric_dict.items():
            wandb.run.summary[f"{split}_metrics/{metric}"] = value

        metric_dict[f"{split}_metrics"] = temp_metric_dict

    return metric_dict


def compute_sdv_metrics(
    args,
    X_real_original: pd.DataFrame,
    y_real_original: pd.DataFrame,
    X_syn_original: pd.DataFrame,
    y_syn_original: pd.DataFrame,
) -> dict:

    metric_dict = {}

    if not args.eval_stats:
        return metric_dict

    # === Prepare the data ===
    real_data = pd.concat([X_real_original, y_real_original], axis=1)
    synthetic_data = pd.concat([X_syn_original, y_syn_original], axis=1)
    metadata = {
        "columns": {col: args.full_feature_col2type_original[col].sdmetrics_dtype for col in real_data.columns},
    }

    # === Compute the metrics ===
    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata, verbose=False)
    metric_dict = report.get_properties().set_index("Property").to_dict()["Score"]

    return metric_dict


def compute_synthcity_metrics(
    args,
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    dependency_dict: dict,
    split: str,
) -> dict:
    # === Preprocess the data ===
    loader_dict_real = prepare_loader_for_eval(args, X_real, y_real)
    loader_dict_syn = prepare_loader_for_eval(args, X_syn, y_syn)

    # === Initialize the metric dictionary ===
    # All sanity and stat metrics are computed with one-hot encoded features + labetls
    # See prior work: https://github.com/amazon-science/tabsyn/tree/main/eval
    metric_dict = {}

    # === Sanity check ===
    # metric_dict |= compute_sanity_check_metrics(args, loader_dict_real, loader_dict_syn)

    # === Statistical metrics ===
    if args.eval_stats:
        metric_dict |= compute_statistical_metrics(args, loader_dict_real, loader_dict_syn)

    # === Privacy metrics ===
    if args.eval_privacy:
        metric_dict |= compute_privacy_metrics(args, loader_dict_real, loader_dict_syn)

    # === Structure learning metrics ===
    if args.eval_structure and split == "test":
        # Evaluate the structure learning metrics when the GT dependency dictionary is available
        if dependency_dict is not None:
            metric_dict |= compute_structure_metrics(args, loader_dict_real, loader_dict_syn, dependency_dict)
        # Evaluate the proxy structure learning metrics
        metric_dict |= compute_proxy_structure_metrics(args, loader_dict_real, loader_dict_syn)

    # === Causality metrics ===
    if args.eval_causality:
        metric_dict |= compute_causality_metrics(args, loader_dict_real, loader_dict_syn)

    return metric_dict


def prepare_loader_for_eval(args, X: np.ndarray, y: np.ndarray) -> dict:
    # === Build Dataloader for group 1 ===
    data = pd.DataFrame(X, columns=args.full_feature_col_list_processed)
    data[args.full_target_col_processed] = y
    loader_all = GenericDataLoader(data=data, target_column=args.full_target_col_processed)
    loader_cat = loader_all.drop(columns=args.num_feature_col_list_processed)
    loader_num = loader_all.drop(columns=args.cat_feature_col_list_processed)
    # For classification, locader_num contains the target column, as target column not in cat_feature_col_list_processed
    if args.task == "classification":
        loader_num = loader_num.drop(columns=[args.full_target_col_processed])
    # For regression, loader_cat contains the target column, as target column not in num_feature_col_list_processed
    elif args.task == "regression":
        loader_cat = loader_cat.drop(columns=[args.full_target_col_processed])

    # === Build Dataloader for group 2 ===
    loader_all_onehot = loader_all
    loader_cat_onehot = loader_cat
    if args.task == "classification":
        data = pd.DataFrame(X, columns=args.full_feature_col_list_processed)
        onehot_encoder = OneHotEncoder(sparse_output=False, categories=[args.class_encoded_list]).fit(y.reshape(-1, 1))
        y_onehot = onehot_encoder.transform(y.reshape(-1, 1))
        y_onehot_df = pd.DataFrame(
            y_onehot, columns=onehot_encoder.get_feature_names_out([args.full_target_col_processed])
        )
        data = pd.concat([data, y_onehot_df], axis=1)
        loader_all_onehot = GenericDataLoader(data=data, target_column=None)
        loader_cat_onehot = loader_all_onehot.drop(columns=args.num_feature_col_list_processed)

    # === Build Dataloader for group 3 ===
    loader_cat_ordinal = loader_cat
    X_df = pd.DataFrame(X, columns=args.full_feature_col_list_processed)
    for scaler_idx, feature_scaler in enumerate(args.feature_scaler_list[::-1]):
        X_df = feature_scaler.inverse_transform(X_df)
        if not isinstance(feature_scaler, CategoryTransform):
            continue
        # Imputation does not gurantee legal onehot values, so inverse transform of CategoryTransform can generate NaNs.
        # Thus, we need to apply the imputation again after the inverse transform of CategoryTransform
        for i in range(scaler_idx):
            if isinstance(args.feature_scaler_list[i], SimpleImputeTransform):
                X_df = args.feature_scaler_list[i].transform(X_df)
                break
        """ `OrdinalEncoder` can automatically infer the categories when `categories` is provided
        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder(categories=[["Male", "Female"], ["1", "2", "3", "4"]])
        X = [["Male", "2"], ["Female", "3"], ["Female", "3"]]
        print(enc.fit(X))
        print(enc.categories_)
        print(enc.transform([["Female", "1"], ["Male", "4"]]))
        """
        # The data has been reversed to the original data, so we need to drop the original numerical features
        X_df_cat = X_df.drop(args.num_feature_col_list_original, axis=1)
        try:
            data_cat = OrdinalEncoder(
                categories=[list(cat) for cat in feature_scaler.categories_],
            ).fit_transform(X_df_cat)
        except Exception as e:
            # In very rare cases (mainly due to splitting cause some features with incomplete categories),
            # the category scaler fitted on train may generate illegal onehot values (e.g., all zeros).
            # In these cases, we need to impute the inverse transformed data with original categories (e.g., str).
            raise ManualStopError(f"OrdinalEncoder failed with error: {e}")
        data_cat = pd.DataFrame(data_cat, columns=X_df_cat.columns)
        if args.task == "classification":
            data_cat[args.full_target_col_processed] = y
        loader_cat_ordinal = GenericDataLoader(data=data_cat, target_column=None)
        break
    data_all_ordinal_df = pd.concat([loader_num.dataframe(), loader_cat_ordinal.dataframe()], axis=1)
    col_list_sorted = args.full_feature_col_list_original
    col_list_sorted = col_list_sorted + [args.full_target_col_original]
    data_all_ordinal_df = data_all_ordinal_df[col_list_sorted]
    loader_all_cardinal = GenericDataLoader(data=data_all_ordinal_df, target_column=args.full_target_col_processed)

    loader_dict = {
        # Group 1: Default to one-hot encoded features + ordinal labels (if applicable)
        "all": loader_all,
        "cat": loader_cat,  # includes labels for classification
        "num": loader_num,  # includes labels for regression
        # Group 2: one-hot encoded categorical features + labels
        "all_onehot": loader_all_onehot,
        "cat_onehot": loader_cat_onehot,
        # Group 3: ordinal categorical features + labels
        "all_ordinal": loader_all_cardinal,
        "cat_ordinal": loader_cat_ordinal,
    }

    return loader_dict


# ================================================================
# =                                                              =
# =                   Eval dimensions                            =
# =                                                              =
# ================================================================
def compute_sanity_check_metrics(args, loader_dict_real: dict, loader_dict_syn: dict) -> dict:
    sanity_metric_list = [
        DataMismatchScore,
        CommonRowsProportion,
        CloseValuesProbability,
        DistantValuesProbability,
    ]
    sanity_dict = {}

    for sanity_metric in sanity_metric_list:
        if sanity_metric in [DataMismatchScore, CommonRowsProportion, CloseValuesProbability, DistantValuesProbability]:
            metric_name = sanity_metric.name()
            res = sanity_metric().evaluate(loader_dict_real["all_onehot"], loader_dict_syn["all_onehot"])["score"]
        else:
            raise NotImplementedError(f"Sanity metric {sanity_metric} is not implemented.")

        sanity_dict[f"sanity_{metric_name}"] = res

    return sanity_dict


def compute_statistical_metrics(args, loader_dict_real: dict, loader_dict_syn: dict) -> dict:
    stat_metric_list = [
        # # Categorical features (https://arize.com/blog-course/jensen-shannon-divergence/)
        # InverseKLDivergence,
        # JensenShannonDistance,
        # ChiSquaredTest,
        # # Numerical features
        # KolmogorovSmirnovTest,
        # # Mixed features
        # WassersteinDistance,
        # PRDCScore,
        AlphaPrecision,
    ]
    stat_dict = {}

    for stat_metric in stat_metric_list:
        if stat_metric in [InverseKLDivergence, ChiSquaredTest, JensenShannonDistance]:
            metric_name = f"cat_{stat_metric.name()}"
            res = stat_metric().evaluate(loader_dict_real["cat_onehot"], loader_dict_syn["cat_onehot"])
        elif stat_metric in [KolmogorovSmirnovTest]:
            metric_name = f"num_{stat_metric.name()}"
            res = stat_metric().evaluate(loader_dict_real["num"], loader_dict_syn["num"])
        elif stat_metric in [WassersteinDistance, PRDCScore, AlphaPrecision]:
            metric_name = f"mixed_{stat_metric.name()}"
            res = stat_metric().evaluate(loader_dict_real["all_onehot"], loader_dict_syn["all_onehot"])
        else:
            raise NotImplementedError(f"Statistical metric {stat_metric} is not implemented.")

        if isinstance(res, dict):
            for key, value in res.items():
                stat_dict[f"stat_{metric_name}_{key}"] = value
        else:
            stat_dict[f"stat_{metric_name}"] = res

    return stat_dict


def compute_privacy_metrics(args, loader_dict_real: dict, loader_dict_syn: dict) -> dict:
    privacy_metric_list = [
        NearestSyntheticNeighborDistance,
        DeltaPresence,
        # IdentifiabilityScore,
    ]
    privacy_dict = {}

    for privacy_metric in privacy_metric_list:
        if privacy_metric in [NearestSyntheticNeighborDistance]:
            reduction = "median"
            metric_name = f"{privacy_metric.name()}_{reduction}"
            # Reverse the order of the real and synthetic data to get distance of synthetic -> real
            res = privacy_metric(reduction=reduction).evaluate(
                loader_dict_syn["all_onehot"], loader_dict_real["all_onehot"]
            )[reduction]
        elif privacy_metric in [DeltaPresence]:
            metric_name = privacy_metric.name()
            res = privacy_metric().evaluate(loader_dict_real["all_onehot"], loader_dict_syn["all_onehot"])["score"]
        elif privacy_metric in [IdentifiabilityScore]:
            metric_name = privacy_metric.name()
            res = privacy_metric().evaluate(loader_dict_real["all_onehot"], loader_dict_syn["all_onehot"])
        else:
            raise NotImplementedError(f"Privacy metric {privacy_metric} is not implemented.")

        if isinstance(res, dict):
            for key, value in res.items():
                privacy_dict[f"privacy_{metric_name}_{key}"] = value
        else:
            privacy_dict[f"privacy_{metric_name}"] = res

    return privacy_dict


def compute_structure_metrics(
    args,
    loader_dict_real: dict,
    loader_dict_syn: dict,
    dependency_dict: dict,
) -> dict:
    structure_metric_dict = {
        CITest,
    }
    structure_dict = {}

    for structure_metric in structure_metric_dict:
        if structure_metric in [CITest]:
            metric_name = structure_metric.name()
            # Determine the CI test method based on the data type
            col2type = set(args.full_feature_col2type_original.values())
            if len(col2type) == 2:
                test_method = "pillai"  # Mixed data
            elif args.task == "classification":
                test_method = "chi_square"  # Categorical data
            else:
                test_method = "pearsonr"  # Numerical data

            # Evaluate the metric
            res = structure_metric().evaluate(
                loader_dict_real["all_ordinal"],
                loader_dict_syn["all_ordinal"],
                column_list=args.full_feature_col_list_original + [args.full_target_col_original],
                dependency_dict=dependency_dict,
                test_method=test_method,
                max_ratio_ci_test=1,
            )
        else:
            raise NotImplementedError(f"Structure metric {structure_metric} is not implemented.")

        if isinstance(res, dict):
            for key, value in res.items():
                if isinstance(value, dict):
                    wandb.log({f"structure_{metric_name}_{key}": wandb.Table(dataframe=pd.DataFrame(value))})
                structure_dict[f"structure_{metric_name}_{key}"] = value
        else:
            structure_dict[f"structure_{metric_name}"] = res

    return structure_dict


def compute_proxy_structure_metrics(
    args,
    loader_dict_real: dict,
    loader_dict_syn: dict,
) -> dict:
    proxy_structure_metric_list = [
        UtilityPerFeature,
    ]
    proxy_structure_dict = {}

    for proxy_structure_metric in proxy_structure_metric_list:
        if proxy_structure_metric in [UtilityPerFeature]:
            metric_name = proxy_structure_metric.name()
            column_list = args.full_feature_col_list_original + [args.full_target_col_original]
            res = proxy_structure_metric().evaluate(
                loader_dict_real["all_ordinal"],
                loader_dict_syn["all_ordinal"],
                column_list=column_list,
                time_limit=int(900 // len(column_list)),  # 15 minutes max in total
            )
        else:
            raise NotImplementedError(f"Proxy structure metric {proxy_structure_metric} is not implemented.")

        if isinstance(res, dict):
            for key, value in res.items():
                if isinstance(value, dict):
                    wandb.log({f"proxy_structure_{metric_name}_{key}": wandb.Table(dataframe=pd.DataFrame(value))})
                proxy_structure_dict[f"proxy_structure_{metric_name}_{key}"] = value
        else:
            proxy_structure_dict[f"proxy_structure_{metric_name}"] = res

    return proxy_structure_dict


def compute_causality_metrics(args, loader_dict_real: dict, loader_dict_syn: dict) -> dict:
    causality_metric_dict = {StructuralHammingDistance}
    causality_dict = {}

    for causality_metric in causality_metric_dict:
        if causality_metric in [StructuralHammingDistance]:
            metric_name = causality_metric.name()
            res = causality_metric().evaluate(loader_dict_real["all_ordinal"], loader_dict_syn["all_ordinal"])
        else:
            raise NotImplementedError(f"Causality metric {causality_metric} is not implemented.")

        if isinstance(res, dict):
            for key, value in res.items():
                causality_dict[f"causality_{metric_name}_{key}"] = value
        else:
            causality_dict[f"causality_{metric_name}"] = res

    return causality_dict
