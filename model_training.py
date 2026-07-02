from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from scipy.stats import chi2, norm
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split

from .common import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_PATH,
    TRAINING_TARGET_COLUMNS,
    default_metadata,
    load_cleaned_output,
    nan_to_zero,
    save_figures,
    zscore_with_stats,
)

DEFAULT_EXCLUDED_FEATURES = [
    "ChE137Number",
    "PubChem CID",
    "LSER_VX",
    "LSER_E",
    "LSER_S",
    "LSER_A",
    "LSER_B",
]


def prepare_model_matrices(
    data: pd.DataFrame,
    excluded_features: list[str] | None = None,
    tune_features_remove: list[str] | None = None,
) -> dict[str, Any]:
    df = data.copy()
    excluded = set(DEFAULT_EXCLUDED_FEATURES if excluded_features is None else excluded_features)
    excluded.update(tune_features_remove or ["Mw(g/moL)"])
    model_df = df.reset_index(drop=True).copy()
    quant_columns = model_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [column for column in quant_columns if column not in excluded]
    regression_x = model_df[feature_columns]
    regression_y = model_df[TRAINING_TARGET_COLUMNS]
    scaled_x, x_means, x_stds = zscore_with_stats(regression_x)
    scaled_y, y_means, y_stds = zscore_with_stats(regression_y)
    return {
        "model_df": model_df,
        "feature_columns": feature_columns,
        "target_columns": TRAINING_TARGET_COLUMNS,
        "RegressionX": nan_to_zero(scaled_x),
        "RegressionY": nan_to_zero(scaled_y),
        "training_feature_frame": regression_x.copy(),
        "training_target_frame": regression_y.copy(),
        "x_means": x_means,
        "x_stds": x_stds,
        "y_means": y_means,
        "y_stds": y_stds,
    }


def _train_validation_split(
    regression_x: np.ndarray,
    regression_y: np.ndarray,
    test_size: float = 0.4,
    random_state_test: int = 21,
    random_state_val: int = 42,
) -> dict[str, np.ndarray]:
    x_train1, x_test, y_train1, y_test = train_test_split(
        regression_x, regression_y, test_size=test_size, random_state=random_state_test
    )
    x_val, x_train, y_val, y_train = train_test_split(
        x_train1, y_train1, test_size=0.5, random_state=random_state_val
    )
    return {
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "Y_train": y_train,
        "Y_val": y_val,
        "Y_test": y_test,
    }


def _mlr(ls_index: int, x_val: np.ndarray, x_train: np.ndarray, y_val: np.ndarray, y_train: np.ndarray, only_r2: bool = False) -> dict[str, Any] | float:
    regressor = LinearRegression()
    regressor.fit(x_train, y_train[:, ls_index])
    predict = regressor.predict(x_val)
    r2 = sklearn.metrics.r2_score(y_val[:, ls_index], predict)
    if only_r2:
        return float(r2)
    mse = metrics.mean_squared_error(y_val[:, ls_index], predict)
    return {
        "model": regressor,
        "coefficients": pd.DataFrame({"Coefficient": regressor.coef_.ravel()}),
        "prediction": predict,
        "comparison": pd.DataFrame({"Actual": y_val[:, ls_index], "Predicted": predict}),
        "mae": metrics.mean_absolute_error(y_val[:, ls_index], predict),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
    }


def _pcr_mlr(ls_index: int, x_val: np.ndarray, x_train: np.ndarray, y_val: np.ndarray, y_train: np.ndarray, only_r2: bool = False) -> dict[str, Any] | float:
    pca = PCA(n_components=10)
    pca_x_train = pca.fit_transform(x_train)
    pca_x_val = pca.transform(x_val)
    regressor = LinearRegression()
    regressor.fit(pca_x_train, y_train[:, ls_index])
    predict = regressor.predict(pca_x_val)
    r2 = sklearn.metrics.r2_score(y_val[:, ls_index], predict)
    if only_r2:
        return float(r2)
    mse = metrics.mean_squared_error(y_val[:, ls_index], predict)
    return {
        "model": regressor,
        "pca_model": pca,
        "coefficients": pd.DataFrame({"Coefficient": regressor.coef_.ravel()}),
        "prediction": predict,
        "comparison": pd.DataFrame({"Actual": y_val[:, ls_index], "Predicted": predict}),
        "mae": metrics.mean_absolute_error(y_val[:, ls_index], predict),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "explained_variance_ratio": float(sum(pca.explained_variance_ratio_)),
    }


def _pls(ls_index: int, x_val: np.ndarray, x_train: np.ndarray, y_val: np.ndarray, y_train: np.ndarray, only_r2: bool = False) -> dict[str, Any] | float:
    pls = PLSRegression(n_components=5)
    pls.fit(x_train, y_train[:, ls_index])
    predict = pls.predict(x_val).reshape(-1)
    r2 = sklearn.metrics.r2_score(y_val[:, ls_index], predict)
    if only_r2:
        return float(r2)
    mse = metrics.mean_squared_error(y_val[:, ls_index], predict)
    residuals = y_val[:, ls_index].flatten() - predict
    return {
        "model": pls,
        "coefficients": pls.coef_.ravel(),
        "prediction": predict,
        "residuals": residuals,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
    }


def _lasso(ls_index: int, x_val: np.ndarray, x_train: np.ndarray, y_val: np.ndarray, y_train: np.ndarray, only_r2: bool = False) -> dict[str, Any] | float:
    lasso = Lasso(alpha=0.1)
    lasso.fit(x_train, y_train[:, ls_index])
    predict = lasso.predict(x_val)
    r2 = sklearn.metrics.r2_score(y_val[:, ls_index], predict)
    if only_r2:
        return float(r2)
    return {
        "model": lasso,
        "coefficients": lasso.coef_.ravel(),
        "prediction": predict,
        "r2": float(r2),
    }


def _kpi_calc(r2e: float, r2s: float, r2a: float, r2b: float) -> float:
    return 0.1 * r2e + 0.2 * r2s + 0.35 * r2a + 0.35 * r2b


def _resid_normal(resid: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    mu, std = norm.fit(resid)
    xmin = min(resid)
    xmax = max(resid)
    x = np.linspace(xmin, xmax, len(resid))
    normdist = len(resid) * (xmax - xmin) / bins * norm.pdf(x, mu, std)
    return x, normdist


def _feature_corrcoeff(x_val: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    feature_corr = np.zeros(x_val.shape[1])
    for j in range(x_val.shape[1]):
        feature_corr[j] = pd.DataFrame({"Feature": x_val[:, j], "Predicted": prediction}).corr().iloc[0, 1]
    return np.nan_to_num(feature_corr, copy=True, nan=0.0)


def _breusch_pagan(residuals: np.ndarray, x_design: np.ndarray) -> dict[str, Any]:
    resid = np.square(residuals)
    n = len(residuals)
    dof = x_design.shape[1] - 1
    regressor = LinearRegression()
    regressor.fit(x_design, resid)
    predict = regressor.predict(x_design)
    r2 = sklearn.metrics.r2_score(resid, predict)
    chi = n * r2
    p = float(1 - chi2.cdf(chi, dof))
    return {
        "Breusch-Pagan Stat": float(chi),
        "p-val": p,
        "Homoscedasticity": "present" if p > 0.05 else "not present",
    }


def _different_r_tester(regression_x: np.ndarray, regression_y: np.ndarray, num: int) -> dict[str, Any]:
    mlr_kpi = np.zeros(num)
    pcr_kpi = np.zeros(num)
    pls_kpi = np.zeros(num)
    lasso_kpi = np.zeros(num)
    negative_counts = {"MLR": 0, "PCR": 0, "PLS": 0, "LASSO": 0}
    random.seed()
    for i in range(num):
        rand1 = random.randint(1, 1000)
        rand2 = random.randint(1, 1000)
        splits = _train_validation_split(regression_x, regression_y, random_state_test=rand1, random_state_val=rand2)
        mlr_scores = [_mlr(idx, splits["X_val"], splits["X_train"], splits["Y_val"], splits["Y_train"], only_r2=True) for idx in range(4)]
        pcr_scores = [_pcr_mlr(idx, splits["X_val"], splits["X_train"], splits["Y_val"], splits["Y_train"], only_r2=True) for idx in range(4)]
        pls_scores = [_pls(idx, splits["X_val"], splits["X_train"], splits["Y_val"], splits["Y_train"], only_r2=True) for idx in range(4)]
        lasso_scores = [_lasso(idx, splits["X_val"], splits["X_train"], splits["Y_val"], splits["Y_train"], only_r2=True) for idx in range(4)]
        for name, store, scores in [
            ("MLR", mlr_kpi, mlr_scores),
            ("PCR", pcr_kpi, pcr_scores),
            ("PLS", pls_kpi, pls_scores),
            ("LASSO", lasso_kpi, lasso_scores),
        ]:
            kpi = _kpi_calc(*scores)
            if kpi > 0:
                store[i] = kpi
            else:
                negative_counts[name] += 1
    random.seed(1)
    return {
        "MLR": mlr_kpi,
        "PCR": pcr_kpi,
        "PLS": pls_kpi,
        "LASSO": lasso_kpi,
        "negative_counts": negative_counts,
    }


def train_all_supported_models(
    prepared: dict[str, Any],
    splits: dict[str, np.ndarray],
    repeated_split_tests: int = 0,
) -> dict[str, Any]:
    mlr_results = {}
    pcr_results = {}
    pls_results = {}
    lasso_results = {}
    for index, target in enumerate(TRAINING_TARGET_COLUMNS):
        mlr_results[target] = _mlr(index, splits["X_val"], splits["X_train"], splits["Y_val"], splits["Y_train"])
        pcr_results[target] = _pcr_mlr(index, splits["X_val"], splits["X_train"], splits["Y_val"], splits["Y_train"])
        pls_results[target] = _pls(index, splits["X_val"], splits["X_train"], splits["Y_val"], splits["Y_train"])
        lasso_results[target] = _lasso(index, splits["X_val"], splits["X_train"], splits["Y_val"], splits["Y_train"])

    diagnostics: dict[str, Any] = {
        "KPI": {
            "MLR": _kpi_calc(*(mlr_results[target]["r2"] for target in TRAINING_TARGET_COLUMNS)),
            "PCR": _kpi_calc(*(pcr_results[target]["r2"] for target in TRAINING_TARGET_COLUMNS)),
            "PLS": _kpi_calc(*(pls_results[target]["r2"] for target in TRAINING_TARGET_COLUMNS)),
            "LASSO": _kpi_calc(*(lasso_results[target]["r2"] for target in TRAINING_TARGET_COLUMNS)),
        },
        "breusch_pagan": {
            target: _breusch_pagan(pls_results[target]["residuals"], splits["X_val"]) for target in TRAINING_TARGET_COLUMNS
        },
        "feature_correlation": {
            target: _feature_corrcoeff(splits["X_val"], mlr_results[target]["prediction"]) for target in TRAINING_TARGET_COLUMNS
        },
    }
    if repeated_split_tests > 0:
        diagnostics["repeated_split_evaluation"] = _different_r_tester(
            prepared["RegressionX"], prepared["RegressionY"], repeated_split_tests
        )

    return {
        "models": {"MLR": mlr_results, "PCRMLR": pcr_results, "PLS": pls_results, "LASSO": lasso_results},
        "diagnostics": diagnostics,
    }


def _build_training_figures(model_bundle: dict[str, Any], splits: dict[str, np.ndarray]) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    fig, axes = plt.subplots(3, 4, figsize=(15, 5))
    for row_index, family in enumerate(["MLR", "PCRMLR", "PLS"]):
        for col_index, target in enumerate(TRAINING_TARGET_COLUMNS):
            result = model_bundle["models"][family][target]
            axes[row_index, col_index].scatter(splits["Y_val"][:, col_index], result["prediction"], alpha=0.25)
            axes[row_index, col_index].plot(
                splits["Y_val"][:, col_index], splits["Y_val"][:, col_index], color="black", linestyle="-"
            )
            axes[row_index, col_index].set_title(f"{family} {target}, r2={result['r2']:.2f}")
    fig.tight_layout()
    figures["validation_scatter_grid"] = fig

    fig, axes = plt.subplots(2, 4, figsize=(15, 5))
    for idx, target in enumerate(TRAINING_TARGET_COLUMNS):
        residuals = model_bundle["models"]["PLS"][target]["residuals"]
        x_vals, normal_curve = _resid_normal(residuals)
        axes[0, idx].hist(residuals, bins=10)
        axes[0, idx].plot(x_vals, normal_curve)
        axes[0, idx].set_title(target)
        stats.probplot(residuals, plot=axes[1, idx])
    fig.tight_layout()
    figures["pls_residuals_and_qq"] = fig
    return figures


def run_model_training(
    data: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    repeated_split_tests: int = 0,
    save_figures_output: bool = False,
    figures_dir: Path | str | None = None,
) -> dict[str, Any]:
    df = data.copy() if data is not None else load_cleaned_output(output_path or DEFAULT_OUTPUT_PATH).copy()
    prepared = prepare_model_matrices(df)
    splits = _train_validation_split(prepared["RegressionX"], prepared["RegressionY"])
    model_bundle = train_all_supported_models(prepared, splits, repeated_split_tests=repeated_split_tests)
    model_bundle["feature_columns"] = prepared["feature_columns"]
    model_bundle["target_columns"] = prepared["target_columns"]
    model_bundle["x_means"] = prepared["x_means"]
    model_bundle["x_stds"] = prepared["x_stds"]
    model_bundle["y_means"] = prepared["y_means"]
    model_bundle["y_stds"] = prepared["y_stds"]
    model_bundle["model_df"] = prepared["model_df"]
    model_bundle["RegressionX"] = prepared["RegressionX"]
    model_bundle["RegressionY"] = prepared["RegressionY"]
    model_bundle["training_feature_frame"] = prepared["training_feature_frame"]
    model_bundle["training_target_frame"] = prepared["training_target_frame"]

    figures = _build_training_figures(model_bundle, splits)
    figure_paths = (
        save_figures(figures, figures_dir or DEFAULT_FIGURES_DIR, stage_name="model_training")
        if save_figures_output
        else {}
    )
    metadata = default_metadata(df)
    metadata.update(
        {
            "feature_count": len(prepared["feature_columns"]),
            "target_count": len(prepared["target_columns"]),
            "split_sizes": {name: int(len(values)) for name, values in splits.items() if name.startswith(("X_", "Y_"))},
        }
    )
    return {
        "data": df,
        "metadata": metadata,
        "model_bundle": model_bundle,
        "metrics": model_bundle["diagnostics"],
        "splits": splits,
        "artifacts": {"figures": figures, "figure_paths": figure_paths},
    }
