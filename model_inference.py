from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .clustering import run_clustering
from .common import INFERENCE_SCORE_COLUMNS, nan_to_zero
from .data_cleaning import run_data_cleaning
from .data_exploration import run_data_exploration

DEFAULT_PERMEABILITY_COEFFICIENTS = np.array([0.15, 0.25, 0.35, 0.35, -0.1], dtype=float)
DEFAULT_MODEL_FAMILIES = ["MLR", "PCRMLR", "PLS", "LASSO"]


def _predict_targets_for_family(model_bundle: dict[str, Any], family: str, features: np.ndarray) -> np.ndarray:
    predictions = []
    if family == "PCRMLR":
        for target in model_bundle["target_columns"]:
            result = model_bundle["models"][family][target]
            predictions.append(result["model"].predict(result["pca_model"].transform(features)).reshape(-1))
    else:
        for target in model_bundle["target_columns"]:
            result = model_bundle["models"][family][target]
            predictions.append(result["model"].predict(features).reshape(-1))
    return np.vstack(predictions).T


def _normalized_volume(series: pd.Series) -> pd.Series:
    std = series.std(axis=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean(axis=0)) / std


def _relative_score_from_predictions(
    prediction_row: np.ndarray,
    norm_volume: float,
    basis_score: float,
    coefficients: np.ndarray,
) -> float:
    score_vector = np.array(
        [prediction_row[0], prediction_row[1], prediction_row[2], prediction_row[3], norm_volume],
        dtype=float,
    )
    return float(np.dot(score_vector, coefficients) - basis_score)


def _coerce_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ["MolecularName", "CASRN", "CAS-RN", "ChE137Number", "ChemicalType"]:
        if column not in result.columns:
            result[column] = np.nan
    return result


def lookup_molecule(
    data: pd.DataFrame,
    molecular_name: str | None = None,
    casrn: str | None = None,
    che137_number: int | float | None = None,
) -> pd.DataFrame:
    df = _coerce_identifier_columns(data)
    if molecular_name is not None:
        return df[df["MolecularName"] == molecular_name].copy()
    if casrn is not None:
        return df[(df["CASRN"] == casrn) | (df["CAS-RN"] == casrn)].copy()
    if che137_number is not None:
        return df[df["ChE137Number"] == che137_number].copy()
    raise ValueError("Provide one lookup value: molecular_name, casrn, or che137_number.")


def _basis_scores(model_bundle: dict[str, Any], model_families: list[str], basismol: str, coefficients: np.ndarray) -> dict[str, float]:
    model_df = model_bundle["model_df"].reset_index(drop=True)
    regression_x = model_bundle["RegressionX"]
    normvol = _normalized_volume(model_df["LSER_VX"]).reset_index(drop=True)
    scores: dict[str, float] = {}
    for family in model_families:
        predicted_targets = _predict_targets_for_family(model_bundle, family, regression_x)
        basis_index = model_df.index[model_df["MolecularName"] == basismol].tolist()
        if not basis_index:
            raise ValueError(f"Basis molecule '{basismol}' was not found in the model dataset.")
        idx = basis_index[0]
        score_vector = np.array(
            [
                predicted_targets[idx, 0],
                predicted_targets[idx, 1],
                predicted_targets[idx, 2],
                predicted_targets[idx, 3],
                normvol.iloc[idx],
            ],
            dtype=float,
        )
        scores[family] = float(np.dot(score_vector, coefficients))
    return scores


def _training_chemtype_means(model_bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    training_frame = model_bundle["training_feature_frame"].copy()
    training_frame["ChemicalType"] = model_bundle["model_df"]["ChemicalType"].reset_index(drop=True)
    chemtype_means = training_frame.groupby("ChemicalType")[model_bundle["feature_columns"]].mean(numeric_only=True)
    global_means = model_bundle["training_feature_frame"][model_bundle["feature_columns"]].mean(axis=0)
    return chemtype_means, global_means


def _impute_feature_frame(model_bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = model_bundle["feature_columns"]
    result = _coerce_identifier_columns(df)
    for column in feature_columns:
        if column not in result.columns:
            result[column] = np.nan

    chemtype_means, global_means = _training_chemtype_means(model_bundle)
    for index, row in result.iterrows():
        chem_type = row["ChemicalType"]
        for column in feature_columns:
            if pd.notnull(row[column]):
                continue
            if pd.notnull(chem_type) and chem_type in chemtype_means.index and pd.notnull(chemtype_means.loc[chem_type, column]):
                result.at[index, column] = chemtype_means.loc[chem_type, column]
            else:
                result.at[index, column] = global_means[column]
    return result


def _prepare_feature_matrix(model_bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x_means = model_bundle["x_means"]
    x_stds = model_bundle["x_stds"].replace(0, np.nan)
    scaled = (df[model_bundle["feature_columns"]] - x_means) / x_stds
    return nan_to_zero(scaled)


def score_existing_molecules(
    model_bundle: dict[str, Any],
    molecular_name: str | None = None,
    casrn: str | None = None,
    che137_number: int | float | None = None,
    basis_molecule: str = "Diazepam",
    model_families: list[str] | None = None,
    coefficients: np.ndarray | None = None,
) -> dict[str, Any]:
    chosen_families = model_families or DEFAULT_MODEL_FAMILIES
    score_coefficients = DEFAULT_PERMEABILITY_COEFFICIENTS if coefficients is None else np.asarray(coefficients, dtype=float)
    model_df = _coerce_identifier_columns(model_bundle["model_df"]).reset_index(drop=True)
    matches = lookup_molecule(model_df, molecular_name=molecular_name, casrn=casrn, che137_number=che137_number)
    if matches.empty:
        raise ValueError("No molecules matched the provided lookup.")
    regression_x = model_bundle["RegressionX"]
    normvol = _normalized_volume(model_df["LSER_VX"]).reset_index(drop=True)
    basis_scores = _basis_scores(model_bundle, chosen_families, basis_molecule, score_coefficients)
    row_indices = matches.index.tolist()

    predictions: dict[str, list[dict[str, Any]]] = {}
    for family in chosen_families:
        family_predictions = _predict_targets_for_family(model_bundle, family, regression_x)
        family_rows = []
        for idx in row_indices:
            family_rows.append(
                {
                    "row_index": int(idx),
                    "molecule": model_df.iloc[idx][["MolecularName", "CASRN", "CAS-RN", "ChE137Number", "ChemicalType"]].to_dict(),
                    "predicted_targets": {
                        target: float(family_predictions[idx, target_index])
                        for target_index, target in enumerate(model_bundle["target_columns"])
                    },
                    "relative_score": _relative_score_from_predictions(
                        family_predictions[idx], float(normvol.iloc[idx]), basis_scores[family], score_coefficients
                    ),
                }
            )
        predictions[family] = family_rows
    return {"matches": matches.reset_index(drop=True), "predictions": predictions, "basis_molecule": basis_molecule}


def score_new_molecules_direct(
    model_bundle: dict[str, Any],
    new_data: pd.DataFrame,
    basis_molecule: str = "Diazepam",
    model_families: list[str] | None = None,
    coefficients: np.ndarray | None = None,
) -> dict[str, Any]:
    chosen_families = model_families or DEFAULT_MODEL_FAMILIES
    score_coefficients = DEFAULT_PERMEABILITY_COEFFICIENTS if coefficients is None else np.asarray(coefficients, dtype=float)
    prepared = _impute_feature_frame(model_bundle, new_data)
    if prepared.empty:
        return {"prepared_data": prepared, "predictions": {family: [] for family in chosen_families}, "basis_molecule": basis_molecule}
    feature_matrix = _prepare_feature_matrix(model_bundle, prepared)
    basis_scores = _basis_scores(model_bundle, chosen_families, basis_molecule, score_coefficients)
    reference_volume = model_bundle["model_df"]["LSER_VX"]
    combined_volume = pd.concat([reference_volume.reset_index(drop=True), prepared["LSER_VX"].reset_index(drop=True)], ignore_index=True)
    combined_norm_volume = _normalized_volume(combined_volume).iloc[len(reference_volume):].reset_index(drop=True)

    predictions: dict[str, list[dict[str, Any]]] = {}
    for family in chosen_families:
        family_predictions = _predict_targets_for_family(model_bundle, family, feature_matrix)
        family_rows = []
        for idx in range(len(prepared)):
            family_rows.append(
                {
                    "row_index": int(idx),
                    "molecule": prepared.iloc[idx][["MolecularName", "CASRN", "CAS-RN", "ChE137Number", "ChemicalType"]].to_dict(),
                    "predicted_targets": {
                        target: float(family_predictions[idx, target_index])
                        for target_index, target in enumerate(model_bundle["target_columns"])
                    },
                    "relative_score": _relative_score_from_predictions(
                        family_predictions[idx],
                        float(combined_norm_volume.iloc[idx]),
                        basis_scores[family],
                        score_coefficients,
                    ),
                }
            )
        predictions[family] = family_rows
    return {"prepared_data": prepared, "predictions": predictions, "basis_molecule": basis_molecule}


def score_new_molecules_pipeline(
    pipeline_results: dict[str, Any],
    new_data: pd.DataFrame,
    basis_molecule: str = "Diazepam",
    model_families: list[str] | None = None,
    coefficients: np.ndarray | None = None,
    feature_hooks: list[Any] | None = None,
) -> dict[str, Any]:
    model_bundle = pipeline_results["model_training"]["model_bundle"]
    reference_data = pipeline_results["data_cleaning"]["data"].copy()
    reference_data["InferenceSource"] = "reference"
    raw_new_data = new_data.copy()
    raw_new_data["InferenceSource"] = "new"
    raw_combined = pd.concat([reference_data, raw_new_data], ignore_index=True, sort=False)
    cleaned = run_data_cleaning(
        data=raw_combined,
        save_output=False,
        feature_hooks=feature_hooks,
        append_new_molecules=False,
    )
    explored = run_data_exploration(cleaned["data"], save_output=False)
    clustered = run_clustering(explored["data"], save_output=False)

    new_candidates = clustered["data"]
    candidate_rows = new_candidates[new_candidates["InferenceSource"] == "new"].copy()
    if "InferenceSource" in candidate_rows.columns:
        candidate_rows = candidate_rows.drop(columns=["InferenceSource"])
    direct_results = score_new_molecules_direct(
        model_bundle,
        candidate_rows,
        basis_molecule=basis_molecule,
        model_families=model_families,
        coefficients=coefficients,
    )
    return {
        "cleaning": cleaned,
        "exploration": explored,
        "clustering": clustered,
        "direct_scoring": direct_results,
    }


def _predict_relative_score(
    model_bundle: dict[str, Any],
    family: str,
    chemtype: str,
    basismol: str,
    coefficients: np.ndarray,
) -> dict[str, Any]:
    model_df = model_bundle["model_df"].reset_index(drop=True)
    regression_x = model_bundle["RegressionX"]
    predicted_targets = _predict_targets_for_family(model_bundle, family, regression_x)
    normvol = _normalized_volume(model_df["LSER_VX"]).reset_index(drop=True)

    permeability_scores = []
    basis_scores = _basis_scores(model_bundle, [family], basismol, coefficients)
    for i in range(len(model_df)):
        if model_df.iloc[i]["ChemicalType"] == chemtype:
            permeability_scores.append(
                _relative_score_from_predictions(
                    predicted_targets[i],
                    float(normvol.iloc[i]),
                    basis_scores[family],
                    coefficients,
                )
                + basis_scores[family]
            )

    return {
        "relative_score": float(np.mean(permeability_scores) - basis_scores[family]),
        "group_mean_score": float(np.mean(permeability_scores)),
        "basis_score": float(basis_scores[family]),
        "family": family,
        "chemtype": chemtype,
        "basis_molecule": basismol,
        "score_columns": INFERENCE_SCORE_COLUMNS,
    }


def run_model_inference(
    model_bundle: dict[str, Any],
    chemtype: str = "pharma",
    basismol: str = "Diazepam",
    model_families: list[str] | None = None,
    coefficients: np.ndarray | None = None,
) -> dict[str, Any]:
    chosen_families = model_families or DEFAULT_MODEL_FAMILIES
    score_coefficients = DEFAULT_PERMEABILITY_COEFFICIENTS if coefficients is None else np.asarray(coefficients, dtype=float)
    predictions = {
        family: _predict_relative_score(model_bundle, family, chemtype, basismol, score_coefficients)
        for family in chosen_families
    }
    return {
        "predictions": predictions,
        "chemtype": chemtype,
        "basis_molecule": basismol,
        "coefficients": score_coefficients,
    }
