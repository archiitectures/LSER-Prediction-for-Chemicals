from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_NEW_MOLECULES_PATH,
    DEFAULT_OUTPUT_PATH,
    POSITIVE_ONLY_COLUMNS,
    apply_feature_hooks,
    default_metadata,
    load_excel,
    load_new_molecules,
    save_excel,
)


def _remove_invariant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    invariant_columns = df.columns[df.nunique(dropna=False) == 1].tolist()
    if invariant_columns:
        df = df.drop(columns=invariant_columns)
    return df, invariant_columns


def _drop_high_missing_columns(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    missing_fraction = df.isnull().sum() / len(df)
    columns_to_drop = missing_fraction[missing_fraction >= threshold].index.tolist()
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df, columns_to_drop


def _append_new_molecules(df: pd.DataFrame, new_molecules_path: Path | str | None) -> tuple[pd.DataFrame, int]:
    new_molecules = load_new_molecules(new_molecules_path)
    appended_count = len(new_molecules)
    df = pd.concat([df, new_molecules], ignore_index=True, sort=False)
    if appended_count and "ChE137Number" in df.columns:
        for offset in range(appended_count, 0, -1):
            row_index = len(df) - offset
            if row_index > 0:
                df.iloc[row_index, df.columns.get_loc("ChE137Number")] = df.iloc[row_index - 1, df.columns.get_loc("ChE137Number")] + 1
    return df, appended_count


def _build_preprocessing_features(df: pd.DataFrame) -> pd.DataFrame:
    if "PolarSurfaceArea(sqAngstrom)" in df.columns and "HBondDonorCount" in df.columns:
        values = np.where(
            df["HBondDonorCount"].fillna(0).to_numpy() == 0,
            0,
            df["PolarSurfaceArea(sqAngstrom)"].fillna(0).to_numpy(),
        )
        df["PolarSurfaceAreaifHbonding"] = values
    return df


def _cleanup_impossible_values(df: pd.DataFrame) -> dict[str, int]:
    replaced_counts: dict[str, int] = {}
    for column in POSITIVE_ONLY_COLUMNS:
        if column not in df.columns:
            continue
        negative_mask = df[column] < 0
        replaced_counts[column] = int(negative_mask.sum())
        if negative_mask.any():
            df.loc[negative_mask, column] = np.nan
    return replaced_counts


def run_data_cleaning(
    data: pd.DataFrame | None = None,
    dataset_path: Path | str | None = None,
    new_molecules_path: Path | str | None = None,
    output_path: Path | str | None = None,
    missing_threshold: float = 0.30,
    save_output: bool = True,
    feature_hooks: list[Any] | None = None,
    append_new_molecules: bool = True,
) -> dict[str, Any]:
    df = data.copy() if data is not None else load_excel(dataset_path or DEFAULT_DATASET_PATH).copy()
    raw_shape = df.shape
    df = df.drop_duplicates()
    deduplicated_shape = df.shape

    df, invariant_columns = _remove_invariant_columns(df)
    if "ChemicalType" in df.columns:
        df["ChemicalType"] = df["ChemicalType"].fillna("missing")
    df, high_missing_columns = _drop_high_missing_columns(df, missing_threshold)

    appended_count = 0
    if append_new_molecules:
        df, appended_count = _append_new_molecules(df, new_molecules_path or DEFAULT_NEW_MOLECULES_PATH)
    df = _build_preprocessing_features(df)
    df, applied_feature_hooks = apply_feature_hooks(df, feature_hooks)
    impossible_value_counts = _cleanup_impossible_values(df)

    if save_output:
        save_excel(df, output_path or DEFAULT_OUTPUT_PATH)

    metadata = default_metadata(df)
    metadata.update(
        {
            "raw_shape": raw_shape,
            "deduplicated_shape": deduplicated_shape,
            "invariant_columns": invariant_columns,
            "high_missing_columns": high_missing_columns,
            "missing_threshold": missing_threshold,
            "appended_molecules": appended_count,
            "impossible_value_counts": impossible_value_counts,
            "applied_feature_hooks": applied_feature_hooks,
            "output_path": str(output_path or DEFAULT_OUTPUT_PATH),
        }
    )
    return {"data": df, "metadata": metadata, "artifacts": {}}
