from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import DEFAULT_FIGURES_DIR, DEFAULT_OUTPUT_PATH, LSER_COLUMNS, default_metadata, load_cleaned_output, save_excel, save_figures


def _find_similar_columns(df: pd.DataFrame, quant_columns: list[str]) -> tuple[list[str], list[str]]:
    similar_one: list[str] = []
    similar_two: list[str] = []
    for column_a in quant_columns:
        std_a = df[column_a].std()
        mean_a = df[column_a].mean()
        if pd.isna(std_a) or std_a == 0 or pd.isna(mean_a) or mean_a == 0:
            continue
        for column_b in quant_columns:
            if column_a == column_b:
                continue
            std_close = abs(std_a - df[column_b].std()) / abs(std_a) < 0.1
            mean_close = abs(mean_a - df[column_b].mean()) / abs(mean_a) < 0.1
            if std_close and mean_close:
                similar_one.append(column_a)
                similar_two.append(column_b)
    return similar_one, similar_two


def _merge_vcalc_into_lser_vx(df: pd.DataFrame) -> pd.DataFrame:
    if "Vcalc" not in df.columns or "LSER_VX" not in df.columns:
        return df
    combined = []
    for _, row in df.iterrows():
        vcalc = row["Vcalc"]
        lser_vx = row["LSER_VX"]
        if pd.isnull(vcalc) or pd.isnull(lser_vx):
            combined.append(float(lser_vx if pd.notnull(lser_vx) else vcalc))
        else:
            combined.append(float(0.5 * (vcalc + lser_vx)))
    df["LSER_VX"] = combined
    return df.drop(columns=["Vcalc"])


def _drop_rows_missing_lser(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    required = [column for column in LSER_COLUMNS if column in df.columns]
    missing_rows = df.index[df[required].isnull().any(axis=1)].tolist()
    if missing_rows:
        df = df.drop(index=missing_rows).reset_index(drop=True)
    return df, missing_rows


def _pairwise_r2(df: pd.DataFrame, quant_columns: list[str]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], list[str], list[str]]:
    r2dict: dict[tuple[str, str], float] = {}
    r2strict: dict[tuple[str, str], float] = {}
    r2plot1: list[str] = []
    r2plot2: list[str] = []
    for column_a in quant_columns:
        for column_b in quant_columns:
            if column_a == column_b:
                r2dict[(column_a, column_b)] = 1.0
                r2strict[(column_a, column_b)] = 1.0
                continue
            x = df[column_a]
            y = df[column_b]
            idx = np.isfinite(x) & np.isfinite(y)
            x_valid = x[idx]
            y_valid = y[idx]
            if len(x_valid) <= 1 or x_valid.std() <= 1e-10 or y_valid.std() <= 1e-10:
                r2 = np.nan
            else:
                x_norm = (x_valid - x_valid.min()) / (x_valid.max() - x_valid.min())
                y_norm = (y_valid - y_valid.min()) / (y_valid.max() - y_valid.min())
                ab = np.polyfit(x_norm, y_norm, 1)
                y_mean = y_norm.mean()
                mean_resid = ((y_norm - y_mean) ** 2).sum()
                lin_resid = ((y_norm - (ab[0] * x_norm + ab[1])) ** 2).sum()
                r2 = np.nan if mean_resid <= 1e-10 else max(min(1 - (lin_resid / mean_resid), 1.0), -1.0)
            r2dict[(column_a, column_b)] = r2
            if pd.notnull(r2) and r2 > 0.3:
                r2strict[(column_a, column_b)] = float(r2)
            if pd.notnull(r2) and r2 > 0.7:
                r2plot1.append(column_a)
                r2plot2.append(column_b)
    return r2dict, r2strict, r2plot1, r2plot2


def _build_exploration_figures(df: pd.DataFrame, r2strict: dict[tuple[str, str], float]) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    if {"TotalAtomsNumber", "LSER_VX"}.issubset(df.columns):
        fig, ax = plt.subplots()
        df.plot.scatter(x="TotalAtomsNumber", y="LSER_VX", s=100, ax=ax)
        figures["total_atoms_vs_lser_vx"] = fig
    if {"HeavyAtomCount", "SumAtomicVolume"}.issubset(df.columns):
        fig, ax = plt.subplots()
        df.plot.scatter(x="HeavyAtomCount", y="SumAtomicVolume", s=100, ax=ax)
        figures["heavy_atoms_vs_sum_atomic_volume"] = fig

    columns = sorted({column for pair in r2strict.keys() for column in pair})
    corr_matrix = pd.DataFrame(0.0, index=columns, columns=columns)
    for (col1, col2), value in r2strict.items():
        corr_matrix.loc[col1, col2] = value
        corr_matrix.loc[col2, col1] = value

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Correlation Matrix (R² values)")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    figures["correlation_heatmap"] = fig
    return figures


def run_data_exploration(
    data: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    save_output: bool = True,
    save_figures_output: bool = False,
    figures_dir: Path | str | None = None,
) -> dict[str, Any]:
    df = data.copy() if data is not None else load_cleaned_output(output_path or DEFAULT_OUTPUT_PATH).copy()
    similar_one, similar_two = _find_similar_columns(df, df.select_dtypes(include=[np.number]).columns.tolist())
    df = _merge_vcalc_into_lser_vx(df)
    df, missing_lser_rows = _drop_rows_missing_lser(df)
    quant_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    r2dict, r2strict, r2plot1, r2plot2 = _pairwise_r2(df, quant_columns)
    figures = _build_exploration_figures(df, r2strict)

    if save_output:
        save_excel(df, output_path or DEFAULT_OUTPUT_PATH)
    figure_paths = (
        save_figures(figures, figures_dir or DEFAULT_FIGURES_DIR, stage_name="data_exploration")
        if save_figures_output
        else {}
    )

    metadata = default_metadata(df)
    metadata.update(
        {
            "similar_columns": similar_one,
            "similar_columns_partner": similar_two,
            "missing_lser_rows": missing_lser_rows,
            "high_r2_plot_x": r2plot1,
            "high_r2_plot_y": r2plot2,
            "r2_strict_count": len(r2strict),
            "output_path": str(output_path or DEFAULT_OUTPUT_PATH),
        }
    )
    return {
        "data": df,
        "metadata": metadata,
        "artifacts": {
            "r2dict": r2dict,
            "r2strict": r2strict,
            "figures": figures,
            "figure_paths": figure_paths,
        },
    }
