from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from .common import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_PATH,
    PCA_LSER_COLUMNS,
    default_metadata,
    load_cleaned_output,
    nan_to_zero,
    save_excel,
    save_figures,
    zscore_frame,
)

PCB_CHEMICAL_TYPES = [
    "PCB (polychlorinated biphenyl - 2)",
    "PCB (polychlorinated biphenyl - 3)",
    "PCB (polychlorinated biphenyl - 4)",
    "PCB (polychlorinated biphenyl - 5)",
    "PCB (polychlorinated biphenyl - 6)",
    "PCB (polychlorinated biphenyl - 7)",
    "PCB (polychlorinated biphenyl - 8)",
    "PCB (polychlorinated biphenyl - 9)",
    "PCB (polychlorinated biphenyl - 10)",
]


def _modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True).copy()


def _pca_input(df: pd.DataFrame) -> np.ndarray:
    pca_frame = zscore_frame(_modeling_frame(df)[PCA_LSER_COLUMNS])
    return nan_to_zero(pca_frame)


def _pharma_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    pca = PCA(n_components=2)
    pca_data = pd.DataFrame(pca.fit_transform(_pca_input(df)), columns=["PCA1", "PCA2"])
    labels = _modeling_frame(df)["ChemicalType"].reset_index(drop=True)
    pharma_mask = labels.isin(["pharma", "hormone"])
    pharmadf = pca_data.loc[pharma_mask].reset_index(drop=True)
    return pharmadf, pca_data, pca


def _mahalanobis_pca(df: pd.DataFrame, threshold: float, outlier: bool = False) -> tuple[np.ndarray, Ellipse | None, pd.DataFrame | None]:
    pharma, pca_data, _ = _pharma_df(df)
    mu = pharma.mean(axis=0).to_numpy(dtype=float)
    invcov = np.linalg.inv(np.cov(pharma.values.T))
    mahala = np.zeros(len(pca_data))
    points_under_threshold: list[list[float]] = []
    for i in range(len(pca_data)):
        delta = pca_data.iloc[i].to_numpy() - mu
        dsquare = float(np.dot(np.dot(delta.T, invcov), delta))
        mahala[i] = np.sqrt(dsquare)
        if outlier and dsquare <= chi2.ppf(threshold, df=2):
            points_under_threshold.append([pca_data.iloc[i, 0], pca_data.iloc[i, 1]])
    if not outlier:
        return mahala, None, None
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(pharma.values.T))
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    chi2_val = chi2.ppf(threshold, df=2)
    width, height = 2 * np.sqrt(chi2_val * eigenvalues)
    ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, edgecolor="red", facecolor="none", linewidth=4)
    points = pd.DataFrame(points_under_threshold, columns=["PCA1", "PCA2"])
    return mahala, ellipse, points


def _mahala_keep_indices(df: pd.DataFrame, threshold: float) -> list[int]:
    _, pca_data, _ = _pharma_df(df)
    mahala, _, _ = _mahalanobis_pca(df, threshold, outlier=True)
    sliced = _modeling_frame(df).reset_index(drop=True)
    keep = []
    for i in range(len(pca_data)):
        if (mahala[i] ** 2) <= chi2.ppf(threshold, df=2) or sliced.iloc[i]["ChemicalType"] in {"pharma", "hormone"}:
            keep.append(i)
    return keep


def _plot_clustering(df: pd.DataFrame, pca_data: pd.DataFrame, cluster_labels: np.ndarray, ellipse: Ellipse, keep_indices: list[int]) -> dict[str, plt.Figure]:
    figures: dict[str, plt.Figure] = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 1], c=cluster_labels)
    ax1.set_title("DBSCAN of PCA1, PCA2 with Outliers Highlighted")
    ax2.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 1], alpha=0.4)
    ax2.add_patch(ellipse)
    ax2.set_title("PCA Space with Mahalanobis Ellipse")
    figures["dbscan_and_ellipse"] = fig

    sliced = _modeling_frame(df).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(pca_data.iloc[keep_indices, 0], pca_data.iloc[keep_indices, 1], alpha=0.7)
    for i in range(len(pca_data)):
        chemical_type = sliced.iloc[i]["ChemicalType"]
        alpha = 0.125 if chemical_type == "missing" else 0.25
        color = "green" if chemical_type in {"pharma", "hormone"} else None
        if color:
            ax.scatter(pca_data.iloc[i, 0], pca_data.iloc[i, 1], c=color)
        ax.annotate(chemical_type, (pca_data.iloc[i, 0], pca_data.iloc[i, 1]), fontsize=8, alpha=alpha)
    ax.add_patch(
        Ellipse(
            xy=ellipse.center,
            width=ellipse.width,
            height=ellipse.height,
            angle=ellipse.angle,
            edgecolor=ellipse.get_edgecolor(),
            facecolor="none",
            linewidth=ellipse.get_linewidth(),
        )
    )
    ax.grid(True)
    figures["annotated_pca_space"] = fig
    return figures


def _rand_elim(df: pd.DataFrame, cut: int, chem_type: str, seed: int = 1) -> pd.DataFrame:
    random.seed(seed)
    cut_ratio = 100 / cut
    indices = df.index[df["ChemicalType"] == chem_type].tolist()
    number_to_cut = int(np.floor(len(indices) / cut_ratio))
    random.shuffle(indices)
    return df.drop(indices[:number_to_cut]).reset_index(drop=True)


def run_clustering(
    data: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    threshold: float = 0.88,
    save_output: bool = True,
    save_figures_output: bool = False,
    figures_dir: Path | str | None = None,
) -> dict[str, Any]:
    df = data.copy() if data is not None else load_cleaned_output(output_path or DEFAULT_OUTPUT_PATH).copy()
    for chemical_type in PCB_CHEMICAL_TYPES:
        if "ChemicalType" in df.columns:
            df = _rand_elim(df, 80, chemical_type)

    pharmadf, pca_data, pca = _pharma_df(df)
    dbscan = DBSCAN()
    cluster_labels = dbscan.fit_predict(pca_data)
    mahala, ellipse, points_under_threshold = _mahalanobis_pca(df, threshold, outlier=True)
    keep_indices = _mahala_keep_indices(df, threshold)

    sliced = _modeling_frame(df).reset_index(drop=True)
    filtered_df = sliced.iloc[keep_indices].reset_index(drop=True)
    figures = _plot_clustering(df, pca_data, cluster_labels, ellipse, keep_indices)

    if save_output:
        save_excel(filtered_df, output_path or DEFAULT_OUTPUT_PATH)
    figure_paths = (
        save_figures(figures, figures_dir or DEFAULT_FIGURES_DIR, stage_name="clustering")
        if save_figures_output
        else {}
    )

    metadata = default_metadata(filtered_df)
    metadata.update(
        {
            "threshold": threshold,
            "dbscan_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "dbscan_noise_points": int((cluster_labels == -1).sum()),
            "kept_model_rows": len(keep_indices),
            "pharma_points": len(pharmadf),
            "output_path": str(output_path or DEFAULT_OUTPUT_PATH),
        }
    )
    return {
        "data": filtered_df,
        "metadata": metadata,
        "artifacts": {
            "pca_data": pca_data,
            "pca_model": pca,
            "cluster_labels": cluster_labels,
            "mahalanobis": mahala,
            "ellipse": ellipse,
            "points_under_threshold": points_under_threshold,
            "keep_indices": keep_indices,
            "figures": figures,
            "figure_paths": figure_paths,
        },
    }
