from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_DATASET_PATH = PROJECT_ROOT / "ChE 141 LSER Parameters 2025 Class Release.xlsx"
DEFAULT_NEW_MOLECULES_PATH = PROJECT_ROOT / "NewMolecules.xlsx"
DEFAULT_SOLUBILITY_PATH = PROJECT_ROOT / "SolubilityDataCited3.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "Data Output.xlsx"
DEFAULT_OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUTS_DIR / "figures"

LSER_COLUMNS = ["LSER_VX", "LSER_E", "LSER_S", "LSER_A", "LSER_B", "LSER_L"]
PCA_LSER_COLUMNS = ["LSER_VX", "LSER_E", "LSER_S", "LSER_A"]
TRAINING_TARGET_COLUMNS = ["LSER_E", "LSER_S", "LSER_A", "LSER_B"]
INFERENCE_SCORE_COLUMNS = ["LSER_E", "LSER_S", "LSER_A", "LSER_B", "LSER_VX"]
POSITIVE_ONLY_COLUMNS = [
    "LSER_VX",
    "LSER_A",
    "LSER_B",
    "Mw(g/moL)",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "PolarSurfaceArea(sqAngstrom)",
    "HeavyAtomCount",
    "CAtomsNumber",
    "HAtomsNumber",
    "OAtomsNumber",
    "SAtomsNumber",
    "NAtomsNumber",
    "PAtomsNumber",
    "ClAtoms",
    "BrAtoms",
    "FAtoms",
    "IAtoms",
    "TotalAtomsNumber",
    "SumAtomicVolume",
    "RingStruc turesNumber",
]


def load_excel(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_excel(Path(path) if path is not None else DEFAULT_DATASET_PATH)


def load_cleaned_output(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_excel(Path(path) if path is not None else DEFAULT_OUTPUT_PATH)


def load_new_molecules(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_excel(Path(path) if path is not None else DEFAULT_NEW_MOLECULES_PATH)


def load_solubility(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(Path(path) if path is not None else DEFAULT_SOLUBILITY_PATH)


def save_excel(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    output_path = Path(path) if path is not None else DEFAULT_OUTPUT_PATH
    df.to_excel(output_path, sheet_name="Sheet1", index=False)
    return output_path


def get_column_lists(df: pd.DataFrame) -> dict[str, list[str]]:
    quant_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    all_columns = df.columns.tolist()
    categorical_columns = [column for column in all_columns if column not in quant_columns]
    return {
        "quant_columns": quant_columns,
        "all_columns": all_columns,
        "categorical_columns": categorical_columns,
    }


def zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean(axis=0)) / frame.std(axis=0)


def zscore_with_stats(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    means = frame.mean(axis=0)
    stds = frame.std(axis=0).replace(0, np.nan)
    scaled = (frame - means) / stds
    return scaled, means, stds


def nan_to_zero(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values), copy=True, nan=0.0)


def normalize_series(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0 or pd.isna(span):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def default_metadata(df: pd.DataFrame) -> dict[str, Any]:
    metadata = get_column_lists(df)
    metadata["rows"] = df.index.tolist()
    return metadata


def ensure_directory(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def apply_feature_hooks(df: pd.DataFrame, feature_hooks: list[Any] | None = None) -> tuple[pd.DataFrame, list[str]]:
    applied_hooks: list[str] = []
    for hook in feature_hooks or []:
        hook_name = getattr(hook, "__name__", hook.__class__.__name__)
        df = hook(df.copy())
        applied_hooks.append(hook_name)
    return df, applied_hooks


def save_figures(
    figures: dict[str, plt.Figure],
    figures_dir: Path | str | None = None,
    stage_name: str | None = None,
    close_figures: bool = False,
) -> dict[str, str]:
    base_dir = ensure_directory(figures_dir or DEFAULT_FIGURES_DIR)
    if stage_name is not None:
        base_dir = ensure_directory(base_dir / stage_name)
    saved_paths: dict[str, str] = {}
    for figure_name, figure in figures.items():
        figure_path = base_dir / f"{figure_name}.png"
        figure.savefig(figure_path, bbox_inches="tight")
        saved_paths[figure_name] = str(figure_path)
        if close_figures:
            plt.close(figure)
    return saved_paths
