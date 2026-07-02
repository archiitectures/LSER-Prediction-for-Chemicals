from __future__ import annotations

from pathlib import Path
from typing import Any

from .clustering import run_clustering
from .common import DEFAULT_FIGURES_DIR, DEFAULT_OUTPUT_PATH
from .data_cleaning import run_data_cleaning
from .data_exploration import run_data_exploration
from .model_inference import run_model_inference
from .model_training import run_model_training


def run_full_pipeline(
    dataset_path: Path | str | None = None,
    new_molecules_path: Path | str | None = None,
    output_path: Path | str | None = None,
    feature_hooks: list[Any] | None = None,
    save_output: bool = True,
    save_figures: bool = True,
    figures_dir: Path | str | None = None,
    model_families: list[str] | None = None,
    repeated_split_tests: int = 0,
    run_inference: bool = True,
    chemtype: str = "pharma",
    basis_molecule: str = "Diazepam",
) -> dict[str, Any]:
    chosen_output_path = output_path or DEFAULT_OUTPUT_PATH
    chosen_figures_dir = figures_dir or DEFAULT_FIGURES_DIR

    cleaning = run_data_cleaning(
        dataset_path=dataset_path,
        new_molecules_path=new_molecules_path,
        output_path=chosen_output_path,
        save_output=save_output,
        feature_hooks=feature_hooks,
    )
    exploration = run_data_exploration(
        cleaning["data"],
        output_path=chosen_output_path,
        save_output=save_output,
        save_figures_output=save_figures,
        figures_dir=chosen_figures_dir,
    )
    clustering = run_clustering(
        exploration["data"],
        output_path=chosen_output_path,
        save_output=save_output,
        save_figures_output=save_figures,
        figures_dir=chosen_figures_dir,
    )
    training = run_model_training(
        clustering["data"],
        repeated_split_tests=repeated_split_tests,
        save_figures_output=save_figures,
        figures_dir=chosen_figures_dir,
    )

    inference = None
    if run_inference:
        inference = run_model_inference(
            training["model_bundle"],
            chemtype=chemtype,
            basismol=basis_molecule,
            model_families=model_families,
        )

    return {
        "data_cleaning": cleaning,
        "data_exploration": exploration,
        "clustering": clustering,
        "model_training": training,
        "model_inference": inference,
        "artifacts": {
            "output_path": str(chosen_output_path),
            "figures_dir": str(chosen_figures_dir),
        },
    }
