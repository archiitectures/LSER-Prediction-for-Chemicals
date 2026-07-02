Use this as continuation context for a new session.

**Goal**
Maintain and continue cleaning up `LSER_Project_Cleaned/` as the primary package and user-facing LSER workflow. The package now supports:
- direct stage execution
- a full orchestration pipeline
- custom feature hooks
- saved figures
- notebook-based interaction
- scoring for both existing and new molecules

The current task state is no longer “initial structural extraction.” The package has already been refactored into a usable internal API plus a notebook interface. Future work should build on that state rather than redoing the extraction.

**Repo Context**
Working directory:
- `/Users/Archi/Caltech_Local/ChE137 (local)`

This is not a git repo.

Legacy reference files remain untouched:
- `ChE137Week1.py` through `ChE137Week10.py`

Relevant root data files:
- `ChE 141 LSER Parameters 2025 Class Release.xlsx`
- `NewMolecules.xlsx`
- `SolubilityDataCited3.csv`
- `Data Output.xlsx`

Relevant package:
- `LSER_Project_Cleaned/`

**Current Package Structure**
Main package files:
- `LSER_Project_Cleaned/__init__.py`
- `LSER_Project_Cleaned/common.py`
- `LSER_Project_Cleaned/data_cleaning.py`
- `LSER_Project_Cleaned/data_exploration.py`
- `LSER_Project_Cleaned/clustering.py`
- `LSER_Project_Cleaned/model_training.py`
- `LSER_Project_Cleaned/model_inference.py`
- `LSER_Project_Cleaned/pipeline.py`
- `LSER_Project_Cleaned/LSER_Project_Interface.ipynb`
- `LSER_Project_Cleaned/CONTINUATION_SUMMARY.md`

Generated outputs directory now exists:
- `LSER_Project_Cleaned/outputs/figures/`

**What Was Learned From Legacy Scripts**
Legacy workflow was heavily import-time/global-state driven.

Recovered stage intent:
- `Week1`: base load, dedupe, invariant-column removal, categorical detection, missing-threshold column deletion, `ChemicalType` fill
- `Week9`: append `NewMolecules.xlsx`, build preprocessing feature(s), optional solubility merge
- `Week2`: `Vcalc` and `LSER_VX` merge behavior, LSER-missing row removal
- `Week3`: impossible negative-value cleanup for positive-only columns
- `Week4`: pairwise comparisons, similar-column detection, pairwise R², heatmap/scatter plots
- `Week5`: PCA and DBSCAN
- `Week6`: pharma/hormone PCA subset, Mahalanobis ellipse filtering, PCB downsampling, feature selection, dataset shaping
- `Week7`: `MLR`, `PCRMLR`, `PLS`, `LASSO`
- `Week8`: KPI and diagnostics
- `Week10`: relative scoring from predicted LSER values relative to basis molecule

Important legacy decisions that still inform the package:
- training targets are `LSER_E`, `LSER_S`, `LSER_A`, `LSER_B`
- PCA space uses `LSER_VX`, `LSER_E`, `LSER_S`, `LSER_A`
- default feature exclusion includes:
  `ChE137Number`, `PubChem CID`, `LSER_VX`, `LSER_E`, `LSER_S`, `LSER_A`, `LSER_B`
- `Mw(g/moL)` is still removed by default from training features
- inference uses weighted score coefficients:
  `[0.15, 0.25, 0.35, 0.35, -0.1]`

**What Has Been Implemented**

`common.py`
- Centralized relative paths
- Added `DEFAULT_OUTPUTS_DIR` and `DEFAULT_FIGURES_DIR`
- Added helpers:
  - `ensure_directory(...)`
  - `apply_feature_hooks(...)`
  - `save_figures(...)`
- Kept matplotlib non-interactive (`Agg`) and writable cache env vars

`data_cleaning.py`
- `run_data_cleaning(...)` now supports:
  - `data=...` for dataframe input
  - `feature_hooks=[...]`
  - `append_new_molecules=True/False`
- Feature hooks run after built-in preprocessing feature creation
- Default pipeline still does not integrate the solubility feature
- Returns consistent shape:
  - `data`
  - `metadata`
  - `artifacts`
- Metadata now includes `applied_feature_hooks`

`data_exploration.py`
- Still performs:
  - similar-column detection
  - `Vcalc` / `LSER_VX` merge
  - LSER-missing row removal
  - pairwise R² analysis
  - heatmap/scatter figures
- Added figure saving support:
  - `save_figures_output`
  - `figures_dir`
- Returns `figure_paths` in artifacts

`clustering.py`
- Row handling was normalized away from the old `iloc[2:-2]` convention
- Uses a full reset-index modeling frame instead of preserving first/last 2 rows
- Still performs:
  - PCB downsampling
  - PCA
  - DBSCAN
  - pharma/hormone subset construction
  - Mahalanobis filtering
- Added figure saving support and returns `figure_paths`

`model_training.py`
- Normalized training to use the full clustering output instead of `iloc[2:-2]`
- Added to outputs:
  - `training_feature_frame`
  - `training_target_frame`
- These are used by inference/imputation logic
- Added figure saving support and returns `figure_paths`

`model_inference.py`
- Fully rewritten from the initial extraction version
- Default inference now uses all 4 families:
  - `MLR`
  - `PCRMLR`
  - `PLS`
  - `LASSO`
- Added:
  - `lookup_molecule(...)`
  - `score_existing_molecules(...)`
  - `score_new_molecules_direct(...)`
  - `score_new_molecules_pipeline(...)`
- New-molecule scoring behavior:
  - direct-model path for training-ready rows
  - fuller pipeline path for raw-ish rows
  - missing required model features are imputed by:
    1. mean of training rows with matching `ChemicalType`
    2. global training-set mean if no same-type rows exist
- Direct scoring handles empty candidate sets cleanly

`pipeline.py`
- New orchestration file
- Added `run_full_pipeline(...)`
- This runs:
  - cleaning
  - exploration
  - clustering
  - training
  - optional inference
- Centralizes:
  - `feature_hooks`
  - `save_output`
  - `save_figures`
  - `figures_dir`
  - `basis_molecule`
  - `model_families`
  - `repeated_split_tests`

`__init__.py`
- Exports current public package API:
  - `run_data_cleaning`
  - `run_data_exploration`
  - `run_clustering`
  - `run_model_training`
  - `run_model_inference`
  - `run_full_pipeline`
  - `lookup_molecule`
  - `score_existing_molecules`
  - `score_new_molecules_direct`
  - `score_new_molecules_pipeline`

Notebook
- Added:
  - `LSER_Project_Cleaned/LSER_Project_Interface.ipynb`
- Notebook is intended to be the main user-facing interface
- It currently includes cells for:
  - running the full pipeline
  - inspecting KPI and figure outputs
  - molecule lookup
  - basis-molecule scoring
  - defining custom feature hooks
  - rerunning the pipeline with hooks
  - direct-model new-molecule scoring
  - fuller pipeline new-molecule scoring

**Current Public API**
Typical usage:

```python
from LSER_Project_Cleaned import (
    run_full_pipeline,
    score_existing_molecules,
    score_new_molecules_direct,
    score_new_molecules_pipeline,
)

pipeline = run_full_pipeline()
model_bundle = pipeline["model_training"]["model_bundle"]
```

Main orchestration:
- `run_full_pipeline(...)`

Stage entrypoints:
- `run_data_cleaning(...)`
- `run_data_exploration(...)`
- `run_clustering(...)`
- `run_model_training(...)`
- `run_model_inference(...)`

Inference/lookup helpers:
- `lookup_molecule(...)`
- `score_existing_molecules(...)`
- `score_new_molecules_direct(...)`
- `score_new_molecules_pipeline(...)`

Feature-hook contract:
```python
def custom_feature(df: pd.DataFrame) -> pd.DataFrame:
    ...
    return df
```

**Observed Shapes and Current Behavior**
Raw main Excel:
- `(3721, 103)`

`NewMolecules.xlsx`:
- `(2, 39)`

Earlier first-pass smoke run before normalization:
- cleaning output: `(3723, 41)`
- exploration output: `(2635, 40)`
- clustering output: `(276, 40)`

Current post-normalization smoke run:
- cleaning output: `(3723, 41)`
- exploration output: `(2635, 40)`
- clustering output: `(259, 40)`

This clustering change is expected because the old “preserve first 2 and last 2 rows” logic was removed during normalization.

**Validation Already Done**
Passed:
- package imports
- `python3 -m py_compile LSER_Project_Cleaned/*.py`
- `run_full_pipeline(save_output=False, save_figures=False)` end-to-end smoke run
- figure export into `LSER_Project_Cleaned/outputs/figures/`
- molecule lookup
- existing-molecule scoring
- direct new-molecule scoring
- fuller pipeline new-molecule scoring API execution

Confirmed inference families now returned:
- `MLR`
- `PCRMLR`
- `PLS`
- `LASSO`

Figure export confirmed for:
- exploration
- clustering
- training

Notebook file is valid JSON and present.

**Known Review Items / Risks**
These are the main places a future agent should inspect carefully:

1. Raw-ish new-molecule pipeline scoring can produce zero candidate rows.
This happens when clustering filters the new rows out before direct scoring. The method handles that cleanly, but this behavior has not been tuned for user expectations yet.

2. `score_new_molecules_pipeline(...)` is functional but still a first-pass implementation.
It currently appends raw-ish rows to the reference dataset, runs them through cleaning/exploration/clustering, then tries to recover the surviving “new” rows using an `InferenceSource` tag.

3. New-molecule pipeline scoring does not retrain models.
This is intentional. It uses the already trained model bundle and pushes new rows toward that feature space.

4. Solubility remains intentionally excluded.
Do not reintroduce it into the default pipeline unless the user explicitly requests it.

5. The notebook is a guided interface, not yet a polished product.
It uses the package APIs correctly, but it has not been optimized for UX or made especially robust against arbitrary user edits.

6. No exact legacy parity audit was done.
The user explicitly said not to spend time on exact parity validation against the old `ChE137Week*.py` scripts.

7. There are no formal regression tests yet.
Validation so far is smoke-test based via direct Python commands.

**Open Questions / Likely Next Work**
Good next tasks:

1. Add a proper regression test harness.
At minimum:
- import test
- `run_full_pipeline()` smoke test
- figure-path existence test
- inference family coverage test
- new-molecule imputation behavior test

2. Tighten `score_new_molecules_pipeline(...)`.
Decide what should happen when all new rows are filtered out:
- current behavior: return empty prepared/prediction sets
- possible improvement: return filtered-out reasons or pre/post row summaries

3. Improve notebook usability.
Possible next steps:
- cleaner sections and display tables
- prefilled examples for new molecules
- optional helper cells to render saved PNGs inline

4. Consider a dedicated feature-engineering module.
Right now feature hooks are supported through function injection. If experimentation expands, a real registry/module may be useful.

5. Add better row-level diagnostics for clustering and inference.
Especially useful for explaining why a new molecule was filtered out or heavily imputed.

**Suggested First Prompt For Next Session**
“Continue work on `LSER_Project_Cleaned`. Start by adding a minimal regression test harness for `run_full_pipeline`, figure export, and both new-molecule scoring paths. Then review `score_new_molecules_pipeline(...)`, especially the zero-candidate case when clustering filters out all new rows, and improve the returned diagnostics if needed.”
