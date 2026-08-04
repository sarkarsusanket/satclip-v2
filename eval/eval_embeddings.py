"""
eval_embeddings.py

Evaluate a set of location-embedding models against a set of CSV datasets.

Datasets are declared in config.py as DatasetSpec(name, path, task, pred, target):
  - `pred`   = (start_col, end_col), inclusive, 0-indexed column range that
               contains the predictor block (must include a lat and a lon
               column, auto-detected by name within that range).
  - `target` = (start_col, end_col), inclusive, 0-indexed column range that
               contains one or more target columns. Every column in this
               range is evaluated as a separate row in the report.
  - `task`   = "reg" or "class", applied to every target column in `target`.
               (If one CSV has a mix of regression and classification
               targets, declare it twice with two different `target` ranges
               and two DatasetSpecs -- see README.)

For every (dataset, target_column, model) combination this script:
  1. embeds the lon/lat pairs with the model (once per dataset, reused
     across all its targets),
  2. runs 5-fold cross-validation with a default-params LightGBM
     (LGBMRegressor for "reg", LGBMClassifier for "class"), using ONLY the
     embedding as X,
  3. scores with R2 (reg) or macro-F1 (class),
and writes a single tidy CSV report: rows = (dataset, target), columns =
model names.

See config.py for how to declare datasets/models, and README.md for the
full picture.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from models import EmbeddingModel

LAT_NAMES = {"lat", "latitude", "y", "lat_wgs84"}
LON_NAMES = {"lon", "lng", "long", "longitude", "x", "lon_wgs84"}

N_FOLDS = 5
RANDOM_STATE = 42

TASK_ALIASES = {"reg": "regression", "class": "classification"}


@dataclass
class DatasetSpec:
    name: str
    path: str
    task: str  # "reg" or "class" -- applies to every column in `target`
    pred: Tuple[int, int]  # inclusive column-index range holding lat/lon
    target: Tuple[int, int]  # inclusive column-index range of target col(s)

    def __post_init__(self) -> None:
        if self.task not in TASK_ALIASES:
            raise ValueError(f"{self.name}: task must be 'reg' or 'class', got {self.task!r}")


def _find_coord_columns(df: pd.DataFrame, pred: Tuple[int, int]) -> Tuple[str, str]:
    """Locate lat/lon columns by name, restricted to the `pred` column range."""
    start, end = pred
    block = df.columns[start : end + 1]
    block_lower = {c.lower(): c for c in block}
    lat_col = next((block_lower[c] for c in LAT_NAMES if c in block_lower), None)
    lon_col = next((block_lower[c] for c in LON_NAMES if c in block_lower), None)
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not find lat/lon columns among pred range {pred} = {list(block)}. "
            f"Expected one of {sorted(LAT_NAMES)} and one of {sorted(LON_NAMES)}."
        )
    return lat_col, lon_col


def _target_columns(df: pd.DataFrame, target: Tuple[int, int]) -> List[str]:
    start, end = target
    return list(df.columns[start : end + 1])


def _score_regression(emb: np.ndarray, y: np.ndarray) -> float:
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = cross_val_predict(LGBMRegressor(verbose=-1), emb, y, cv=kfold, n_jobs=-1)
    return r2_score(y, preds)


def _score_classification(emb: np.ndarray, y: np.ndarray) -> float:
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = cross_val_predict(LGBMClassifier(verbose=-1), emb, y, cv=kfold, n_jobs=-1)
    return f1_score(y, preds, average="macro")


def evaluate_target(emb: np.ndarray, y: pd.Series, task_type: str) -> Tuple[float, int]:
    """Returns (metric, n_samples_used). Rows with a missing target are dropped."""
    mask = y.notna().to_numpy()
    emb_clean, y_clean = emb[mask], y.to_numpy()[mask]
    if task_type == "classification":
        # LightGBM wants contiguous integer labels
        _, y_encoded = np.unique(y_clean, return_inverse=True)
        counts = np.bincount(y_encoded)
        if counts.min() < N_FOLDS:
            raise ValueError(
                f"Smallest class has {counts.min()} samples, need >= {N_FOLDS} "
                f"for {N_FOLDS}-fold stratified CV."
            )
        score = _score_classification(emb_clean, y_encoded)
    else:
        score = _score_regression(emb_clean, y_clean.astype(float))
    return score, len(y_clean)


def run_evaluation(
    datasets: Sequence[DatasetSpec],
    models: Sequence[EmbeddingModel],
    output_csv: str = "eval_report.csv",
) -> pd.DataFrame:
    """
    datasets: list of DatasetSpec (see config.py).
    models: list of EmbeddingModel (see models.py).

    Returns the report DataFrame and also writes it to `output_csv`
    (incrementally, after every target, so a long run can be killed/resumed
    without losing prior results).
    """
    rows: List[dict] = []
    scores: Dict[str, List[float]] = {m.name: [] for m in models}

    for spec in datasets:
        df = pd.read_csv(spec.path)
        lat_col, lon_col = _find_coord_columns(df, spec.pred)
        target_cols = _target_columns(df, spec.target)
        task_type = TASK_ALIASES[spec.task]

        lon = df[lon_col].to_numpy(dtype=float)
        lat = df[lat_col].to_numpy(dtype=float)

        print(f"\n=== {spec.name} ({spec.path}) | task={spec.task} | targets={target_cols} ===")

        # embed once per model, reused across every target in this dataset
        embeddings: Dict[str, np.ndarray] = {}
        for model in models:
            print(f"[{spec.name}] embedding with {model.name} ({len(df)} points)...")
            embeddings[model.name] = model.embed(lon, lat)

        for target in target_cols:
            y = df[target]
            print(f"[{spec.name}] target={target!r} -> {task_type}")

            row = {"dataset": spec.name, "target": target, "task_type": spec.task, "n_samples": None}
            for model in models:
                try:
                    score, n = evaluate_target(embeddings[model.name], y, task_type)
                    scores[model.name].append(score)
                    row["n_samples"] = n
                except Exception as e:  # one bad model/target combo shouldn't kill the run
                    print(f"  ! {model.name} failed on {spec.name}/{target}: {e}")
                    scores[model.name].append(np.nan)
            rows.append(row)

            # write incrementally after every target
            report = pd.DataFrame(rows)
            for model_name, vals in scores.items():
                report[model_name] = vals
            report.to_csv(output_csv, index=False)

    report = pd.DataFrame(rows)
    for model_name, vals in scores.items():
        report[model_name] = vals
    report.to_csv(output_csv, index=False)
    print(f"\nWrote report to {output_csv}")
    return report