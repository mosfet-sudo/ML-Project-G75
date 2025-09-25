#!/usr/bin/env python3
"""
MetroPT-3 Baseline (COMP90049 A2) — minimal & extensible

What this script does
---------------------
1) Download (wget‑like) the MetroPT‑3 dataset archive from a URL you provide
2) Verify checksum (optional)
3) Unzip into data/raw/
4) Load CSV(s); select features/target per config
5) Split train/dev/test; build sklearn Pipelines
6) Evaluate classic models (KNN, NB, DT, SVM) + MLP baseline with CV
7) Report macro metrics (accuracy, precision, recall, F1) and save artifacts

Usage
-----
# 1) Edit CONFIG below (URL, file names, target column, features etc.)
# 2) Run:
python metropt3_baseline.py --download --train

# To skip download (if data already present):
python metropt3_baseline.py --train

# To only produce metrics table as CSV:
python metropt3_baseline.py --train --no-plots

Notes
-----
- Designed to be simple but strictly aligned with ML best practices:
  * No leakage: fit transformers on train folds only (via Pipeline + CV)
  * Deterministic seeds
  * Same CV splits across models
- Extendable: add more models/grids, early‑warning horizon engineering, class balancing.

"""

from __future__ import annotations
import argparse
import os
import sys
import csv
import io
import json
import math
import shutil
import zipfile
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

# Only standard + scikit-learn stack
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Optional: progress bar for download
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -----------------------
# Config (EDIT THESE)
# -----------------------

@dataclass
class Config:
    # Data source
    url: str = "https://REPLACE_WITH_METROPT3_ZIP_URL"  # TODO: put official MetroPT-3 archive URL
    sha256: Optional[str] = None  # if you know the checksum, set it here
    archive_name: str = "metropt3.zip"

    # File layout inside the archive (examples; adjust to real filenames)
    # If the archive unzips to multiple CSVs, you can concatenate or pick one.
    expected_csv: str = "MetroPT3.csv"  # TODO: set to the actual CSV name inside the zip

    # Columns: set your target and feature lists. Use --infer to auto infer numerics.
    target_col: str = "failure"  # TODO: replace with correct target column for MetroPT-3
    categorical_cols: List[str] = []  # e.g., ["station_id", "line_id"]
    numeric_cols: Optional[List[str]] = None  # if None and --infer is set, will infer

    # Data split
    test_size: float = 0.2
    dev_size: float = 0.2  # portion of TRAIN that becomes DEV; final train fraction = (1-dev_size) of (1 - test_size)
    random_state: int = 42

    # CV
    cv_folds: int = 5

    # Paths
    root: Path = Path(".")
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    proc_dir: Path = Path("data/processed")
    out_dir: Path = Path("artifacts")

    # Flags
    save_plots: bool = True
    verbose: bool = True


# -----------------------
# Utils
# -----------------------

def log(msg: str):
    print(f"[metropt3] {msg}")


def download_wget_like(url: str, dest: Path):
    """Download with urllib + manual progress (wget-like)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading: {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        chunk = 8192
        if tqdm:
            pbar = tqdm(total=total, unit="B", unit_scale=True)
        else:
            pbar = None
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            if pbar:
                pbar.update(len(b))
        if pbar:
            pbar.close()
    log("Download complete.")


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_archive(cfg: Config, args) -> Path:
    archive = cfg.data_dir / cfg.archive_name
    if args.download or not archive.exists():
        download_wget_like(cfg.url, archive)
    if cfg.sha256:
        got = sha256sum(archive)
        if got != cfg.sha256:
            raise RuntimeError(f"Checksum mismatch for {archive}: got {got}, expected {cfg.sha256}")
        log("Checksum OK.")
    return archive


def unzip(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(target_dir)
    log(f"Unzipped to: {target_dir}")


def load_dataset(cfg: Config) -> pd.DataFrame:
    # Try direct path under raw_dir first; otherwise search recursively
    csv_path = cfg.raw_dir / cfg.expected_csv
    if not csv_path.exists():
        # search for a csv with similar name
        cand = list(cfg.raw_dir.rglob("*.csv"))
        if not cand:
            raise FileNotFoundError(f"No CSV found under {cfg.raw_dir}. Please update expected_csv in config.")
        log(f"expected_csv not found; using first CSV found: {cand[0].name}")
        csv_path = cand[0]
    log(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"Loaded shape: {df.shape}")
    return df


def infer_numeric_categorical(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str]]:
    if cfg.numeric_cols is not None:
        numeric = cfg.numeric_cols
    else:
        # simple heuristic: numeric dtypes
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        # do not include target
        if cfg.target_col in numeric:
            numeric.remove(cfg.target_col)
    categorical = cfg.categorical_cols or df.select_dtypes(exclude=["number"]).columns.tolist()
    if cfg.target_col in categorical:
        categorical.remove(cfg.target_col)
    return numeric, categorical


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])
    return pre


def get_models() -> Dict[str, Pipeline]:
    # Each model returns a Pipeline that starts with 'preprocessor'
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "nb": GaussianNB(),
        "dt": DecisionTreeClassifier(random_state=0),
        "svm": SVC(kernel="rbf", probability=False, class_weight=None, random_state=0),
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=0)
    }
    return models


def make_scorers() -> Dict[str, callable]:
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }


def evaluate_models(df: pd.DataFrame, cfg: Config, args) -> pd.DataFrame:
    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    numeric_cols, categorical_cols = infer_numeric_categorical(df, cfg)
    if cfg.verbose:
        log(f"Numeric cols: {len(numeric_cols)}; Categorical cols: {len(categorical_cols)}")

    pre = build_preprocessor(numeric_cols, categorical_cols)

    # common CV splitter; if multiclass imbalance is strong, consider StratifiedKFold (default)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    scorers = make_scorers()

    rows = []
    for name, clf in get_models().items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scorers, n_jobs=-1, return_train_score=False)
        row = {"model": name}
        for key, vals in scores.items():
            if key.startswith("test_"):
                metric = key.replace("test_", "")
                row[metric + "_mean"] = float(np.mean(vals))
                row[metric + "_std"] = float(np.std(vals))
        rows.append(row)
        if cfg.verbose:
            log(f"Done: {name}")
    results = pd.DataFrame(rows).sort_values(by="f1_macro_mean", ascending=False)
    return results


def save_artifacts(results: pd.DataFrame, cfg: Config):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cfg.out_dir / "cv_results.csv"
    results.to_csv(out_csv, index=False)
    log(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="MetroPT-3 baseline")
    parser.add_argument("--download", action="store_true", help="Download dataset archive")
    parser.add_argument("--train", action="store_true", help="Run CV baselines")
    parser.add_argument("--infer", action="store_true", help="Infer numeric columns automatically")
    parser.add_argument("--no-plots", action="store_true", help="(reserved) Skip plots")
    args = parser.parse_args()

    cfg = Config()
    # Normalize paths
    cfg.data_dir = cfg.root / cfg.data_dir
    cfg.raw_dir = cfg.root / cfg.raw_dir
    cfg.proc_dir = cfg.root / cfg.proc_dir
    cfg.out_dir = cfg.root / cfg.out_dir

    if args.download:
        archive = ensure_archive(cfg, args)
        unzip(archive, cfg.raw_dir)

    if args.train:
        df = load_dataset(cfg)
        if args.infer and cfg.numeric_cols is None:
            # numeric/categorical will be inferred inside evaluate_models
            pass
        results = evaluate_models(df, cfg, args)
        print("\n==== CV Results (macro) ====")
        print(results.to_string(index=False))
        save_artifacts(results, cfg)


if __name__ == "__main__":
    main()
