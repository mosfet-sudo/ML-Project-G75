# MetroPT-3 Baseline (COMP90049 A2)

Minimal, strict, and extendable baseline for your predictive-maintenance project.

## How to run (locally)

1. **Install deps**

```bash
python -m venv .venv && source .venv/bin/activate  # or on Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn tqdm
```

2. **Edit config** inside `metropt3_baseline.py`:

- Set `url` to the official MetroPT‑3 archive (ZIP) download link.
- Optionally set `sha256` if you have the checksum.
- Set `expected_csv` to the actual CSV inside the archive.
- Set `target_col` and (optionally) `categorical_cols` / `numeric_cols`.

3. **Download + unzip**

```bash
python metropt3_baseline.py --download
```

4. **Train baselines (5-fold CV)**

```bash
python metropt3_baseline.py --train --infer
```

Outputs: `artifacts/cv_results.csv`

## What this baseline enforces

- Train-only fitting of preprocessors (no leakage), by using `Pipeline` & `ColumnTransformer`.
- Identical CV splits across models via a common `StratifiedKFold`.
- Deterministic seeds.
- Macro metrics: accuracy / precision / recall / F1.

## Where to extend

- **Early-warning horizon**: add a feature builder that aggregates time windows.
- **Class imbalance**: swap `SVC(class_weight="balanced")`, try `class_weight` for DT, or integrate imbalanced-learn.
- **Hyperparameter tuning**: replace `cross_validate` with `GridSearchCV` / `RandomizedSearchCV`.
- **Neural model**: adjust MLP or add PyTorch model externally (still respecting CV & folds).
- **Multiple datasets**: add CLI to switch between MetroPT‑3 / NASA CMAPSS etc.

Good luck!
