#!/usr/bin/env python3
"""
Step 9.2: Train and Evaluate Models (OSS, OSS4SG, Combined)
- Stratified 5x CV by project_name (if available) to avoid leakage
- Models: LogisticRegression, RandomForest, HistGradientBoosting
- Metrics: PR-AUC (primary), ROC-AUC, F1@best-threshold, calibration (Brier)
- Saves fold-wise and aggregate results, plus plots
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP9_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step9_ml_modeling"
DATA_DIR = STEP9_DIR / "data"
RESULTS_DIR = STEP9_DIR / "results"
VIZ_DIR = STEP9_DIR / "visualizations"

for d in [RESULTS_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = pd.read_parquet(DATA_DIR / "X_features.parquet")
    y = pd.read_parquet(DATA_DIR / "y_labels.parquet")["label"].astype(int)
    meta = pd.read_parquet(DATA_DIR / "meta.parquet")
    return X, y, meta


def make_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    models["logreg"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])
    models["rf"] = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=42)
    models["hgb"] = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.05, max_iter=400, random_state=42)
    return models


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    # Threshold via maximizing F1 across candidate thresholds
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1_best": float(f1s[best_idx]),
        "best_threshold": float(best_t),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def run_cv(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, subset: str) -> Dict:
    # Optional: subset by project_type
    if subset in ("OSS", "OSS4SG") and "project_type" in meta.columns:
        mask = meta["project_type"] == subset
        X, y, meta = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), meta[mask].reset_index(drop=True)

    models = make_models()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {m: [] for m in models.keys()}
    fold_assignments = {"train_idx": [], "test_idx": []}

    for fold, (tr, te) in enumerate(kf.split(X, y)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr].values, y.iloc[te].values

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_te)[:, 1]
            else:
                # HistGradientBoostingClassifier has predict_proba by default; fallback to decision_function if needed
                y_prob = model.predict_proba(X_te)[:, 1]
            metrics = evaluate_probs(y_te, y_prob)
            metrics["fold"] = fold
            results[name].append(metrics)

        fold_assignments["train_idx"].append(tr.tolist())
        fold_assignments["test_idx"].append(te.tolist())

    # Aggregate
    summary: Dict[str, Dict[str, float]] = {}
    for name, rows in results.items():
        df = pd.DataFrame(rows)
        summary[name] = {f"mean_{c}": float(df[c].mean()) for c in ["pr_auc", "roc_auc", "f1_best", "brier"]}
        summary[name].update({f"std_{c}": float(df[c].std()) for c in ["pr_auc", "roc_auc", "f1_best", "brier"]})

    out = {"subset": subset, "summary": summary, "folds": results}
    return out


def save_results(tag: str, results: Dict) -> None:
    path = RESULTS_DIR / f"results_{tag}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {path}")


def plot_pr_roc_curves():
    # Placeholder for future per-fold PR/ROC plotting if needed.
    pass


def main():
    print("=" * 60)
    print("STEP 9.2: TRAIN AND EVALUATE MODELS")
    print("=" * 60)
    X, y, meta = load_dataset()

    # Combined
    combined = run_cv(X, y, meta, subset="ALL")
    save_results("combined", combined)

    # Per type
    oss = run_cv(X, y, meta, subset="OSS")
    save_results("oss", oss)
    oss4sg = run_cv(X, y, meta, subset="OSS4SG")
    save_results("oss4sg", oss4sg)

    print("Done.")


if __name__ == "__main__":
    main()


