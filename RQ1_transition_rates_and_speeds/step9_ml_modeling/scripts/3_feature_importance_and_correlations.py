#!/usr/bin/env python3
"""
Step 9.3: Feature Importance & Correlations
- Permutation importance for trained models (refit on full data per subset)
- Correlation matrix heatmaps for features
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
STEP9_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step9_ml_modeling"
DATA_DIR = STEP9_DIR / "data"
RESULTS_DIR = STEP9_DIR / "results"
VIZ_DIR = STEP9_DIR / "visualizations"

for d in [RESULTS_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_dataset():
    X = pd.read_parquet(DATA_DIR / "X_features.parquet")
    y = pd.read_parquet(DATA_DIR / "y_labels.parquet")["label"].astype(int)
    meta = pd.read_parquet(DATA_DIR / "meta.parquet")
    return X, y, meta


def fit_models(X, y):
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]),
        "rf": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "hgb": HistGradientBoostingClassifier(max_iter=500, random_state=42),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(X, y)
        fitted[name] = model
    return fitted


def do_permutation_importance(models, X, y, tag: str):
    out = {}
    for name, model in models.items():
        print(f"  Computing permutation importance for {name} [{tag}] ...")
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        importances = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
        out[name] = importances.head(50).round(6).to_dict()

        # Plot top-20
        top20 = importances.head(20)
        plt.figure(figsize=(8, 6))
        top20[::-1].plot(kind='barh')
        plt.title(f"Permutation Importance (Top 20): {name} [{tag}]")
        plt.tight_layout()
        save_path = VIZ_DIR / f"perm_importance_{name}_{tag}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_path}")

    with open(RESULTS_DIR / f"perm_importance_{tag}.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved JSON: {RESULTS_DIR / f'perm_importance_{tag}.json'}")


def correlation_heatmap(X, tag: str):
    print(f"  Computing correlation heatmap [{tag}] ...")
    corr = X.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, cbar_kws={'shrink': 0.5})
    plt.title(f"Feature Correlation Heatmap [{tag}]")
    plt.tight_layout()
    save_path = VIZ_DIR / f"correlation_heatmap_{tag}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def run_for_subset(tag: str, X, y, meta=None):
    print(f"Running importance/correlations for subset: {tag}")
    if tag in ("OSS", "OSS4SG") and meta is not None and 'project_type' in meta.columns:
        mask = meta['project_type'] == tag
        X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    models = fit_models(X, y)
    do_permutation_importance(models, X, y, tag)
    correlation_heatmap(X, tag)
    print(f"Completed subset: {tag}")


def main():
    print("=" * 60)
    print("STEP 9.3: FEATURE IMPORTANCE & CORRELATIONS")
    print("=" * 60)
    X, y, meta = load_dataset()
    for tag in ["ALL", "OSS", "OSS4SG"]:
        run_for_subset(tag, X, y, meta)
    print("Done.")


if __name__ == "__main__":
    main()


