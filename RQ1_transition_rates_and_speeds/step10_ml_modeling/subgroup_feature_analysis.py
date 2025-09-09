#!/usr/bin/env python3
"""
Subgroup feature analysis for Step 10
====================================

Computes per-group (OSS vs OSS4SG) correlations between features and target,
and subgroup-specific permutation importances using the best overall model
(RandomForest) trained on the full dataset.

Outputs:
- subgroup_correlations.csv
- subgroup_permutation_importance.csv
- subgroup_top3_summary.txt
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance


OUTPUT_DIR = Path("RQ1_transition_rates_and_speeds/step10_ml_modeling")
FEATURES_CSV = OUTPUT_DIR / "features_90day_comprehensive.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    # Ensure boolean -> int for correlation
    if df["became_core"].dtype != np.int64 and df["became_core"].dtype != np.int32:
        df["became_core"] = df["became_core"].astype(int)
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {
        "project_name",
        "project_type",
        "author_name",
        "author_email",
        "became_core",
        "first_commit_date",
    }
    # Keep only numeric feature columns
    feature_cols = [
        c for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]
    return feature_cols


def compute_subgroup_correlations(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for group in ["OSS", "OSS4SG"]:
        sub = df[df["project_type"] == group]
        y = sub["became_core"].astype(int).values
        for f in feature_cols:
            x = sub[f].values
            # If constant, correlation undefined; skip
            if np.all(x == x[0]) or len(sub) < 3:
                corr, p = np.nan, np.nan
            else:
                try:
                    corr, p = pearsonr(x, y)
                except Exception:
                    corr, p = np.nan, np.nan
            rows.append({
                "group": group,
                "feature": f,
                "correlation": corr,
                "abs_correlation": np.abs(corr) if pd.notna(corr) else np.nan,
                "p_value": p,
                "n": len(sub),
                "n_core": int(sub["became_core"].sum()),
            })
    corr_df = pd.DataFrame(rows)
    corr_df.sort_values(["group", "abs_correlation"], ascending=[True, False], inplace=True)
    return corr_df


def train_full_rf(df: pd.DataFrame, feature_cols: List[str]) -> RandomForestClassifier:
    X = df[feature_cols].values
    y = df["became_core"].astype(int).values
    # Reasonable, stable RF settings
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X, y)
    # Quick sanity check for overall AUC
    try:
        proba = rf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = np.nan
    print(f"Trained RF on full data. In-sample ROC-AUC: {auc:.3f}")
    return rf


def compute_subgroup_permutation_importance(
    df: pd.DataFrame, feature_cols: List[str], rf: RandomForestClassifier
) -> pd.DataFrame:
    rows: List[Dict] = []
    for group in ["OSS", "OSS4SG"]:
        sub = df[df["project_type"] == group]
        Xg = sub[feature_cols].values
        yg = sub["became_core"].astype(int).values
        if len(sub) < 10 or yg.sum() == 0 or yg.sum() == len(sub):
            # Degenerate group for permutation importance
            importances_mean = [np.nan] * len(feature_cols)
            importances_std = [np.nan] * len(feature_cols)
        else:
            r = permutation_importance(
                rf, Xg, yg, n_repeats=10, random_state=42, n_jobs=-1, scoring="roc_auc"
            )
            importances_mean = r.importances_mean
            importances_std = r.importances_std
        for f, m, s in zip(feature_cols, importances_mean, importances_std):
            rows.append({
                "group": group,
                "feature": f,
                "perm_importance_mean": m,
                "perm_importance_std": s,
                "n": len(sub),
                "n_core": int(sub["became_core"].sum()),
            })
    pidf = pd.DataFrame(rows)
    pidf.sort_values(["group", "perm_importance_mean"], ascending=[True, False], inplace=True)
    return pidf


def write_top3_summary(corr_df: pd.DataFrame, pidf: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    for group in ["OSS", "OSS4SG"]:
        lines.append(f"Group: {group}")
        lines.append("- Top 3 by correlation (abs):")
        top_corr = corr_df[corr_df["group"] == group].head(3)
        for _, r in top_corr.iterrows():
            lines.append(
                f"  • {r['feature']}: corr={r['correlation']:.3f} (p={r['p_value']:.2e})"
            )
        lines.append("- Top 3 by permutation importance (RF, scoring=roc_auc):")
        top_perm = pidf[pidf["group"] == group].head(3)
        for _, r in top_perm.iterrows():
            m = r["perm_importance_mean"]
            s = r["perm_importance_std"]
            if pd.isna(m):
                lines.append(f"  • insufficient signal to compute (degenerate group)")
            else:
                lines.append(f"  • {r['feature']}: mean={m:.5f} ± {s:.5f}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    feature_cols = get_feature_cols(df)

    corr_df = compute_subgroup_correlations(df, feature_cols)
    corr_path = OUTPUT_DIR / "subgroup_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Wrote {corr_path}")

    rf = train_full_rf(df, feature_cols)
    pidf = compute_subgroup_permutation_importance(df, feature_cols, rf)
    pidf_path = OUTPUT_DIR / "subgroup_permutation_importance.csv"
    pidf.to_csv(pidf_path, index=False)
    print(f"Wrote {pidf_path}")

    summary_path = OUTPUT_DIR / "subgroup_top3_summary.txt"
    write_top3_summary(corr_df, pidf, summary_path)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

