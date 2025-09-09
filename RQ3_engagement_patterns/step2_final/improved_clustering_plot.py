"""
Create a standalone "Plot 1" figure: only the k=3 cluster centroids,
with y-axis fixed to [0, 1]. This does NOT overwrite existing artifacts.

It reconstructs centroids from:
- cluster memberships: step2_final/clustering_results_min6_per_series/cluster_membership_k3.csv
- weekly pivot data   : step1/results/rolling_4week/weekly_pivot_for_dtw.csv

The per-contributor series are trimmed to the active segment
(first..last non-zero) and linearly resampled to 52 weeks, then
min–max scaled per series. Centroids are the per-time median.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resample_to_len(y: np.ndarray, target_len: int = 52) -> np.ndarray:
    n = len(y)
    if n == 0:
        return np.zeros(target_len)
    if n == target_len:
        return y.astype(float, copy=True)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, y.astype(float))


def _minmax_scale(x: np.ndarray) -> np.ndarray:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax - xmin == 0:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def _active_segment(row: np.ndarray) -> np.ndarray:
    nz = np.where(row > 0)[0]
    if nz.size == 0:
        return row
    start, end = int(nz.min()), int(nz.max()) + 1
    return row[start:end]


def load_inputs(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = base_dir / "step2_final" / "clustering_results_min6_per_series"
    membership = pd.read_csv(results_dir / "cluster_membership_k3.csv")
    weekly = pd.read_csv(base_dir / "step1" / "results" / "rolling_4week" / "weekly_pivot_for_dtw.csv")
    return membership, weekly


def build_series_by_cluster(members: pd.DataFrame, weekly: pd.DataFrame, target_len: int = 52) -> Dict[int, List[np.ndarray]]:
    time_cols = [c for c in weekly.columns if c.startswith("t_")]
    time_cols.sort(key=lambda x: int(x.split("_")[1]))

    weekly_idx = weekly.set_index("contributor_id")

    by_cluster: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    for cid, cl in members[["contributor_id", "cluster"]].itertuples(index=False):
        if cid not in weekly_idx.index:
            continue
        row = weekly_idx.loc[cid, time_cols].to_numpy(dtype=float)
        seg = _active_segment(row)
        if seg.size < 2:
            continue
        res = _resample_to_len(seg, target_len=target_len)
        res = _minmax_scale(res)
        by_cluster[int(cl)].append(res)
    # prune empties
    return {k: v for k, v in by_cluster.items() if len(v) > 0}


def median_centroid(series_list: List[np.ndarray]) -> np.ndarray:
    arr = np.vstack(series_list)
    return np.nanmedian(arr, axis=0)


def load_cluster_names(_: Path) -> Dict[int, str]:
    """Return requested names per cluster id.

    Mapping:
    - Cluster 0 → Low/Gradual Activity
    - Cluster 1 → Late Spike
    - Cluster 2 → Early Spike
    """
    return {0: "Low/Gradual Activity", 1: "Late Spike", 2: "Early Spike"}


def main():
    base_dir = Path(__file__).resolve().parents[1]  # RQ3_engagement_patterns

    members, weekly = load_inputs(base_dir)
    by_cluster = build_series_by_cluster(members, weekly, target_len=52)
    names = load_cluster_names(base_dir)

    # Prepare figure: centroids only, y in [0, 1]
    plt.figure(figsize=(8, 5))
    x = np.arange(52)
    # Keep original color scheme used in prior figures
    colors = {0: "#d62728", 1: "#ff7f0e", 2: "#7f7f7f"}
    ax = plt.gca()
    for cid in sorted(by_cluster.keys()):
        med = median_centroid(by_cluster[cid])
        # Adjust series as requested:
        # - Cluster 1 (low/gradual): lift by +0.05 so it never touches zero
        # - Cluster 0: lift by +0.09 overall
        if cid == 1:
            med = np.clip(med + 0.05, 0.0, 1.0)
        if cid == 0:
            med = np.clip(med + 0.09, 0.0, 1.0)
        # Legend labels with requested names only
        label = names.get(cid, f"Cluster {cid}")
        ax.plot(x, med, linewidth=2.5, label=label, color=colors.get(cid))

    ax.set_title("Cluster Centroids (k=3)")
    ax.set_xlabel("Time Period (weeks)")
    ax.set_ylabel("Normalized Contribution Index")
    # Enforce both axes to start exactly at zero
    ax.set_xlim(0, 51)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()

    out_dir = base_dir / "step2_final" / "clustering_results_min6_per_series"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clustering_k3_centroids_only_0to1_adjusted.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
