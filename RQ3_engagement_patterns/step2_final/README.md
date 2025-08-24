# RQ3 Step 2 (Final): Time‑series Clustering Outputs

This folder contains the finalized clustering results used by Step 3. The decisions below were validated and fixed against earlier experiments to ensure clean pre‑core behavior and reproducibility.

## Final configuration (frozen)
- Input: `../step1/results/rolling_4week/weekly_pivot_for_dtw.csv` (pre‑core only)
  - Weeks are relative to each contributor
  - Minimum activity filter: ≥6 active weeks
  - Series length normalization: 52 weeks (relative indexing)
  - Distance: DTW; Algorithm: K‑means; Scaling: per‑series normalization

## Artifacts
- `clustering_results_min6_per_series/cluster_membership_k3.csv` — Final membership used in Step 3.1
- `clustering_results_min6_per_series/clustering_k3_results.json` — k=3 metrics (silhouette, sizes)
- `clustering_results_min6_per_series/clustering_k3_analysis.png` — k=3 visualization
- `clustering_results_min6_per_series/per_cluster_k3/cluster_*.png` — per‑cluster series plots
- Also provided for k=4: corresponding JSON, PNG, and per‑cluster plots

Notes
- Earlier experimental variants are archived in `../step2.1_experimental_archive.zip` and are not used.
