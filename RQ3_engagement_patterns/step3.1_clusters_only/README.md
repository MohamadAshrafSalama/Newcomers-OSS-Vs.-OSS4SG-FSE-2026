# RQ3 Step 3.1: Pattern Effectiveness from Step 2 Clusters Only

This step re-runs the pattern effectiveness analysis using ONLY the finalized Step 2 cluster membership as input for patterns, combined with RQ1 transition outcomes to compute time to core. No Step 1 time series are used here.

## Inputs
- Step 2_final membership: `../step2_final/clustering_results_min6_per_series/cluster_membership_k3.csv`
- RQ1 transitions: `../../RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv`

## Method
1. Load Step 2 clusters and RQ1 transitions.
2. Normalize identifiers (use `contributor_email` where available; fallback to `contributor_id`).
3. Inner-join to keep contributors present in both datasets who became core (became_core == True).
4. Compute the target: `weeks_to_core` from RQ1 transitions (`weeks_to_core` column). If not present, compute from `first_core_week - first_commit_week`.
5. Map cluster ids to pattern names: 0=Early Spike, 1=Sustained Activity, 2=Low/Gradual Activity.
6. Form groups by `project_type × pattern`, summarize stats, run Kruskal–Wallis and Dunn’s test, then assign Scott–Knott-like ranks.
7. Save results and a compact visualization colored by distinct ranks.

## How to Run
```bash
cd "RQ3_engagement_patterns/step3.1_clusters_only"
python step3_1_clusters_only_analysis.py
```

## Outputs
- `results/group_statistics.csv`
- `results/scott_knott_results.csv`
- `results/pattern_effectiveness.png`
- `results/analysis_report.txt`
