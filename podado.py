import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d

import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# -------------------------------------------------------------------------
# TOGGLE SETTINGS
# -------------------------------------------------------------------------
use_forced_k = True  # If True, we skip silhouette approach and just use 'forced_k'
forced_k = 4          # If use_forced_k=True, pick whichever you want: 3, 4, etc.

# Path to your data
file_path = "/Users/mohamadashraf/Desktop/Projects/Jenny/df_all_commits_with_stats_clean.csv"

# -------------------------------------------------------------------------
# 1. Read & Prep
# -------------------------------------------------------------------------
df = pd.read_csv(file_path)

df['commit_date'] = pd.to_datetime(df['commit_date'])
df = df.sort_values('commit_date')

# Create a column "core_churn" which is churn *only* for core commits
# If is_core == True => core_churn = churn, else 0
df['core_churn'] = np.where(df['is_core'] == True, df['churn'], 0.0)

# -------------------------------------------------------------------------
# 2. Strict Stage Logic
#    - If total lifetime < 7 years => discard project entirely
#    - Otherwise keep only these exact slices:
#         early = [start, start+7)
#         middle = [start+7, start+10)
#         late = [start+10, start+13)
# -------------------------------------------------------------------------
def filter_and_label_stages(project_df):
    """
    Given all commits for a single project, returns a new df with
    stage labels assigned to each commit. We discard any commits
    outside the relevant slice or if total years <7.
    """
    project_df = project_df.copy()
    if project_df.empty:
        return project_df

    project_start = project_df['commit_date'].min()
    project_end   = project_df['commit_date'].max()
    total_years   = (project_end - project_start).days / 365.0

    # If <7 years, discard entire project
    if total_years < 7:
        return project_df.iloc[0:0]  # empty

    # Build cutoff dates:
    early_end   = project_start + pd.DateOffset(years=7)
    middle_end  = early_end + pd.DateOffset(years=3)   # start+10
    late_end    = middle_end + pd.DateOffset(years=3)  # start+13

    # We'll create a 'stage' column. By default, empty string => discard
    project_df['stage'] = ''

    # Always keep 'early' if >=7 total years => [start, start+7)
    # Commits in that range => stage = 'early'
    mask_early = (project_df['commit_date'] >= project_start) & \
                 (project_df['commit_date'] < early_end)
    project_df.loc[mask_early, 'stage'] = 'early'

    # Keep 'middle' only if total_years >=10 => [start+7, start+10)
    if total_years >= 10:
        mask_mid = (project_df['commit_date'] >= early_end) & \
                   (project_df['commit_date'] < middle_end)
        project_df.loc[mask_mid, 'stage'] = 'middle'

    # Keep 'late' only if total_years >=13 => [start+10, start+13)
    if total_years >= 13:
        mask_late = (project_df['commit_date'] >= middle_end) & \
                    (project_df['commit_date'] < late_end)
        project_df.loc[mask_late, 'stage'] = 'late'

    # Discard anything not labeled (outside final window if project is older than 13, or partial leftover if <10 or <13)
    project_df = project_df[project_df['stage'] != '']

    return project_df

# -------------------------------------------------------------------------
# 3. Apply to each project; build a new DataFrame
# -------------------------------------------------------------------------
projects = df['project'].unique()
all_filtered = []

for project in projects:
    sub_df = df[df['project'] == project]
    labeled_sub_df = filter_and_label_stages(sub_df)
    if not labeled_sub_df.empty:
        all_filtered.append(labeled_sub_df)

df_filtered = pd.concat(all_filtered, ignore_index=True)
df_filtered = df_filtered.sort_values('commit_date')

# -------------------------------------------------------------------------
# 4. Create monthly time series
#    Group by (project, stage, month) => sum churn & core_churn
#    Then drop months with churn=0 to avoid ratio=NaN
# -------------------------------------------------------------------------
df_filtered['year_month'] = df_filtered['commit_date'].dt.to_period('M')

monthly_stats = df_filtered.groupby(['project', 'stage', 'year_month']).agg({
    'churn': 'sum',       # sum of total churn for that month
    'core_churn': 'sum'   # sum of churn from core commits
}).reset_index()

# Drop rows where churn=0
monthly_stats = monthly_stats[monthly_stats['churn'] != 0].copy()

# ratio
monthly_stats['core_churn_ratio'] = monthly_stats['core_churn'] / monthly_stats['churn']

# Drop any leftover NaN
monthly_stats.dropna(subset=['core_churn_ratio'], inplace=True)

# Convert to Timestamp for sorting
monthly_stats['date'] = monthly_stats['year_month'].dt.to_timestamp()

# -------------------------------------------------------------------------
# 5. Build (project, stage) time series
# -------------------------------------------------------------------------
time_series_collection = []
unique_projects = monthly_stats['project'].unique()

for project in unique_projects:
    stages_for_project = monthly_stats[monthly_stats['project'] == project]['stage'].unique()
    for stg in stages_for_project:
        sub = monthly_stats[(monthly_stats['project'] == project) &
                            (monthly_stats['stage'] == stg)].copy()
        if not sub.empty:
            sub_sorted = sub.sort_values('date')
            time_series_collection.append({
                'project': project,
                'stage': stg,
                'time_series': sub_sorted,
                'label': f"{project} - {stg}"
            })

# -------------------------------------------------------------------------
# 6. Interpolate each time series to a fixed length (linear interpolation)
#    - skip series <2 points
# -------------------------------------------------------------------------
def interpolate_time_series(ts_collection):
    """
    Returns a list of dicts, each with:
      - project
      - stage
      - core_churn_ratio (interpolated array)
      - label
    All time series are interpolated to the length = max_length among them.
    """
    # find max length among series that have >=2 points
    valid_lengths = [len(item['time_series']) for item in ts_collection if len(item['time_series']) >= 2]
    if not valid_lengths:
        return []
    max_len = max(valid_lengths)

    result = []
    for item in ts_collection:
        ts = item['time_series']
        if len(ts) < 2:
            # skip single-point or empty
            continue
        x = np.arange(len(ts))  # 0..len(ts)-1
        x_new = np.linspace(0, len(ts) - 1, max_len)
        y = ts['core_churn_ratio'].values

        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
        if np.isnan(y_new).any():
            # skip if interpolation produced NaN for some reason
            continue

        result.append({
            'project': item['project'],
            'stage': item['stage'],
            'core_churn_ratio': y_new,
            'label': item['label']
        })
    return result

interpolated_time_series = interpolate_time_series(time_series_collection)

# -------------------------------------------------------------------------
# 7. Prepare matrix X for clustering => each row is a time series
# -------------------------------------------------------------------------
if len(interpolated_time_series) < 2:
    print("Not enough valid time series after filtering. Exiting.")
    exit()

X = np.array([ts['core_churn_ratio'] for ts in interpolated_time_series])
# Double-check for any remaining NaN
mask_nan = np.isnan(X).any(axis=1)
if np.any(mask_nan):
    # drop those
    keep_rows = [i for i, flag in enumerate(mask_nan) if not flag]
    X = X[keep_rows]
    interpolated_time_series = [interpolated_time_series[i] for i in keep_rows]

if len(interpolated_time_series) < 2:
    print("Not enough rows for KMeans after removing NaNs. Exiting.")
    exit()

# -------------------------------------------------------------------------
# 8. Scale & KMeans
# -------------------------------------------------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

if use_forced_k:
    best_k = forced_k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    if len(set(cluster_labels)) > 1:
        best_score = silhouette_score(X_scaled, cluster_labels)
    else:
        best_score = -1
else:
    possible_ks = range(2, min(10, len(X_scaled) + 1))
    best_k, best_score = 2, -1
    sil_scores = []
    for k in possible_ks:
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = kmeans_tmp.fit_predict(X_scaled)
        if len(set(labels_tmp)) > 1:
            score = silhouette_score(X_scaled, labels_tmp)
        else:
            score = -1
        sil_scores.append((k, score))
    if sil_scores:
        best_k, best_score = max(sil_scores, key=lambda x: x[1])

    # final KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

# Attach cluster info
for i, ts in enumerate(interpolated_time_series):
    ts['cluster'] = cluster_labels[i]

# -------------------------------------------------------------------------
# 9. Compute centroids & medoids
# -------------------------------------------------------------------------
def compute_medoids(X_scaled, labels, n_clusters):
    medoids = {}
    for c in range(n_clusters):
        idx_in_cluster = np.where(labels == c)[0]
        if len(idx_in_cluster) == 0:
            continue
        if len(idx_in_cluster) == 1:
            # single point => trivially the medoid
            medoids[c] = (idx_in_cluster[0], X_scaled[idx_in_cluster[0]])
        else:
            cluster_data = X_scaled[idx_in_cluster]
            dist_matrix = pairwise_distances(cluster_data, metric='euclidean')
            sum_dists = dist_matrix.sum(axis=1)
            medoid_idx_local = np.argmin(sum_dists)
            global_idx = idx_in_cluster[medoid_idx_local]
            medoids[c] = (global_idx, X_scaled[global_idx])
    return medoids

centroids = []
for c_id in range(best_k):
    cluster_points = X_scaled[cluster_labels == c_id]
    if len(cluster_points) == 0:
        continue
    centroid_scaled = cluster_points.mean(axis=0)
    centroid_orig   = scaler.inverse_transform(centroid_scaled.reshape(1, -1))[0]
    centroids.append((c_id, centroid_scaled, centroid_orig))

medoids_dict = compute_medoids(X_scaled, cluster_labels, best_k)
medoids_list = []
for c in sorted(medoids_dict.keys()):
    g_idx, med_scaled = medoids_dict[c]
    med_orig = scaler.inverse_transform(med_scaled.reshape(1, -1))[0]
    medoids_list.append((c, g_idx, med_scaled, med_orig))

# -------------------------------------------------------------------------
# 10. Summaries: how many series in each stage-cluster
# -------------------------------------------------------------------------
stage_cluster_counts = {}
for ts in interpolated_time_series:
    stg = ts['stage']
    cl  = ts['cluster']
    if stg not in stage_cluster_counts:
        stage_cluster_counts[stg] = {}
    if cl not in stage_cluster_counts[stg]:
        stage_cluster_counts[stg][cl] = 0
    stage_cluster_counts[stg][cl] += 1

stage_cluster_percentages = {}
for stg, c_dict in stage_cluster_counts.items():
    total = sum(c_dict.values())
    stage_cluster_percentages[stg] = { cl: (count/total)*100 for cl, count in c_dict.items()}

# -------------------------------------------------------------------------
# 11. Plot
# -------------------------------------------------------------------------
plt.figure(figsize=(20, 15))

# (A) All raw time series color-coded by cluster
plt.subplot(3, 1, 1)
for c_id in range(best_k):
    cluster_series = [ts for ts in interpolated_time_series if ts['cluster'] == c_id]
    for ts_data in cluster_series:
        plt.plot(ts_data['core_churn_ratio'], alpha=0.2, color=f'C{c_id}')
plt.title(f"Raw Time Series by Cluster (k={best_k}, silhouette={best_score:.3f})")
plt.ylabel("Core Churn Ratio")
plt.grid(True, linestyle='--', alpha=0.7)

# (B) Centroids & Medoids (original scale)
plt.subplot(3, 1, 2)
for (c_id, centroid_scaled, centroid_orig) in centroids:
    plt.plot(centroid_orig, linewidth=3, label=f"Cluster {c_id} centroid", color=f"C{c_id}")
for (c_id, g_idx, med_scaled, med_orig) in medoids_list:
    plt.plot(med_orig, '--', linewidth=2, label=f"Cluster {c_id} medoid", color=f"C{c_id}")
plt.title("Centroids & Medoids (Original Scale)")
plt.ylabel("Core Churn Ratio")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# (C) Distribution of clusters by stage
plt.subplot(3, 1, 3)
stages = sorted(stage_cluster_percentages.keys())
clusters = range(best_k)

width = 0.8 / best_k
x_positions = np.arange(len(stages))

for i, cl in enumerate(clusters):
    bar_heights = []
    for stg in stages:
        bar_heights.append(stage_cluster_percentages[stg].get(cl, 0.0))
    plt.bar(x_positions + i*width - 0.4 + width/2, bar_heights, width,
            label=f"Cluster {cl}", color=f"C{cl}")

plt.xticks(x_positions, stages)
plt.xlabel("Project Stage")
plt.ylabel("Percentage of Time Series")
plt.title("Cluster Distribution by Stage")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("project_patterns_analysis_core_churn.png", dpi=300)
plt.show()

# (Optional) Plot centroids + medoids separately
plt.figure(figsize=(12, 8))
for (c_id, centroid_scaled, centroid_orig) in centroids:
    plt.plot(centroid_orig, linewidth=3, label=f"Cluster {c_id} centroid", color=f"C{c_id}")
for (c_id, g_idx, med_scaled, med_orig) in medoids_list:
    plt.plot(med_orig, '--', linewidth=2, label=f"Cluster {c_id} medoid", color=f"C{c_id}")
plt.title("Centroids & Medoids (Original Scale) - Core Churn Ratio")
plt.xlabel("Time (normalized)")
plt.ylabel("Core Churn Ratio")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("cluster_centroids_medoids_core_churn.png", dpi=300)
plt.show()

# -------------------------------------------------------------------------
# 12. Merge category, stage, cluster info
# -------------------------------------------------------------------------
df_clusters = df[['project', 'category']].drop_duplicates()

project_stage_clusters = pd.DataFrame({
    'project': [ts['project'] for ts in interpolated_time_series],
    'stage':   [ts['stage'] for ts in interpolated_time_series],
    'cluster': [ts['cluster'] for ts in interpolated_time_series]
}).drop_duplicates()

df_clusters = df_clusters.merge(project_stage_clusters, on='project', how='inner')

# Summaries
stage_cluster_counts_df = df_clusters.groupby(['stage', 'cluster']).size().unstack(fill_value=0)
stage_cluster_category_counts = df_clusters.groupby(['stage', 'cluster', 'category']).size().unstack(fill_value=0)

# Save
stage_cluster_counts_df.to_csv("stage_cluster_distribution.csv")
stage_cluster_category_counts.to_csv("stage_cluster_category_distribution.csv")

if use_forced_k:
    print(f"\nForced k = {forced_k}")
else:
    print(f"\nSilhouette-based best k = {best_k} (score={best_score:.3f})")

print("\nStage & cluster raw counts:")
for stg, c_dict in stage_cluster_counts.items():
    print(f"  Stage={stg}: {c_dict}")

print("\nStage & cluster percentages:")
for stg, pct_dict in stage_cluster_percentages.items():
    print(f"  Stage={stg}: {pct_dict}")

print("\nNumber of project-stage in each cluster:")
print(stage_cluster_counts_df)

print("\nNumber of OSS / OSS4SG by stage-cluster:")
print(stage_cluster_category_counts)

print("\nDONE.")