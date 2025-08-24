"""
Helper script to extract and save cluster assignments from your clustering results.
Run this AFTER your clustering script to prepare data for the pattern effectiveness analysis.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def save_cluster_assignments_from_existing():
    """
    Extract cluster assignments from your existing clustering results.
    This recreates the exact clustering from step2 using the same preprocessing.
    """
    
    # Configuration - FIXED PATHS
    input_data_path = "../step1/results/rolling_4week/weekly_pivot_for_dtw.csv"
    clustering_output_dir = Path("clustering_results_fixed")
    k = 3  # Using k=3 as decided
    
    print(f"Loading data from {input_data_path}...")
    
    # Load the original data
    df = pd.read_csv(input_data_path)
    
    # Extract contributor identifier - FIXED
    # Based on the actual data structure: contributor_id contains the email
    contributor_col = 'contributor_id'  # This contains the email format needed for merging
    
    print(f"Using '{contributor_col}' as contributor identifier")
    
    # Need to extract the actual email from the composite ID
    # Format is: project_contributor+email_at_domain -> we need to convert back to email format
    # Example: EbookFoundation_free-programming-books_43023629+chastiefol_at_users.noreply.github.com
    df['contributor_email'] = df[contributor_col].str.extract(r'[^+]+\+(.+_at_.+)', expand=False)
    df['contributor_email'] = df['contributor_email'].str.replace('_at_', '@')
    print(f"Extracted {df['contributor_email'].notna().sum()} valid emails from {len(df)} contributor IDs")
    
    # Get time series columns (those starting with 't_')
    time_cols = [col for col in df.columns if col.startswith('t_')]
    time_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Prepare data for clustering (same as your clustering script)
    print("Preparing data for clustering...")
    
    # Extract time series and handle missing values
    X = df[time_cols].values
    X = np.nan_to_num(X, nan=0)  # Replace NaN with 0
    
    # Apply same preprocessing as your script
    target_length = 52
    min_weeks = 6
    
    processed_series = []
    valid_indices = []
    
    for idx, series in enumerate(X):
        # Remove leading/trailing zeros
        non_zero = np.where(series != 0)[0]
        
        if len(non_zero) < min_weeks:
            continue
            
        # Extract activity period
        first_activity = non_zero[0]
        last_activity = non_zero[-1]
        activity_period = series[first_activity:last_activity+1]
        
        # Interpolate to target length
        if len(activity_period) == target_length:
            interpolated = activity_period
        elif len(activity_period) > target_length:
            indices = np.linspace(0, len(activity_period)-1, target_length)
            interpolated = np.interp(indices, np.arange(len(activity_period)), activity_period)
        else:
            if len(activity_period) >= 2:
                x_new = np.linspace(0, len(activity_period)-1, target_length)
                interpolated = np.interp(x_new, np.arange(len(activity_period)), activity_period)
            else:
                continue
        
        processed_series.append(interpolated)
        valid_indices.append(idx)
    
    # Convert to array and scale
    X_processed = np.array(processed_series)
    # Use per-series scaling to align with finalized Step 2 settings
    X_scaled = []
    for series in X_processed:
        s_min, s_max = series.min(), series.max()
        if s_max > s_min:
            X_scaled.append((series - s_min) / (s_max - s_min))
        else:
            X_scaled.append(series * 0)
    X_scaled = np.array(X_scaled)
    
    print(f"Processed {len(X_scaled)} valid contributors")
    
    # Perform clustering
    print(f"Performing K-means clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    
    # Create dataframe with assignments - FIXED to use extracted emails
    cluster_assignments = pd.DataFrame({
        'contributor_id': df.iloc[valid_indices][contributor_col].values,
        'contributor_email': df.iloc[valid_indices]['contributor_email'].values,
        'cluster': labels
    })
    
    # Also add project information - FIXED column names
    if 'project' in df.columns:
        cluster_assignments['project'] = df.iloc[valid_indices]['project'].values
    if 'project_type' in df.columns:
        cluster_assignments['project_type'] = df.iloc[valid_indices]['project_type'].values
    
    # Save cluster assignments
    output_file = clustering_output_dir / f"cluster_assignments_k{k}.csv"
    clustering_output_dir.mkdir(exist_ok=True, parents=True)
    cluster_assignments.to_csv(output_file, index=False)
    
    print(f"✅ Saved cluster assignments to {output_file}")
    print(f"   Total contributors: {len(cluster_assignments)}")
    print(f"   Cluster distribution:")
    for cluster_id in range(k):
        count = sum(labels == cluster_id)
        pct = 100 * count / len(labels)
        print(f"     Cluster {cluster_id}: {count} ({pct:.1f}%)")
    
    # Also save clustering metadata
    metadata = {
        'k': k,
        'n_contributors': len(cluster_assignments),
        'cluster_sizes': [int(sum(labels == i)) for i in range(k)],
        'cluster_names': {
            0: "Early Spike",
            1: "Sustained Activity",
            2: "Low/Gradual Activity"
        }
    }
    
    metadata_file = clustering_output_dir / f"clustering_k{k}_results.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved clustering metadata to {metadata_file}")
    
    return cluster_assignments


def verify_data_compatibility():
    """
    Verify that cluster assignments can be merged with transition data.
    """
    print("\n" + "="*60)
    print("VERIFYING DATA COMPATIBILITY")
    print("="*60)
    
    # Load cluster assignments
    cluster_file = Path("clustering_results_fixed") / "cluster_assignments_k3.csv"
    if not cluster_file.exists():
        print("❌ Cluster assignments file not found. Run save_cluster_assignments_from_existing() first.")
        return False
    
    df_clusters = pd.read_csv(cluster_file)
    print(f"✅ Loaded cluster assignments: {len(df_clusters)} contributors")
    print(f"   Columns: {', '.join(df_clusters.columns)}")
    
    # Load transition data - FIXED PATH
    transition_file = Path("../../RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv")
    if not transition_file.exists():
        transition_file = Path("../../../RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv")
    
    if not transition_file.exists():
        print(f"❌ Transition data not found at expected path: {transition_file}")
        return False
    
    df_transitions = pd.read_csv(transition_file, nrows=5)  # Just check structure
    print(f"✅ Found transition data")
    print(f"   Columns: {', '.join(df_transitions.columns[:10])}...")  # Show first 10 columns
    
    # Check for common identifier - FIXED to use extracted email
    cluster_id_col = 'contributor_email'  # We created this column with extracted emails
    transition_id_col = 'contributor_email'  # This exists in transition data
    
    if cluster_id_col in df_clusters.columns and transition_id_col in df_transitions.columns:
        print(f"✅ Can merge on: clusters['{cluster_id_col}'] = transitions['{transition_id_col}']")
        
        # Check sample overlap
        df_transitions_full = pd.read_csv(transition_file)
        df_transitions_core = df_transitions_full[df_transitions_full['became_core'] == True]
        
        # Check overlap
        cluster_ids = set(df_clusters[cluster_id_col].values)
        transition_ids = set(df_transitions_core[transition_id_col].values)
        
        overlap = cluster_ids.intersection(transition_ids)
        print(f"\n📊 Data overlap:")
        print(f"   Contributors in clusters: {len(cluster_ids)}")
        print(f"   Contributors who became core: {len(transition_ids)}")
        print(f"   Overlap (for analysis): {len(overlap)}")
        print(f"   Overlap percentage: {100*len(overlap)/len(cluster_ids):.1f}%")
        
        if len(overlap) < 100:
            print("⚠️  Warning: Low overlap. Check if contributor IDs match format.")
            print("   Sample cluster IDs:", list(cluster_ids)[:3])
            print("   Sample transition IDs:", list(transition_ids)[:3])
        else:
            print("✅ Good overlap for analysis!")
        
        return True
    else:
        print("❌ No common identifier column found for merging")
        return False


def main():
    """Run the data preparation."""
    print("="*60)
    print("PREPARING DATA FOR PATTERN EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    # Step 1: Save cluster assignments
    print("\nStep 1: Extracting cluster assignments...")
    try:
        cluster_assignments = save_cluster_assignments_from_existing()
        print("✅ Cluster assignments saved successfully")
    except Exception as e:
        print(f"❌ Error saving cluster assignments: {e}")
        return
    
    # Step 2: Verify compatibility
    print("\nStep 2: Verifying data compatibility...")
    if verify_data_compatibility():
        print("\n" + "="*60)
        print("✅ DATA PREPARATION COMPLETE!")
        print("="*60)
        print("\nYou can now run the pattern effectiveness analysis script!")
        print("Next step: Run the main analysis using pattern_effectiveness_analysis.py")
    else:
        print("\n❌ Data compatibility issues found. Please check the file paths and column names.")


if __name__ == "__main__":
    main()