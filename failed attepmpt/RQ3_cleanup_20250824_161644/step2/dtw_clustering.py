import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import gc
from datetime import datetime
import json
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.interpolate import interp1d


class ImprovedContributionClustering:
    """
    Clustering with improved interpolation that properly represents activity patterns.
    """
    
    def __init__(self, data_path, output_dir="clustering_results", verbose=True):
        """Initialize clustering analysis."""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        
    def log(self, message, level="INFO"):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.verbose:
            prefix = {
                "START": "🚀",
                "SUCCESS": "✅",
                "ERROR": "❌",
                "WARNING": "⚠️",
                "INFO": "ℹ️",
                "PROGRESS": "⏳",
                "SAVE": "💾"
            }.get(level, "")
            print(f"[{timestamp}] {prefix} {message}")
        
    def analyze_data(self):
        """Analyze dataset characteristics."""
        self.log("="*60, "INFO")
        self.log("ANALYZING DATASET", "START")
        self.log("="*60, "INFO")
        
        # Sample to understand structure
        df_sample = pd.read_csv(self.data_path, nrows=100)
        
        # Count time columns
        time_cols = [col for col in df_sample.columns if col.startswith('t_')]
        n_timepoints = len(time_cols)
        
        # Total rows
        total_rows = sum(1 for _ in open(self.data_path)) - 1
        
        # Analyze actual activity patterns
        activity_stats = []
        for _, row in df_sample.iterrows():
            series = row[time_cols].values
            series_numeric = pd.to_numeric(series, errors='coerce')
            
            # Find actual activity span (first non-zero to last non-zero)
            non_zero_indices = np.where(~np.isnan(series_numeric) & (series_numeric != 0))[0]
            
            if len(non_zero_indices) > 0:
                first_activity = non_zero_indices[0]
                last_activity = non_zero_indices[-1]
                activity_span = last_activity - first_activity + 1
                total_activity_weeks = len(non_zero_indices)
                
                activity_stats.append({
                    'total_weeks': len(series_numeric[~np.isnan(series_numeric)]),
                    'activity_span': activity_span,
                    'active_weeks': total_activity_weeks,
                    'first_activity_week': first_activity,
                    'last_activity_week': last_activity
                })
        
        if activity_stats:
            df_stats = pd.DataFrame(activity_stats)
            self.log(f"Dataset: {total_rows} contributors × {n_timepoints} max timepoints", "INFO")
            self.log(f"Activity patterns (sample):", "INFO")
            self.log(f"  Median activity span: {df_stats['activity_span'].median():.0f} weeks", "INFO")
            self.log(f"  Mean active weeks: {df_stats['active_weeks'].mean():.0f} weeks", "INFO")
            self.log(f"  Median first activity: week {df_stats['first_activity_week'].median():.0f}", "INFO")
            self.log(f"  Median last activity: week {df_stats['last_activity_week'].median():.0f}", "INFO")
        else:
            self.log(f"Dataset: {total_rows} contributors × {n_timepoints} max timepoints", "INFO")
        
        return total_rows, n_timepoints
    
    def load_and_preprocess(self, target_length=52, min_weeks=6, scaling='minmax'):
        """
        Load data with improved preprocessing that preserves activity patterns.
        """
        self.log(f"Loading and preprocessing data...", "START")
        self.log(f"  Target length: {target_length} weeks", "INFO")
        self.log(f"  Minimum activity: {min_weeks} weeks", "INFO")
        
        # Load full dataset
        df = pd.read_csv(self.data_path)
        
        # Extract metadata
        meta_cols = ['contributor_id', 'project', 'project_type']
        meta_cols = [col for col in meta_cols if col in df.columns]
        metadata = df[meta_cols].copy()
        
        # Extract time series
        time_cols = [col for col in df.columns if col.startswith('t_')]
        time_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        processed_series = []
        valid_indices = []
        
        # Track different preprocessing cases
        stats = {
            'total': len(df),
            'too_short': 0,
            'no_activity': 0,
            'compressed': 0,
            'stretched': 0,
            'preserved': 0
        }
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                self.log(f"  Processing {idx}/{len(df)}...", "PROGRESS")
            
            series = row[time_cols].values
            series_numeric = pd.to_numeric(series, errors='coerce')
            
            # Remove NaN
            valid_values = series_numeric[~np.isnan(series_numeric)]
            
            if len(valid_values) == 0:
                stats['no_activity'] += 1
                continue
            
            # Find actual activity period (remove leading and trailing zeros)
            non_zero_indices = np.where(valid_values != 0)[0]
            
            if len(non_zero_indices) < min_weeks:
                stats['too_short'] += 1
                continue
            
            # Extract the activity period (from first to last non-zero)
            first_activity = non_zero_indices[0]
            last_activity = non_zero_indices[-1]
            activity_period = valid_values[first_activity:last_activity+1]
            
            # Now interpolate this activity period to target_length
            if len(activity_period) == target_length:
                # Perfect length - no interpolation needed
                interpolated = activity_period
                stats['preserved'] += 1
                
            elif len(activity_period) > target_length:
                # Compress: Sample evenly from the activity period
                # This preserves the overall pattern while reducing resolution
                indices = np.linspace(0, len(activity_period)-1, target_length)
                interpolated = np.interp(indices, np.arange(len(activity_period)), activity_period)
                stats['compressed'] += 1
                
            else:  # len(activity_period) < target_length
                # Stretch: Use interpolation but be careful about distribution
                # We want to preserve the relative timing of activities
                
                if len(activity_period) >= 2:
                    # Create interpolation function
                    x_old = np.arange(len(activity_period))
                    # Map to target length proportionally
                    x_new = np.linspace(0, len(activity_period)-1, target_length)
                    
                    # Use nearest-neighbor for very sparse data to preserve spikes
                    if len(non_zero_indices) < 10:
                        # For sparse data, use nearest to preserve activity spikes
                        f = interp1d(x_old, activity_period, kind='nearest', fill_value=0, bounds_error=False)
                    else:
                        # For denser data, linear is fine
                        f = interp1d(x_old, activity_period, kind='linear', fill_value=0, bounds_error=False)
                    
                    interpolated = f(x_new)
                    stats['stretched'] += 1
                else:
                    # Single point - can't interpolate
                    stats['too_short'] += 1
                    continue
            
            # Ensure no negative values from interpolation
            interpolated = np.maximum(interpolated, 0)
            
            processed_series.append(interpolated)
            valid_indices.append(idx)
        
        # Convert to array
        X = np.array(processed_series)
        
        # Apply scaling
        if scaling == 'minmax':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X.T).T
        elif scaling == 'standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X.T).T
        
        # Filter metadata
        metadata_filtered = metadata.iloc[valid_indices].reset_index(drop=True)
        
        # Report statistics
        self.log(f"\nPreprocessing complete:", "SUCCESS")
        self.log(f"  Total contributors: {stats['total']}", "INFO")
        self.log(f"  Valid series: {len(X)}", "INFO")
        self.log(f"  Filtered out:", "INFO")
        self.log(f"    Too short (<{min_weeks} active weeks): {stats['too_short']}", "INFO")
        self.log(f"    No activity: {stats['no_activity']}", "INFO")
        self.log(f"  Processing types:", "INFO")
        self.log(f"    Compressed (>{target_length} weeks): {stats['compressed']}", "INFO")
        self.log(f"    Stretched (<{target_length} weeks): {stats['stretched']}", "INFO")
        self.log(f"    Preserved (={target_length} weeks): {stats['preserved']}", "INFO")
        
        return X, metadata_filtered
    
    def cluster_with_k(self, X, k):
        """
        Perform clustering with a specific k value.
        """
        self.log(f"\nClustering with k={k}...", "START")
        
        # Use KMeans
        kmeans = KMeans(
                n_clusters=k,
                random_state=42,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        if len(set(labels)) > 1:
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = -1
        
        self.log(f"  Silhouette score: {sil_score:.4f}", "INFO")
        
        # Calculate cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        for i, count in enumerate(counts):
            pct = 100 * count / len(X)
            self.log(f"  Cluster {i}: {count} contributors ({pct:.1f}%)", "INFO")
        
        return {
            'k': k,
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'silhouette': sil_score,
            'model': kmeans
        }
    
    def plot_clustering_results(self, X, result, metadata):
        """
        Create comprehensive visualization for clustering results.
        """
        k = result['k']
        labels = result['labels']
        centroids = result['centroids']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Define colors for clusters
        colors = plt.cm.Set1(np.linspace(0, 1, k))
        
        # 1. Centroids only (large plot)
        ax1 = plt.subplot(2, 3, 1)
        for i, centroid in enumerate(centroids):
            ax1.plot(centroid, linewidth=3, label=f'Cluster {i}', color=colors[i], alpha=0.9)
        ax1.set_title(f'Cluster Centroids (k={k})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Period (weeks)', fontsize=12)
        ax1.set_ylabel('Normalized Contribution Index', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Sample time series with centroids
        ax2 = plt.subplot(2, 3, 2)
        n_samples_per_cluster = min(30, len(X) // k)
        
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_series = X[cluster_mask]
            
            if len(cluster_series) > 0:
                # Random sample from cluster
                sample_size = min(n_samples_per_cluster, len(cluster_series))
                sample_indices = np.random.choice(len(cluster_series), sample_size, replace=False)
                
                # Plot samples with low alpha
                for idx in sample_indices:
                    ax2.plot(cluster_series[idx], alpha=0.1, color=colors[cluster_id])
                
                # Plot centroid on top
                ax2.plot(centroids[cluster_id], linewidth=3, color=colors[cluster_id], 
                        alpha=0.9, label=f'Cluster {cluster_id}')
        
        ax2.set_title(f'Sample Series with Centroids', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Period (weeks)', fontsize=12)
        ax2.set_ylabel('Normalized Contribution Index', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cluster sizes
        ax3 = plt.subplot(2, 3, 3)
        unique_labels, counts = np.unique(labels, return_counts=True)
        bars = ax3.bar(unique_labels, counts, color=colors[:len(unique_labels)])
        ax3.set_title(f'Cluster Sizes', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Cluster', fontsize=12)
        ax3.set_ylabel('Number of Contributors', fontsize=12)
        ax3.set_xticks(unique_labels)
        
        # Add count and percentage labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            pct = 100 * count / len(X)
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
        
        # 4. Cluster composition by project type
        ax4 = plt.subplot(2, 3, 4)
        if 'project_type' in metadata.columns:
            # Create composition matrix
            composition = pd.crosstab(labels, metadata['project_type'], normalize='index') * 100
            
            # Plot stacked bar
            composition.plot(kind='bar', stacked=True, ax=ax4, color=['#ff9999', '#66b3ff'])
            ax4.set_title('Cluster Composition by Project Type', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Cluster', fontsize=12)
            ax4.set_ylabel('Percentage', fontsize=12)
            ax4.legend(title='Project Type', fontsize=10)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        
        # 5. Centroid characteristics
        ax5 = plt.subplot(2, 3, 5)
        characteristics = []
        
        for i, centroid in enumerate(centroids):
            # Calculate characteristics
            peak_week = np.argmax(centroid)
            peak_value = np.max(centroid)
            total_activity = np.sum(centroid)
            
            # Find activity duration (first and last significant activity)
            threshold = 0.1 * peak_value
            active_weeks = np.where(centroid > threshold)[0]
            if len(active_weeks) > 0:
                duration = active_weeks[-1] - active_weeks[0] + 1
            else:
                duration = 0
            
            characteristics.append({
                'Cluster': i,
                'Peak Week': peak_week,
                'Peak Value': peak_value,
                'Duration': duration,
                'Total': total_activity
            })
        
        char_df = pd.DataFrame(characteristics)
        
        # Plot characteristics as text
        text = "Cluster Characteristics:\n" + "="*30 + "\n"
        for _, row in char_df.iterrows():
            text += f"\nCluster {int(row['Cluster'])}:\n"
            text += f"  Peak at week {int(row['Peak Week'])}\n"
            text += f"  Peak value: {row['Peak Value']:.3f}\n"
            text += f"  Active duration: {int(row['Duration'])} weeks\n"
            text += f"  Total activity: {row['Total']:.2f}\n"
        
        ax5.text(0.1, 0.9, text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Centroid Analysis', fontsize=14, fontweight='bold')
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        summary_text = f"""
Clustering Summary (k={k})
{'='*30}

Total Contributors: {len(X):,}
Silhouette Score: {result['silhouette']:.4f}

Cluster Distribution:
"""
        for i, count in enumerate(counts):
            pct = 100 * count / len(X)
            summary_text += f"  Cluster {i}: {count:,} ({pct:.1f}%)\n"
        
        if 'project_type' in metadata.columns:
            summary_text += f"\nProject Type Distribution:\n"
            type_counts = metadata['project_type'].value_counts()
            for ptype, count in type_counts.items():
                pct = 100 * count / len(metadata)
                summary_text += f"  {ptype}: {count:,} ({pct:.1f}%)\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold')
        
        # Overall title
        plt.suptitle(f'Contribution Pattern Clustering Analysis (k={k})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = self.output_dir / f'clustering_k{k}_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"Plot saved: {filename}", "SAVE")
        
        return fig
    
    def analyze_patterns(self, X, results):
        """
        Analyze and interpret the clustering patterns found.
        """
        self.log("\n" + "="*60, "INFO")
        self.log("PATTERN INTERPRETATION", "START")
        self.log("="*60, "INFO")
        
        for result in results:
            k = result['k']
            centroids = result['centroids']
            labels = result['labels']
            
            self.log(f"\n📊 Patterns for k={k}:", "INFO")
            self.log("-"*40, "INFO")
            
            pattern_names = {
                'early_burst': 'Early Burst Contributors',
                'sustained': 'Sustained Contributors',
                'late_bloomer': 'Late Bloomers',
                'intermittent': 'Intermittent Contributors',
                'declining': 'Declining Contributors',
                'growing': 'Growing Contributors'
            }
            
            for i, centroid in enumerate(centroids):
                # Analyze pattern type
                peak_week = np.argmax(centroid)
                peak_value = np.max(centroid)
                
                # Calculate trend (first half vs second half)
                mid = len(centroid) // 2
                first_half_avg = np.mean(centroid[:mid])
                second_half_avg = np.mean(centroid[mid:])
                
                # Determine pattern type
                if peak_week < 10:
                    pattern = 'early_burst'
                elif peak_week > 35:
                    pattern = 'late_bloomer'
                elif second_half_avg > first_half_avg * 1.5:
                    pattern = 'growing'
                elif first_half_avg > second_half_avg * 1.5:
                    pattern = 'declining'
                elif np.std(centroid) < 0.1:
                    pattern = 'sustained'
                else:
                    pattern = 'intermittent'
                
                cluster_size = np.sum(labels == i)
                pct = 100 * cluster_size / len(labels)
                
                self.log(f"\n  Cluster {i}: {pattern_names.get(pattern, 'Unknown Pattern')}", "INFO")
                self.log(f"    Size: {cluster_size} contributors ({pct:.1f}%)", "INFO")
                self.log(f"    Peak activity: Week {peak_week}", "INFO")
                self.log(f"    Trend: {'Growing' if second_half_avg > first_half_avg else 'Declining'}", "INFO")
    
    def run_analysis(self, k_values=[2, 3, 4]):
        """
        Run the complete clustering analysis for specified k values.
        """
        self.log("="*70, "INFO")
        self.log("STARTING CLUSTERING ANALYSIS", "START")
        self.log("="*70, "INFO")
        self.log(f"Testing k values: {k_values}", "INFO")
        
        # Analyze data characteristics
        self.analyze_data()
        
        # Load and preprocess data
        X, metadata = self.load_and_preprocess(
            target_length=52,  # 1 year
            min_weeks=6,       # Minimum 6 weeks of activity
            scaling='minmax'
        )
        
        # Run clustering for each k
        results = []
        for k in k_values:
            result = self.cluster_with_k(X, k)
            results.append(result)
            
            # Create visualization
            self.plot_clustering_results(X, result, metadata)
            
            # Save results
            result_file = self.output_dir / f'clustering_k{k}_results.json'
            with open(result_file, 'w') as f:
                json.dump({
                    'k': k,
                    'silhouette': float(result['silhouette']),
                    'cluster_sizes': [int(c) for c in np.bincount(result['labels'])],
                    'n_contributors': len(X)
                }, f, indent=2)
            self.log(f"Results saved: {result_file}", "SAVE")
        
        # Analyze patterns
        self.analyze_patterns(X, results)
        
        # Find best k
        best_result = max(results, key=lambda x: x['silhouette'])
        self.log(f"\n🏆 Best k={best_result['k']} with silhouette={best_result['silhouette']:.4f}", "SUCCESS")
        
        return results, X, metadata


def main():
    """Main function to run clustering analysis."""
    
    # Configuration
    data_path = "../step1/results/rolling_4week/weekly_pivot_for_dtw.csv"
    output_dir = "clustering_results_fixed/"
    
    # Initialize clustering
    clustering = ImprovedContributionClustering(
        data_path=data_path,
        output_dir=output_dir,
        verbose=True
    )
    
    # Run analysis for k=5, 6 (k=2,3,4 already completed)
    results, X, metadata = clustering.run_analysis(k_values=[2, 3, 4, 5, 6])
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    # Summary
    print("\n📊 SUMMARY:")
    for result in results:
        print(f"  k={result['k']}: silhouette={result['silhouette']:.4f}")
    
    print(f"\nDataset size: {len(X)} contributors")
    print("Visualizations created for each k value")


if __name__ == "__main__":
    main()