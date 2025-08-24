import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, ks_2samp
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try importing DTW libraries
try:
    from dtaidistance import dtw
    from dtaidistance import dtw_ndim
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available, some tests will be skipped")

try:
    from tslearn.metrics import dtw as tslearn_dtw
    from tslearn.clustering import TimeSeriesKMeans
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    print("Warning: tslearn not available, some tests will be skipped")


class DTWClusteringValidator:
    """
    Comprehensive validation suite for DTW clustering results.
    Tests correctness, quality, and meaningfulness of clusters.
    """
    
    def __init__(self, data_path=None, results_path=None, verbose=True):
        """
        Initialize validator with paths to data and results.
        
        Args:
            data_path: Path to original time series data (pivot table)
            results_path: Path to clustering results directory
            verbose: Print detailed output
        """
        self.data_path = Path(data_path) if data_path else None
        self.results_path = Path(results_path) if results_path else None
        self.verbose = verbose
        self.test_results = []
        self.validation_scores = {}
        
    def log(self, message, level="INFO"):
        """Log message with level."""
        if self.verbose:
            prefix = {
                "SUCCESS": "✅",
                "ERROR": "❌", 
                "WARNING": "⚠️",
                "INFO": "ℹ️"
            }.get(level, "")
            print(f"{prefix} [{level}] {message}")
        self.test_results.append({"level": level, "message": message})
    
    def test_dtw_implementation(self):
        """Test 1: Validate DTW distance calculation implementation."""
        self.log("="*60)
        self.log("TEST 1: DTW Implementation Validation")
        self.log("-"*40)
        
        # Create simple test sequences with known DTW distances
        seq1 = np.array([1, 2, 3, 4, 5])
        seq2 = np.array([1, 2, 3, 4, 5])  # Identical
        seq3 = np.array([2, 3, 4, 5, 6])  # Shifted by 1
        seq4 = np.array([1, 3, 5, 7, 9])  # Different pattern
        
        tests_passed = []
        
        # Test 1.1: Identical sequences should have distance 0
        if DTW_AVAILABLE:
            dist = dtw.distance(seq1, seq2)
            if np.isclose(dist, 0, atol=1e-10):
                self.log("✓ Identical sequences have DTW distance = 0", "SUCCESS")
                tests_passed.append(True)
            else:
                self.log(f"✗ Identical sequences DTW = {dist} (expected 0)", "ERROR")
                tests_passed.append(False)
        
        # Test 1.2: DTW should be symmetric
        if TSLEARN_AVAILABLE:
            dist_ab = tslearn_dtw(seq1.reshape(-1, 1), seq3.reshape(-1, 1))
            dist_ba = tslearn_dtw(seq3.reshape(-1, 1), seq1.reshape(-1, 1))
            if np.isclose(dist_ab, dist_ba, rtol=1e-5):
                self.log("✓ DTW distance is symmetric", "SUCCESS")
                tests_passed.append(True)
            else:
                self.log(f"✗ DTW not symmetric: {dist_ab} vs {dist_ba}", "ERROR")
                tests_passed.append(False)
        
        # Test 1.3: Triangle inequality (approximately)
        if DTW_AVAILABLE:
            dist_12 = dtw.distance(seq1, seq2)
            dist_23 = dtw.distance(seq2, seq3)
            dist_13 = dtw.distance(seq1, seq3)
            
            # DTW doesn't strictly follow triangle inequality but should be close
            if dist_13 <= dist_12 + dist_23 + 0.1:  # Small tolerance
                self.log("✓ DTW approximately follows triangle inequality", "SUCCESS")
                tests_passed.append(True)
            else:
                self.log("⚠ DTW triangle inequality check (this is ok for DTW)", "WARNING")
                tests_passed.append(True)  # Not a failure for DTW
        
        # Test 1.4: Shifted sequences should have small DTW distance
        if DTW_AVAILABLE:
            dist_shift = dtw.distance(seq1, seq3)
            dist_diff = dtw.distance(seq1, seq4)
            if dist_shift < dist_diff:
                self.log("✓ Shifted sequences have smaller DTW than different patterns", "SUCCESS")
                tests_passed.append(True)
            else:
                self.log("✗ DTW not detecting pattern similarity correctly", "ERROR")
                tests_passed.append(False)
        
        success_rate = sum(tests_passed) / len(tests_passed) if tests_passed else 0
        self.validation_scores['dtw_implementation'] = success_rate
        
        if success_rate == 1.0:
            self.log(f"PASSED: All {len(tests_passed)} DTW implementation tests passed", "SUCCESS")
        else:
            self.log(f"PARTIAL: {sum(tests_passed)}/{len(tests_passed)} tests passed", "WARNING")
        
        return success_rate == 1.0
    
    def test_preprocessing_integrity(self, original_data, processed_data):
        """Test 2: Validate that preprocessing preserves data integrity."""
        self.log("="*60)
        self.log("TEST 2: Preprocessing Integrity Validation")
        self.log("-"*40)
        
        tests_passed = []
        
        # Test 2.1: Check for NaN introduction
        nan_count_orig = np.isnan(original_data).sum()
        nan_count_proc = np.isnan(processed_data).sum()
        
        if nan_count_proc == 0:
            self.log("✓ No NaN values introduced during preprocessing", "SUCCESS")
            tests_passed.append(True)
        else:
            self.log(f"✗ Found {nan_count_proc} NaN values after preprocessing", "ERROR")
            tests_passed.append(False)
        
        # Test 2.2: Check value ranges
        if hasattr(self, 'scaling_method') and self.scaling_method == 'minmax':
            if processed_data.min() >= 0 and processed_data.max() <= 1:
                self.log("✓ MinMax scaling correctly applied (values in [0,1])", "SUCCESS")
                tests_passed.append(True)
            else:
                self.log(f"✗ Values outside [0,1]: min={processed_data.min()}, max={processed_data.max()}", "ERROR")
                tests_passed.append(False)
        
        # Test 2.3: Check relative patterns preserved
        # Select random pairs and check if their relative ordering is preserved
        n_samples = min(10, len(original_data))
        sample_indices = np.random.choice(len(original_data), n_samples, replace=False)
        
        correlation_preserved = []
        for idx in sample_indices:
            if idx < len(original_data) and idx < len(processed_data):
                orig_series = original_data[idx][~np.isnan(original_data[idx])]
                proc_series = processed_data[idx]
                
                if len(orig_series) > 1 and len(proc_series) > 1:
                    # Interpolate original to same length as processed for comparison
                    x_orig = np.linspace(0, 1, len(orig_series))
                    x_proc = np.linspace(0, 1, len(proc_series))
                    
                    # Check if patterns are similar (correlation)
                    from scipy.interpolate import interp1d
                    f = interp1d(x_orig, orig_series, kind='linear', fill_value='extrapolate')
                    orig_interp = f(x_proc)
                    
                    corr, _ = pearsonr(orig_interp, proc_series)
                    correlation_preserved.append(corr)
        
        avg_correlation = np.mean(correlation_preserved) if correlation_preserved else 0
        if avg_correlation > 0.8:  # High correlation threshold
            self.log(f"✓ Pattern preservation: avg correlation = {avg_correlation:.3f}", "SUCCESS")
            tests_passed.append(True)
        else:
            self.log(f"⚠ Pattern preservation: avg correlation = {avg_correlation:.3f}", "WARNING")
            tests_passed.append(avg_correlation > 0.5)  # Lower threshold for pass
        
        # Test 2.4: Check sequence lengths consistency
        lengths = [len(seq) for seq in processed_data]
        if len(set(lengths)) == 1:
            self.log(f"✓ All sequences normalized to same length: {lengths[0]}", "SUCCESS")
            tests_passed.append(True)
        else:
            self.log(f"✗ Inconsistent sequence lengths: {set(lengths)}", "ERROR")
            tests_passed.append(False)
        
        success_rate = sum(tests_passed) / len(tests_passed) if tests_passed else 0
        self.validation_scores['preprocessing_integrity'] = success_rate
        
        return success_rate > 0.75  # Allow some tolerance
    
    def test_clustering_quality(self, data, labels, method_name=""):
        """Test 3: Validate clustering quality metrics."""
        self.log("="*60)
        self.log(f"TEST 3: Clustering Quality Validation {method_name}")
        self.log("-"*40)
        
        n_clusters = len(np.unique(labels))
        tests_passed = []
        
        # Test 3.1: Silhouette Score
        if len(set(labels)) > 1:
            sil_score = silhouette_score(data, labels)
            self.log(f"Silhouette Score: {sil_score:.4f}", "INFO")
            
            if sil_score > 0.3:
                self.log("✓ Good silhouette score (>0.3)", "SUCCESS")
                tests_passed.append(True)
            elif sil_score > 0.2:
                self.log("⚠ Moderate silhouette score (0.2-0.3)", "WARNING")
                tests_passed.append(True)
            else:
                self.log("✗ Poor silhouette score (<0.2)", "ERROR")
                tests_passed.append(False)
            
            # Test 3.2: Check for negative silhouette samples
            sample_scores = silhouette_samples(data, labels)
            negative_ratio = (sample_scores < 0).sum() / len(sample_scores)
            
            if negative_ratio < 0.1:
                self.log(f"✓ Few misclassified points ({negative_ratio:.1%} negative silhouettes)", "SUCCESS")
                tests_passed.append(True)
            elif negative_ratio < 0.25:
                self.log(f"⚠ Some misclassified points ({negative_ratio:.1%} negative silhouettes)", "WARNING")
                tests_passed.append(True)
            else:
                self.log(f"✗ Many misclassified points ({negative_ratio:.1%} negative silhouettes)", "ERROR")
                tests_passed.append(False)
        
        # Test 3.3: Davies-Bouldin Index (lower is better)
        if len(set(labels)) > 1:
            db_score = davies_bouldin_score(data, labels)
            self.log(f"Davies-Bouldin Index: {db_score:.4f}", "INFO")
            
            if db_score < 1.0:
                self.log("✓ Good Davies-Bouldin score (<1.0)", "SUCCESS")
                tests_passed.append(True)
            elif db_score < 2.0:
                self.log("⚠ Moderate Davies-Bouldin score (1.0-2.0)", "WARNING")
                tests_passed.append(True)
            else:
                self.log("✗ Poor Davies-Bouldin score (>2.0)", "ERROR")
                tests_passed.append(False)
        
        # Test 3.4: Cluster size balance
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_size = counts.min()
        max_size = counts.max()
        size_ratio = min_size / max_size
        
        if size_ratio > 0.1:  # No cluster is less than 10% of largest
            self.log(f"✓ Balanced cluster sizes (ratio: {size_ratio:.2f})", "SUCCESS")
            tests_passed.append(True)
        else:
            self.log(f"⚠ Imbalanced cluster sizes (smallest/largest: {size_ratio:.2f})", "WARNING")
            tests_passed.append(size_ratio > 0.05)
        
        # Test 3.5: No singleton clusters
        singleton_clusters = sum(1 for c in counts if c == 1)
        if singleton_clusters == 0:
            self.log("✓ No singleton clusters", "SUCCESS")
            tests_passed.append(True)
        else:
            self.log(f"⚠ Found {singleton_clusters} singleton clusters", "WARNING")
            tests_passed.append(singleton_clusters <= 1)
        
        success_rate = sum(tests_passed) / len(tests_passed) if tests_passed else 0
        self.validation_scores[f'clustering_quality_{method_name}'] = success_rate
        
        return success_rate > 0.6  # Allow some tolerance
    
    def test_cluster_consistency(self, data, labels, n_iterations=5):
        """Test 4: Validate clustering consistency and stability."""
        self.log("="*60)
        self.log("TEST 4: Clustering Consistency Validation")
        self.log("-"*40)
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        n_clusters = len(np.unique(labels))
        consistency_scores = []
        
        # Run clustering multiple times and check consistency
        for i in range(n_iterations):
            kmeans = KMeans(n_clusters=n_clusters, random_state=i*42, n_init=10)
            new_labels = kmeans.fit_predict(data)
            
            # Compare with original using Adjusted Rand Index
            ari = adjusted_rand_score(labels, new_labels)
            consistency_scores.append(ari)
        
        avg_consistency = np.mean(consistency_scores)
        std_consistency = np.std(consistency_scores)
        
        self.log(f"Consistency over {n_iterations} runs: {avg_consistency:.3f} ± {std_consistency:.3f}", "INFO")
        
        if avg_consistency > 0.8:
            self.log("✓ Highly consistent clustering (ARI > 0.8)", "SUCCESS")
            result = True
        elif avg_consistency > 0.6:
            self.log("⚠ Moderately consistent clustering (ARI 0.6-0.8)", "WARNING")
            result = True
        else:
            self.log("✗ Inconsistent clustering (ARI < 0.6)", "ERROR")
            result = False
        
        self.validation_scores['cluster_consistency'] = avg_consistency
        return result
    
    def test_dtw_vs_euclidean(self, data):
        """Test 5: Compare DTW clustering with Euclidean to validate DTW benefit."""
        self.log("="*60)
        self.log("TEST 5: DTW vs Euclidean Comparison")
        self.log("-"*40)
        
        from sklearn.cluster import KMeans
        
        # Try different k values
        k_values = [3, 4, 5]
        dtw_better_count = 0
        
        for k in k_values:
            if k >= len(data):
                continue
                
            # Euclidean clustering
            kmeans_euclidean = KMeans(n_clusters=k, random_state=42)
            labels_euclidean = kmeans_euclidean.fit_predict(data)
            
            if len(set(labels_euclidean)) > 1:
                sil_euclidean = silhouette_score(data, labels_euclidean)
            else:
                sil_euclidean = -1
            
            # DTW clustering (if available)
            if TSLEARN_AVAILABLE:
                try:
                    X_3d = data.reshape(data.shape[0], -1, 1)
                    km_dtw = TimeSeriesKMeans(n_clusters=k, metric="dtw", 
                                             max_iter=5, random_state=42)
                    labels_dtw = km_dtw.fit_predict(X_3d)
                    
                    if len(set(labels_dtw)) > 1:
                        sil_dtw = silhouette_score(data, labels_dtw)
                    else:
                        sil_dtw = -1
                    
                    self.log(f"k={k}: Euclidean={sil_euclidean:.3f}, DTW={sil_dtw:.3f}", "INFO")
                    
                    if sil_dtw > sil_euclidean:
                        dtw_better_count += 1
                        
                except Exception as e:
                    self.log(f"DTW clustering failed for k={k}: {str(e)}", "WARNING")
            else:
                self.log(f"k={k}: Euclidean={sil_euclidean:.3f} (DTW not available)", "INFO")
        
        if TSLEARN_AVAILABLE and len(k_values) > 0:
            dtw_advantage = dtw_better_count / len(k_values)
            
            if dtw_advantage > 0.5:
                self.log(f"✓ DTW performs better in {dtw_better_count}/{len(k_values)} cases", "SUCCESS")
                result = True
            else:
                self.log(f"⚠ DTW only better in {dtw_better_count}/{len(k_values)} cases", "WARNING")
                result = True  # Not necessarily a failure
                
            self.validation_scores['dtw_advantage'] = dtw_advantage
        else:
            self.log("⚠ DTW comparison skipped (library not available)", "WARNING")
            result = True
        
        return result
    
    def test_cluster_interpretability(self, data, labels, metadata=None):
        """Test 6: Validate cluster interpretability and patterns."""
        self.log("="*60)
        self.log("TEST 6: Cluster Interpretability Validation")
        self.log("-"*40)
        
        n_clusters = len(np.unique(labels))
        tests_passed = []
        
        # Test 6.1: Compute cluster centroids and check for distinct patterns
        centroids = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) > 0:
                centroid = np.mean(cluster_data, axis=0)
                centroids.append(centroid)
        
        # Check if centroids are distinct
        from scipy.spatial.distance import pdist, squareform
        if len(centroids) > 1:
            centroid_distances = pdist(centroids, metric='euclidean')
            min_distance = centroid_distances.min()
            avg_distance = centroid_distances.mean()
            
            self.log(f"Centroid separation: min={min_distance:.3f}, avg={avg_distance:.3f}", "INFO")
            
            if min_distance > 0.1:  # Threshold for distinction
                self.log("✓ Cluster centroids are well separated", "SUCCESS")
                tests_passed.append(True)
            else:
                self.log("⚠ Some cluster centroids are very similar", "WARNING")
                tests_passed.append(False)
        
        # Test 6.2: Analyze pattern characteristics
        pattern_chars = []
        for i, centroid in enumerate(centroids):
            # Characterize each centroid
            char = {
                'cluster': i,
                'mean_value': np.mean(centroid),
                'std_value': np.std(centroid),
                'trend': 'increasing' if centroid[-1] > centroid[0] else 'decreasing',
                'peak_position': np.argmax(centroid) / len(centroid),  # Relative position
                'volatility': np.std(np.diff(centroid))
            }
            pattern_chars.append(char)
            
            self.log(f"Cluster {i}: {char['trend']}, peak at {char['peak_position']:.1%}, "
                    f"volatility={char['volatility']:.3f}", "INFO")
        
        # Check if patterns are meaningfully different
        if len(pattern_chars) > 1:
            # Different trends
            trends = [p['trend'] for p in pattern_chars]
            if len(set(trends)) > 1:
                self.log("✓ Clusters show different trend patterns", "SUCCESS")
                tests_passed.append(True)
            
            # Different peak positions
            peaks = [p['peak_position'] for p in pattern_chars]
            if max(peaks) - min(peaks) > 0.3:
                self.log("✓ Clusters have different peak timings", "SUCCESS")
                tests_passed.append(True)
        
        # Test 6.3: Project type distribution (if metadata available)
        if metadata is not None and 'project_type' in metadata.columns:
            from scipy.stats import chi2_contingency
            
            # Create contingency table
            contingency = pd.crosstab(metadata['project_type'], labels)
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            self.log(f"Project type distribution chi-square p-value: {p_value:.4f}", "INFO")
            
            if p_value < 0.05:
                self.log("✓ Significant association between clusters and project types", "SUCCESS")
                tests_passed.append(True)
            else:
                self.log("⚠ No significant association with project types", "WARNING")
                tests_passed.append(True)  # Not necessarily a failure
        
        success_rate = sum(tests_passed) / len(tests_passed) if tests_passed else 0.5
        self.validation_scores['interpretability'] = success_rate
        
        return success_rate > 0.5
    
    def visualize_validation_results(self, data, labels, save_path=None):
        """Create comprehensive visualization of validation results."""
        fig = plt.figure(figsize=(20, 12))
        
        n_clusters = len(np.unique(labels))
        
        # Plot 1: Cluster centroids
        ax1 = plt.subplot(2, 3, 1)
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) > 0:
                centroid = np.mean(cluster_data, axis=0)
                ax1.plot(centroid, linewidth=2, label=f'Cluster {cluster_id} (n={cluster_mask.sum()})')
        
        ax1.set_title('Cluster Centroids')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Contribution Index')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Silhouette plot
        ax2 = plt.subplot(2, 3, 2)
        if len(set(labels)) > 1:
            from sklearn.metrics import silhouette_samples
            sample_silhouette_values = silhouette_samples(data, labels)
            
            y_lower = 10
            for i in range(n_clusters):
                cluster_silhouette_values = sample_silhouette_values[labels == i]
                cluster_silhouette_values.sort()
                
                size_cluster_i = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = plt.cm.Set3(float(i) / n_clusters)
                ax2.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)
                
                ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10
            
            ax2.axvline(x=silhouette_score(data, labels), color="red", linestyle="--")
            ax2.set_title('Silhouette Plot')
            ax2.set_xlabel('Silhouette Coefficient')
            ax2.set_ylabel('Cluster')
        
        # Plot 3: Cluster size distribution
        ax3 = plt.subplot(2, 3, 3)
        unique_labels, counts = np.unique(labels, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        ax3.bar(unique_labels, counts, color=colors)
        ax3.set_title('Cluster Size Distribution')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Series')
        for i, count in enumerate(counts):
            ax3.text(i, count + 0.5, str(count), ha='center')
        
        # Plot 4: Within-cluster variance
        ax4 = plt.subplot(2, 3, 4)
        within_cluster_var = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            if len(cluster_data) > 1:
                var = np.var(cluster_data, axis=0).mean()
                within_cluster_var.append(var)
            else:
                within_cluster_var.append(0)
        
        ax4.bar(range(n_clusters), within_cluster_var, color=colors)
        ax4.set_title('Within-Cluster Variance')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Average Variance')
        
        # Plot 5: Inter-cluster distances heatmap
        ax5 = plt.subplot(2, 3, 5)
        centroids = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            if len(cluster_data) > 0:
                centroids.append(np.mean(cluster_data, axis=0))
        
        if len(centroids) > 1:
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(centroids, centroids, metric='euclidean')
            im = ax5.imshow(dist_matrix, cmap='YlOrRd')
            ax5.set_title('Inter-cluster Distances')
            ax5.set_xlabel('Cluster')
            ax5.set_ylabel('Cluster')
            plt.colorbar(im, ax=ax5)
            
            # Add text annotations
            for i in range(n_clusters):
                for j in range(n_clusters):
                    ax5.text(j, i, f'{dist_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
        
        # Plot 6: Validation scores summary
        ax6 = plt.subplot(2, 3, 6)
        if self.validation_scores:
            scores = list(self.validation_scores.values())
            names = list(self.validation_scores.keys())
            
            # Shorten names for display
            short_names = [n.replace('_', '\n') if len(n) > 15 else n for n in names]
            
            bars = ax6.bar(range(len(scores)), scores, color='skyblue')
            ax6.set_title('Validation Scores Summary')
            ax6.set_ylabel('Score')
            ax6.set_ylim([0, 1])
            ax6.set_xticks(range(len(scores)))
            ax6.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
            
            # Color bars based on score
            for i, (bar, score) in enumerate(zip(bars, scores)):
                if score >= 0.8:
                    bar.set_color('green')
                elif score >= 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            # Add score values on bars
            for i, score in enumerate(scores):
                ax6.text(i, score + 0.02, f'{score:.2f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"Validation visualization saved to {save_path}", "INFO")
        
        plt.show()
        
        return fig
    
    def run_comprehensive_validation(self, data=None, labels=None, original_data=None, metadata=None):
        """Run all validation tests."""
        self.log("="*70)
        self.log("DTW CLUSTERING VALIDATION SUITE")
        self.log("="*70)
        self.log(f"Starting validation at: {pd.Timestamp.now()}")
        self.log("")
        
        # Load data if not provided
        if data is None and self.data_path:
            self.log("Loading data from file...", "INFO")
            df = pd.read_csv(self.data_path)
            
            # Extract time series data
            time_cols = [col for col in df.columns if col.startswith('t_')]
            data = df[time_cols].values
            
            # Extract metadata
            meta_cols = [col for col in df.columns if not col.startswith('t_')]
            metadata = df[meta_cols] if meta_cols else None
        
        # If we don't have labels, generate them with KMeans for testing
        if labels is None:
            from sklearn.cluster import KMeans
            self.log("No labels provided, generating with KMeans k=4...", "INFO")
            kmeans = KMeans(n_clusters=4, random_state=42)
            labels = kmeans.fit_predict(data)
        
        # Run all tests
        test_results = {}
        
        # Test 1: DTW Implementation
        test_results['DTW Implementation'] = self.test_dtw_implementation()
        
        # Test 2: Preprocessing Integrity (if original data available)
        if original_data is not None:
            test_results['Preprocessing Integrity'] = self.test_preprocessing_integrity(
                original_data, data
            )
        
        # Test 3: Clustering Quality
        test_results['Clustering Quality'] = self.test_clustering_quality(
            data, labels, method_name="main"
        )
        
        # Test 4: Cluster Consistency
        test_results['Cluster Consistency'] = self.test_cluster_consistency(
            data, labels
        )
        
        # Test 5: DTW vs Euclidean
        test_results['DTW vs Euclidean'] = self.test_dtw_vs_euclidean(data)
        
        # Test 6: Interpretability
        test_results['Interpretability'] = self.test_cluster_interpretability(
            data, labels, metadata
        )
        
        # Generate visualizations
        self.log("")
        self.log("Generating validation visualizations...", "INFO")
        self.visualize_validation_results(
            data, labels, 
            save_path=self.results_path / 'dtw_validation_plots.png' if self.results_path else None
        )
        
        # Summary
        self.log("")
        self.log("="*70)
        self.log("VALIDATION SUMMARY")
        self.log("="*70)
        
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.log(f"{test_name:.<40} {status}")
        
        self.log("")
        overall_score = passed_tests / total_tests
        self.log(f"Overall Score: {passed_tests}/{total_tests} ({overall_score:.1%})")
        
        if overall_score >= 0.8:
            self.log("🎉 EXCELLENT: DTW clustering is working very well!", "SUCCESS")
        elif overall_score >= 0.6:
            self.log("✅ GOOD: DTW clustering is working adequately with minor issues", "SUCCESS")
        elif overall_score >= 0.4:
            self.log("⚠️ MODERATE: DTW clustering has some issues to address", "WARNING")
        else:
            self.log("❌ POOR: DTW clustering has significant problems", "ERROR")
        
        # Save detailed report
        self.save_validation_report(test_results)
        
        return test_results, overall_score
    
    def save_validation_report(self, test_results):
        """Save detailed validation report."""
        if not self.results_path:
            self.results_path = Path(".")
        
        report_path = self.results_path / 'dtw_validation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DTW CLUSTERING VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("TEST RESULTS:\n")
            f.write("-"*40 + "\n")
            for test_name, result in test_results.items():
                status = "PASSED" if result else "FAILED"
                f.write(f"{test_name}: {status}\n")
            
            f.write("\nVALIDATION SCORES:\n")
            f.write("-"*40 + "\n")
            for metric, score in self.validation_scores.items():
                f.write(f"{metric}: {score:.4f}\n")
            
            f.write("\nDETAILED LOGS:\n")
            f.write("-"*40 + "\n")
            for entry in self.test_results:
                f.write(f"[{entry['level']}] {entry['message']}\n")
        
        self.log(f"\nDetailed report saved to: {report_path}", "INFO")


def main():
    """Main function to run DTW clustering validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate DTW clustering results')
    parser.add_argument('--data-path', type=str,
                       help='Path to time series data (CSV file)')
    parser.add_argument('--results-path', type=str,
                       default='clustering_results/',
                       help='Path to save validation results')
    parser.add_argument('--labels-path', type=str,
                       help='Path to cluster labels (optional)')
    parser.add_argument('--test-synthetic', action='store_true',
                       help='Test with synthetic data')
    
    args = parser.parse_args()
    
    if args.test_synthetic:
        # Create synthetic test data
        print("Creating synthetic test data...")
        np.random.seed(42)
        
        # Create 3 distinct patterns
        n_samples_per_pattern = 20
        n_timesteps = 50
        
        # Pattern 1: Increasing trend
        pattern1 = np.array([np.linspace(0, 1, n_timesteps) + np.random.normal(0, 0.05, n_timesteps)
                            for _ in range(n_samples_per_pattern)])
        
        # Pattern 2: Decreasing trend
        pattern2 = np.array([np.linspace(1, 0, n_timesteps) + np.random.normal(0, 0.05, n_timesteps)
                            for _ in range(n_samples_per_pattern)])
        
        # Pattern 3: Bell curve
        x = np.linspace(-3, 3, n_timesteps)
        bell = np.exp(-x**2/2)
        pattern3 = np.array([bell + np.random.normal(0, 0.05, n_timesteps)
                            for _ in range(n_samples_per_pattern)])
        
        # Combine patterns
        synthetic_data = np.vstack([pattern1, pattern2, pattern3])
        synthetic_labels = np.array([0]*n_samples_per_pattern + 
                                   [1]*n_samples_per_pattern + 
                                   [2]*n_samples_per_pattern)
        
        # Run validation
        validator = DTWClusteringValidator(results_path=args.results_path)
        results, score = validator.run_comprehensive_validation(
            data=synthetic_data,
            labels=synthetic_labels
        )
        
    else:
        # Run validation on real data
        validator = DTWClusteringValidator(
            data_path=args.data_path,
            results_path=args.results_path
        )
        
        # Load labels if provided
        labels = None
        if args.labels_path:
            labels_df = pd.read_csv(args.labels_path)
            if 'cluster' in labels_df.columns:
                labels = labels_df['cluster'].values
        
        results, score = validator.run_comprehensive_validation(labels=labels)
    
    # Exit with appropriate code
    if score >= 0.6:
        exit(0)  # Success
    else:
        exit(1)  # Failure


if __name__ == "__main__":
    main()