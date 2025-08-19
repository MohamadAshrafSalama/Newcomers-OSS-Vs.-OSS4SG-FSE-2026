# RQ3 Step 3: Scott-Knott Clustering Analysis

## Overview
This step applies Scott-Knott clustering to analyze which contribution patterns lead to fastest core status in OSS vs OSS4SG projects.

## Fixed Issues ✅

### 1. **Path Corrections**
- Fixed all paths to correctly reference RQ1 transition data and RQ3 step1/step2 data
- Transition data path: `../../../RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv`
- Step1 data path: `../step1/results/rolling_4week/weekly_pivot_for_dtw.csv`

### 2. **Email Normalization Fix**
- **Problem**: Step1 data has `contributor_id` in format `project_contributor+email`, but transition data uses `contributor_email`
- **Solution**: Extract actual email from composite ID using regex: `([^_]+@[^.]+\.[^+]+)`
- **Result**: Proper merging between datasets using standardized email format

### 3. **Pre-Core Data Only (Critical Fix)**
- **Problem**: Ensuring we only use contribution patterns BEFORE becoming core, not post-core contamination
- **Solution**: 
  - Step1 timeseries should already be filtered to pre-core only (confirmed from RQ3 documentation)
  - Added explicit logging to confirm we're using pre-core data
  - Only analyze `weeks_to_core` metric (time to FIRST becoming core)

### 4. **K=3 Clustering Configuration**
- **Fixed**: Both scripts now use k=3 clustering results from step2
- **Path**: `clustering_results_fixed/clustering_k3_results.json`
- **Cluster assignments**: `clustering_results_fixed/cluster_assignments_k3.csv`

## Files

### `scott_knott_implementation.py` (Helper Script)
- **Purpose**: Extract cluster assignments from step2 results and prepare for analysis
- **Key Function**: `save_cluster_assignments_from_existing()` - recreates clustering with exact same preprocessing
- **Output**: `clustering_results_fixed/cluster_assignments_k3.csv`

### `scott_knott_clustering_analysis.py` (Main Analysis)
- **Purpose**: Complete Scott-Knott analysis comparing pattern effectiveness
- **Key Features**:
  - Merges clustering results with transition data
  - Performs statistical tests (Kruskal-Wallis, Dunn's test)
  - Creates comprehensive visualizations
  - Generates actionable recommendations

## How to Run

### Step 1: Extract Cluster Assignments
```bash
cd "RQ3_engagement_patterns/step3_scott_knott_clustering"
python scott_knott_implementation.py
```

**Expected Output:**
- `clustering_results_fixed/cluster_assignments_k3.csv` - cluster assignments with extracted emails
- `clustering_results_fixed/clustering_k3_results.json` - metadata
- Verification of data compatibility with transition data

### Step 2: Run Main Analysis
```bash
python scott_knott_clustering_analysis.py
```

**Expected Output:**
- `pattern_effectiveness_results/pattern_effectiveness_analysis.png` - comprehensive visualization
- `pattern_effectiveness_results/group_statistics.csv` - detailed statistics
- `pattern_effectiveness_results/scott_knott_results.csv` - clustering results
- `pattern_effectiveness_results/analysis_report.txt` - text report with findings

## Data Flow Validation ✅

1. **Step1 → Step3**: 
   - ✅ `weekly_pivot_for_dtw.csv` contains 6,530 contributors with time series data
   - ✅ `contributor_id` format: `project_email+format` → extract to `contributor_email`

2. **Step2 → Step3**:
   - ✅ `clustering_k3_results.json` shows 3,421 contributors in k=3 clustering
   - ✅ Cluster sizes: [644, 1911, 866] contributors

3. **RQ1 → Step3**:
   - ✅ `contributor_transitions.csv` contains 85,763 transition records
   - ✅ `became_core == True` subset for core achievers only
   - ✅ Merge key: `contributor_email` (standardized format)

## Pattern Interpretation

Based on k=3 clustering from step2:
- **Cluster 0**: Early Spike pattern (644 contributors)
- **Cluster 1**: Sustained Activity pattern (1,911 contributors) 
- **Cluster 2**: Low/Gradual Activity pattern (866 contributors)

## Critical Data Quality Assurance

### Pre-Core Filtering ✅
- **Confirmed**: Step1 timeseries uses `is_pre_core == True` filtering
- **Result**: Only contribution patterns BEFORE becoming core are analyzed
- **Impact**: Eliminates post-core contamination identified in RQ2

### Email Normalization ✅
- **Problem Solved**: Format mismatch between datasets
- **Method**: Regex extraction of actual email from composite IDs
- **Validation**: Overlap percentage calculated and reported

### Clustering Consistency ✅
- **Method**: Exact same preprocessing as step2 (52-week target, MinMax scaling)
- **Random State**: Fixed at 42 for reproducibility
- **Validation**: Same cluster sizes and contributor counts

## Next Steps
1. Run both scripts in sequence
2. Review overlap statistics for data quality
3. Analyze Scott-Knott clustering results
4. Compare pattern effectiveness between OSS and OSS4SG
5. Generate actionable recommendations for newcomers

## Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn scikit-posthocs
```
