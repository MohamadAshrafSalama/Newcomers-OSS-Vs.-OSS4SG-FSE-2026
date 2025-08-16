# RQ2: Newcomer Treatment Patterns Analysis

## Overview
This directory contains the comprehensive analysis of newcomer treatment patterns in OSS vs OSS4SG projects, focusing on how contributors are treated during their transition from newcomer to core contributor status.

## Directory Structure

### `step2_timelines/` - Timeline Data Generation & Analysis
- **Purpose**: Generate and validate timeline data from raw cache files
- **Key Components**:
  - `convert_caches_to_timelines.py`: Converts raw JSON cache files to structured timeline CSV files
  - `validate_timeline_data_comprehensive.py`: Comprehensive validation with outlier analysis
  - `results/`: Analysis outputs and visualizations
  - `drafted results/`: Previous analysis results (archived)

### `step3_treatment_metrics/` - Treatment Pattern Analysis
- **Purpose**: Analyze specific treatment patterns and metrics
- **Status**: In development

### `daraset/` - Dataset Storage
- **Purpose**: Store intermediate and final datasets
- **Status**: In development

## Key Achievements

### ✅ **Timeline Data Generation (Step 2)**
- **6,530 timeline files** successfully generated from raw cache data
- **100% project classification** achieved (4,667 OSS, 1,863 OSS4SG)
- **1,944,698 total events** processed and validated
- **Comprehensive outlier analysis** with statistical significance testing

### 📊 **Statistical Results**
- **OSS4SG contributors show dramatically higher activity** than OSS contributors
- **Large effect sizes** for total events and commits (p < 0.001)
- **Outlier removal maintains statistical significance** while improving data quality
- **67.7% data retention** after outlier cleaning (4,421 contributors)

### 🎨 **Visualizations Generated**
- **Box plots**: OSS vs OSS4SG comparison (with/without outliers)
- **Violin plots**: Distribution shapes and patterns
- **Statistical tables**: Comprehensive comparison metrics
- **Outlier analysis**: Detailed outlier detection and removal statistics

## Data Quality Metrics

| Metric | Value |
|--------|-------|
| Total Timeline Files | 6,530 |
| Valid Files | 6,530 (100%) |
| Empty Files | 0 |
| Corrupted Files | 0 |
| Total Events | 1,944,698 |
| OSS Contributors | 4,667 |
| OSS4SG Contributors | 1,863 |
| Unknown/Missing | 0 |

## Statistical Significance

### With Outliers
- **Total Events**: OSS/OSS4SG ratio = 0.03 (p < 0.001 ***, large effect)
- **Commits**: OSS/OSS4SG ratio = 0.03 (p < 0.001 ***, large effect)
- **Timeline Weeks**: OSS/OSS4SG ratio = 0.19 (p < 0.001 ***, medium effect)

### Without Outliers (Cleaned Data)
- **Total Events**: OSS/OSS4SG ratio = 0.04 (p < 0.001 ***, large effect)
- **Commits**: OSS/OSS4SG ratio = 0.03 (p < 0.001 ***, large effect)
- **Timeline Weeks**: OSS/OSS4SG ratio = 0.04 (p < 0.001 ***, large effect)

## Outlier Analysis

| Metric | Outlier Percentage | Impact |
|--------|-------------------|---------|
| Total Events | 12.2% | High |
| Commits | 12.9% | High |
| Pull Requests | 16.1% | Medium |
| Issues | 18.3% | Medium |
| Timeline Weeks | 5.0% | Low |

**Overall**: 32.3% of contributors identified as outliers, 67.7% retained in clean dataset.

## Key Insights

1. **OSS4SG Superiority**: OSS4SG contributors are significantly more active across all metrics
2. **Robust Results**: Statistical significance maintained even after outlier removal
3. **Data Quality**: 100% classification accuracy with comprehensive validation
4. **Effect Sizes**: Large effects for core metrics (events, commits) indicating substantial differences

## Files Generated

### Visualizations
- `interaction_comparison_boxplots.pdf/png` - Box plot comparisons
- `interaction_comparison_violinplots.pdf/png` - Violin plot distributions

### Data Files
- `timeline_data_with_outliers.csv` - Original dataset (929.3 KB)
- `timeline_data_without_outliers.csv` - Cleaned dataset (635.1 KB)

### Analysis Results
- `comprehensive_statistical_results.json` - Complete statistical analysis (7.0 KB)
- `statistical_comparison_with_without_outliers.csv` - Comparison table (0.9 KB)
- `outlier_analysis_detailed.json` - Detailed outlier analysis (70.0 KB)
- `outlier_analysis_summary.csv` - Outlier summary (0.3 KB)
- `validation_comprehensive.json` - Validation results (0.3 KB)

## Technical Implementation

### Dependencies
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scipy
- Virtual environment: `.venv/`

### Key Scripts
1. **`convert_caches_to_timelines.py`**: Data conversion pipeline
2. **`validate_timeline_data_comprehensive.py`**: Comprehensive analysis engine
3. **Supporting scripts**: Visualization and analysis utilities

### Data Pipeline
1. **Raw Cache Files** → JSON parsing
2. **Timeline Generation** → CSV files with project classification
3. **Validation** → Data integrity checks
4. **Outlier Analysis** → Statistical outlier detection
5. **Statistical Testing** → Mann-Whitney U tests with effect sizes
6. **Visualization** → Publication-quality plots

## Next Steps

1. **Step 3**: Complete treatment metrics analysis
2. **Integration**: Combine with RQ1 and RQ3 results
3. **Publication**: Prepare figures and tables for paper
4. **Validation**: Cross-check with other research teams

## Contact

For questions about this analysis, refer to the main project documentation or contact the research team.

---
*Last Updated: August 16, 2024*
*Analysis Status: Step 2 Complete, Step 3 In Progress*


