# Step 2: Timeline Data Generation & Analysis

## Overview
This directory contains the complete pipeline for generating, validating, and analyzing timeline data from raw cache files. It represents the core data processing step for RQ2 analysis.

## 🎯 **What Was Accomplished**

### ✅ **Data Generation Pipeline**
- **6,530 timeline files** successfully generated from raw JSON cache data
- **100% project classification** achieved using RQ1 master commits dataset
- **1,944,698 total events** processed and structured
- **Zero data corruption** - all files validated successfully

### 🔍 **Comprehensive Validation**
- **Data integrity checks** for all timeline files
- **Project type mapping** from OSS/OSS4SG classification
- **Statistical outlier analysis** using IQR and Z-score methods
- **Robust error handling** throughout the pipeline

### 📊 **Statistical Analysis**
- **Mann-Whitney U tests** for OSS vs OSS4SG comparisons
- **Effect size calculations** using Cliff's delta
- **Outlier impact assessment** with before/after analysis
- **Publication-ready visualizations** and tables

## 📁 **Directory Structure**

```
step2_timelines/
├── README.md                           # This file
├── convert_caches_to_timelines.py      # Main data conversion script
├── validate_timeline_data_comprehensive.py  # Comprehensive validation script
├── from_cache_timelines/               # Generated timeline CSV files (6,530 files)
├── results/                            # Analysis outputs and visualizations
├── drafted results/                    # Archived previous results
└── drafts/                            # Development drafts
```

## 🚀 **Key Scripts**

### 1. **`convert_caches_to_timelines.py`**
- **Purpose**: Convert raw JSON cache files to structured timeline CSV files
- **Input**: Raw cache JSON files from RQ2 data collection
- **Output**: 6,530 timeline CSV files with project classification
- **Key Features**:
  - Loads project types from RQ1 master commits dataset
  - Maps contributor-project pairs to project types
  - Generates structured timeline data with all required columns
  - Handles edge cases and missing data gracefully

### 2. **`validate_timeline_data_comprehensive.py`**
- **Purpose**: Comprehensive validation and statistical analysis
- **Input**: Generated timeline CSV files
- **Output**: Complete analysis with visualizations and statistics
- **Key Features**:
  - Data integrity validation
  - Outlier detection (IQR + Z-score methods)
  - Statistical testing (Mann-Whitney U, effect sizes)
  - Publication-quality visualizations
  - Comprehensive reporting

## 📊 **Data Quality Metrics**

| Metric | Value | Status |
|--------|-------|---------|
| Total Files | 6,530 | ✅ Complete |
| Valid Files | 6,530 | ✅ 100% |
| Empty Files | 0 | ✅ None |
| Corrupted Files | 0 | ✅ None |
| Project Classification | 100% | ✅ Complete |
| Total Events | 1,944,698 | ✅ Processed |
| OSS Contributors | 4,667 | ✅ Classified |
| OSS4SG Contributors | 1,863 | ✅ Classified |

## 🔍 **Outlier Analysis Results**

| Metric | Outliers | Percentage | Impact |
|--------|----------|------------|---------|
| Total Events | 797 | 12.2% | High |
| Commits | 843 | 12.9% | High |
| Pull Requests | 1,054 | 16.1% | Medium |
| Issues | 1,198 | 18.3% | Medium |
| Timeline Weeks | 329 | 5.0% | Low |

**Overall Impact**: 32.3% of contributors identified as outliers, 67.7% retained in clean dataset.

## 📈 **Statistical Significance**

### **With Outliers (Original Data)**
- **Total Events**: OSS/OSS4SG ratio = 0.03 (p < 0.001 ***, **large effect**)
- **Commits**: OSS/OSS4SG ratio = 0.03 (p < 0.001 ***, **large effect**)
- **Timeline Weeks**: OSS/OSS4SG ratio = 0.19 (p < 0.001 ***, **medium effect**)

### **Without Outliers (Cleaned Data)**
- **Total Events**: OSS/OSS4SG ratio = 0.04 (p < 0.001 ***, **large effect**)
- **Commits**: OSS/OSS4SG ratio = 0.03 (p < 0.001 ***, **large effect**)
- **Timeline Weeks**: OSS/OSS4SG ratio = 0.04 (p < 0.001 ***, **large effect**)

## 🎨 **Generated Visualizations**

### **Box Plots**
- **File**: `interaction_comparison_boxplots.png/pdf`
- **Content**: OSS vs OSS4SG comparison for all metrics (with/without outliers)
- **Format**: 2×5 grid showing before/after outlier removal
- **Size**: 421.7 KB (PNG), 80.6 KB (PDF)

### **Violin Plots**
- **File**: `interaction_comparison_violinplots.png/pdf`
- **Content**: Distribution shapes and patterns
- **Format**: 2×5 grid with log-scale y-axis
- **Size**: 907.1 KB (PNG), 66.4 KB (PDF)

## 📋 **Generated Data Files**

### **Timeline Datasets**
- **`timeline_data_with_outliers.csv`** (929.3 KB) - Original dataset with all contributors
- **`timeline_data_without_outliers.csv`** (635.1 KB) - Cleaned dataset after outlier removal

### **Analysis Results**
- **`comprehensive_statistical_results.json`** (7.0 KB) - Complete statistical analysis
- **`statistical_comparison_with_without_outliers.csv`** (0.9 KB) - Comparison table
- **`outlier_analysis_detailed.json`** (70.0 KB) - Detailed outlier analysis
- **`outlier_analysis_summary.csv`** (0.3 KB) - Outlier summary statistics
- **`validation_comprehensive.json`** (0.3 KB) - Validation results

## 🔧 **Technical Implementation**

### **Dependencies**
- **Python**: 3.8+
- **Core Libraries**: pandas, numpy, matplotlib, seaborn, scipy
- **Environment**: Virtual environment (`.venv/`) with all required packages

### **Data Pipeline**
1. **Raw Cache Files** → JSON parsing and validation
2. **Project Classification** → Load from RQ1 master dataset
3. **Timeline Generation** → Structured CSV files with all metadata
4. **Data Validation** → Integrity checks and quality assessment
5. **Outlier Detection** → Statistical outlier identification
6. **Statistical Analysis** → Hypothesis testing and effect sizes
7. **Visualization** → Publication-quality plots and tables
8. **Reporting** → Comprehensive results and documentation

### **Performance**
- **Processing Speed**: ~343 files/second (19 seconds for 6,530 files)
- **Memory Usage**: Efficient chunked processing
- **Error Handling**: Robust error handling with detailed logging

## 🎯 **Key Insights**

1. **OSS4SG Superiority**: OSS4SG contributors show dramatically higher activity across all metrics
2. **Robust Results**: Statistical significance maintained even after outlier removal
3. **Data Quality**: 100% classification accuracy with comprehensive validation
4. **Effect Sizes**: Large effects for core metrics indicating substantial differences
5. **Outlier Impact**: Outliers significantly skew means but don't affect core conclusions

## 🚀 **Next Steps**

1. **Step 3 Integration**: Use cleaned timeline data for treatment metrics analysis
2. **Cross-Validation**: Compare results with other research teams
3. **Publication Preparation**: Finalize figures and tables for paper
4. **Methodology Documentation**: Document outlier detection and statistical methods

## 📚 **References**

- **RQ1 Master Dataset**: Source for project type classification
- **Statistical Methods**: Mann-Whitney U test, Cliff's delta effect size
- **Outlier Detection**: IQR method (1.5×), Z-score method (3σ threshold)
- **Visualization**: Matplotlib/Seaborn with publication-quality styling

## 🆘 **Troubleshooting**

### **Common Issues**
- **Missing Dependencies**: Ensure virtual environment is activated
- **Path Issues**: Verify all paths exist and are accessible
- **Memory Issues**: Scripts use efficient chunked processing

### **Support**
- Check main RQ2 README for overview
- Review script documentation and error messages
- Contact research team for technical issues

---
*Last Updated: August 16, 2024*
*Status: ✅ Complete - All objectives achieved*
*Next Phase: Step 3 - Treatment Metrics Analysis*
