# RQ2: Newcomer Treatment Patterns Analysis

This directory contains the LaTeX source and supporting materials for the RQ2 section on newcomer treatment patterns analysis.

## Overview

RQ2 investigates treatment patterns that characterize successful newcomer-to-core transitions in OSS and OSS4SG projects. The analysis consists of two main components:

1. **Data Preparation for ML-Based Prediction**: Using commit data to predict core contributor status
2. **RQ2 Step 4: Milestone Analysis**: Analyzing key milestones in the transition journey
3. **RQ3: Engagement Patterns Analysis**: Time-series clustering of engagement trajectories

## File Structure

```
paper/
├── RQ2_treatment_metrics.tex              # Main LaTeX content
├── RQ2_README.md                         # This documentation
└── tables/
    ├── RQ1_ml_model_features_and_ranking.tex
    ├── ml_models_and_results.tex
    ├── RQ2_milestones_definitions.tex
    └── RQ2_milestone_results.tex
```

## Data Sources

The analysis draws from several datasets:

1. **Commit Data**: 3.5M commits from 366 projects used for ML prediction
2. **Contributor Classification**: 23,069 contributors (8,045 core, 14,994 non-core)
3. **Timeline Data**: 6,530 core contributors with weekly activity patterns
4. **Feature Data**: 22 behavioral features extracted from first 90 days

## Key Findings

### ML-Based Prediction
- **Best Models**: RandomForest and GradientBoosting (74.6% ROC AUC)
- **Key Predictor**: Code volume (lines_changed_90d) at 22.2% importance
- **Consistency Features**: More important than raw frequency
- **Temporal Patterns**: More predictive than monthly breakdowns

### Milestone Analysis
- **OSS4SG Advantage**: Higher achievement rates in most milestones
- **Quick Wins**: First Accepted milestone achieved immediately
- **Community Building**: Cross-boundary contributions show largest OSS4SG advantage (90% vs 46.5%)

### Engagement Patterns (RQ3)
- **Optimal Pattern**: Low/Gradual Activity (21 weeks to core)
- **Worst Pattern**: Early Spike (51-60 weeks to core)
- **OSS4SG Benefits**: More forgiving, better supports steady contributors

## LaTeX Tables

### Table 1: ML Features Ranking
- Features ranked by RandomForest importance
- Categories: Activity Volume, Temporal Patterns, Growth Trends, Consistency
- Top 3 features account for 40% of predictive power

### Table 2: ML Model Results
- Performance comparison across 5 models
- Metrics: ROC AUC, PR AUC, F1 Score, Precision
- RandomForest and GradientBoosting show best performance

### Table 3: Milestone Definitions
- 7 core milestones with definitions
- Measurement methodology for each milestone
- Focus on behavioral indicators of transition progress

### Table 4: Milestone Results
- Achievement rates by project type (OSS vs OSS4SG)
- Median time to achieve each milestone
- Statistical comparison between communities

## Usage Notes

1. **Include in Main Document**: Use `\input{RQ2_treatment_metrics.tex}` in your main LaTeX file
2. **Table References**: Tables are automatically labeled for cross-referencing
3. **Figure References**: Update figure paths to match your document structure
4. **Data Updates**: If you have updated datasets, modify the numerical results accordingly

## Dependencies

- LaTeX packages: booktabs, graphicx, hyperref, natbib
- Input files: All tables in the `tables/` subdirectory
- Figures: Ensure figure files exist in your `figures/` directory

## Customization

The LaTeX code is structured with:
- `\mohamed{}` comments indicating areas for your input
- `\input{}` commands for table inclusion
- `\cite{}` commands for academic references
- Figure placeholders for visualization inclusion

## Next Steps

1. Update numerical results with your latest analysis
2. Add academic citations where indicated
3. Include relevant figures from your analysis
4. Review and validate all statistical claims
5. Proofread for consistency with your research narrative
