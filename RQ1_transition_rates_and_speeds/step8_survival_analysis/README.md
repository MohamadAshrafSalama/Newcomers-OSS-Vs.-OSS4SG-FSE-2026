# Survival Analysis: OSS vs OSS4SG Newcomer-to-Core Transitions

## Overview

This survival analysis examines how newcomers transition to core contributors in OSS vs OSS4SG projects. We use survival analysis because it properly handles both contributors who became core and those who have not, capturing the complete picture.

## What is Survival Analysis?

In this context:
- **Survival** = Remaining a non-core contributor
- **Event** = Becoming a core contributor
- **Time** = Weeks from first commit to becoming core (or censoring)
- **Censored** = Contributors who have not become core by the observation end

## Directory Structure

```
step8_survival_analysis/
├── scripts/
│   ├── 1_prepare_survival_data.py
│   ├── 2_kaplan_meier_analysis.py
│   ├── 3_cox_regression.py
│   ├── 4_validate_results.py
│   └── run_all_analysis.py
├── data/
│   └── survival_data*.csv
├── results/
│   ├── kaplan_meier_results.json
│   ├── cox_regression_results.json
│   ├── validation_report.txt
│   └── FINAL_REPORT.txt
└── visualizations/
    ├── survival_curves_main.png
    ├── cumulative_incidence.png
    ├── hazard_ratios_forest.png
    └── validation_diagnostics.png
```

## Installation Requirements

```bash
"/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026/.venv/bin/pip" install pandas numpy matplotlib seaborn scipy lifelines scikit-learn
```

## Running the Analysis

### Option 1: Run Everything at Once

```bash
cd "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"
python3 RQ1_transition_rates_and_speeds/step8_survival_analysis/scripts/run_all_analysis.py
```

### Option 2: Run Each Step Individually

```bash
python3 RQ1_transition_rates_and_speeds/step8_survival_analysis/scripts/1_prepare_survival_data.py
python3 RQ1_transition_rates_and_speeds/step8_survival_analysis/scripts/2_kaplan_meier_analysis.py
python3 RQ1_transition_rates_and_speeds/step8_survival_analysis/scripts/3_cox_regression.py
python3 RQ1_transition_rates_and_speeds/step8_survival_analysis/scripts/4_validate_results.py
```

## Outputs

- Results JSONs and reports saved under `results/`
- Plots saved under `visualizations/`
- Prepared datasets saved under `data/`


