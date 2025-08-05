# From Newcomer to Core: A Comparative Study of Developer Transitions in OSS and OSS4SG Communities

## Abstract

Open Source Software (OSS) projects depend on newcomers becoming core contributors to ensure sustainability, yet only a small fraction successfully make this transition. While Open Source Software for Social Good (OSS4SG) projects exhibit distinct community dynamics due to their mission-driven nature, no empirical work has examined whether these differences affect newcomer advancement to leadership positions. This paper presents the first comparative analysis of newcomer-to-core transitions between conventional OSS and OSS4SG projects.

## Research Questions

We investigate four research questions:

**RQ1.** How frequently and how fast do newcomers become core contributors in OSS vs OSS4SG?

This establishes both the rate of successful transitions and the time required, providing essential context for understanding differences between ecosystems.

**RQ2.** How are successful newcomers treated during their transition journey in OSS vs OSS4SG?

Analyzing community response patterns reveals how different support mechanisms facilitate or hinder newcomer advancement.

**RQ3.** What engagement patterns characterize successful newcomer transitions in OSS vs OSS4SG?

Identifying distinct contribution trajectories helps understand diverse pathways to core status and informs targeted support strategies.

## Dataset

This study analyzes **375 verified open source projects** systematically selected and verified to meet strict quality criteria:

- **185 conventional OSS projects** (49.3%)
- **190 OSS for Social Good (OSS4SG) projects** (50.7%)

### Project Selection Criteria

All projects meet the following 5 systematic criteria:

1. **≥ 10 contributors** - Ensures meaningful community involvement
2. **≥ 500 commits** - Indicates substantial development activity  
3. **≥ 50 closed Pull Requests** - Demonstrates collaborative development
4. **> 1 year of project history** - Ensures project maturity
5. **Updated within last year** - Confirms active maintenance

### Stratification

Projects are stratified across:

- **Languages**: JavaScript/TypeScript, Python, Java/Kotlin, Other (Go, Rust, C++, C#, Ruby)
- **Size Tiers**: 500-1K, 1K-5K, 5K-15K, 15K-50K, 50K+ stars

## Repository Structure

```
├── README.md                    # This file
├── PROJECT_OVERVIEW.txt         # Complete methodology and results documentation
├── Dataset/                     # Original raw datasets
│   ├── OSS-Project-List.csv
│   ├── OSS4SG-Project-List.csv
│   ├── Active-OSS-Project-Info.csv
│   └── Active-OSS4SG-Project-Info.csv
├── preparing_dataset/           # Systematic dataset preparation
│   ├── README.md                # Detailed methodology documentation
│   ├── data/                    # Final verified datasets (375 projects)
│   ├── scripts/                 # Verification and processing scripts
│   ├── verification_results/    # Detailed verification outputs
│   └── documentation/           # Additional documentation
└── RQ1_transition_rates_and_speeds/  # ✅ COMPLETED: Community structure analysis
    ├── data_mining/
    │   ├── step1_repository_cloning/     # 372 repositories cloned
    │   └── step2_commit_analysis/        # 3.5M commits with 21 metrics extracted
    └── step3_per_project_metrics/        # ✅ NEW: Statistical comparison results
        ├── calculate_project_metrics.py     # Per-project metrics calculation
        ├── statistical_comparison_analysis.py  # OSS vs OSS4SG comparison
        ├── project_metrics.csv              # 358 projects × 17 metrics
        └── showing_plotting_results/         # Publication-ready results
            ├── community_structure_boxplots.png/pdf
            ├── community_structure_violins.png/pdf
            ├── statistical_test_results.csv
            ├── summary_table_for_paper.csv
            └── analysis_summary.txt
```

## Key Contributions

1. **First empirical comparison** of community structures between OSS and OSS4SG projects
2. **Systematic dataset** of 375 verified projects with transparent methodology
3. **Comprehensive commit analysis** - 3.5M commits with 21 objective metrics from 366 projects
4. **Automated data mining pipeline** - reproducible infrastructure for large-scale OSS analysis
5. **Statistical evidence** showing OSS4SG projects have healthier community structures
6. **Publication-ready results** with significant findings across all measured dimensions

## 🏆 Major Research Findings

Our statistical analysis of 358 projects reveals **remarkable differences** between OSS and OSS4SG community structures:

### **OSS4SG Projects Show Consistently Healthier Communities:**

- **2.4× More Collaborative Leadership**: 12.9% vs 5.3% core contributor ratios (p<0.001)
- **Dramatically Better Newcomer Retention**: 25.1% vs 56.6% one-time contributors (p<0.001, large effect)
- **More Egalitarian Participation**: Lower Gini coefficients (0.832 vs 0.878, p<0.001)
- **Higher Project Resilience**: 3 vs 2 median bus factor (p<0.001)
- **80% Higher Recent Engagement**: 6.3% vs 3.5% active contributors (p<0.001)

**All 5 metrics show significant differences with medium to large effect sizes!**

## Methodology

### Data Collection

- **Real-time verification** using GitHub API v3
- **Systematic sampling** across language and size dimensions
- **Transparent criteria** with 98.9% verification rate
- **Comprehensive coverage** with detailed verification results

### Analysis Approach

**RQ1 - Transition Rates and Speed:**
- Monthly rate of newcomer to core transitions per project
- Time to core status (months from first contribution)
- Number of commits required (commit threshold)

**RQ2 - Community Treatment:**
- Time to first response on PRs/issues
- Number of discussion turns and unique contributors responding
- Response quality (length, actionability)
- Milestone tracking through newcomer journey

**RQ3 - Engagement Patterns:**
- Daily time series analysis of commits, issues, and pull requests
- Time series clustering using Soft-DTW
- Identification of successful transition "personas"

## Research Impact

Understanding newcomer-to-core transition patterns is crucial for:

- **Project sustainability** - ensuring continuous leadership pipeline
- **Developer retention** - reducing high turnover that negatively affects code quality
- **Community building** - creating effective support mechanisms
- **Individual growth** - guiding newcomers seeking advancement

High developer turnover negatively affects team cognition, performance, and code quality, increasing bug density and delaying issue resolution. This research provides actionable insights for both communities aiming to cultivate future leaders and newcomers seeking advancement pathways.

## Related Work

This work extends prior research on:

- **Newcomer barriers and support** (Steinmacher et al., Casalnuovo et al.)
- **Long-term contributor prediction** (Zhou & Mockus, Xia et al.)
- **OSS4SG ecosystem characteristics** (Huang et al., Fang et al.)

No previous study has examined how mission-driven motivations in OSS4SG projects affect newcomer advancement patterns compared to conventional OSS.

## Keywords

Open source software, OSS4SG, social good, newcomer onboarding, core contributors, developer transitions, community dynamics

## Target Conference

This research is being prepared for submission to **FSE 2026** (ACM SIGSOFT International Symposium on the Foundations of Software Engineering).

## Contact

For questions about the research methodology, dataset, or findings, please refer to the research team.

---

**Status**: ✅ RQ1 Community Structure Analysis COMPLETE - Ready for Paper Writing  
**Dataset**: 375 verified projects → 358 analyzed (17 metrics each)  
**Commits Extracted**: 3,519,946 with 21 objective metrics  
**Major Finding**: OSS4SG projects have significantly healthier community structures  
**Statistical Results**: 5/5 metrics significant (p<0.001), medium-large effect sizes  
**Target Venue**: FSE 2026