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
├── Dataset/                     # Original raw datasets
│   ├── OSS-Project-List.csv
│   ├── OSS4SG-Project-List.csv
│   ├── Active-OSS-Project-Info.csv
│   └── Active-OSS4SG-Project-Info.csv
└── preparing_dataset/           # Systematic dataset preparation
    ├── README.md                # Detailed methodology documentation
    ├── data/                    # Final verified datasets
    ├── scripts/                 # Verification and processing scripts
    ├── verification_results/    # Detailed verification outputs
    └── documentation/           # Additional documentation
```

## Key Contributions

1. **First empirical comparison** of newcomer-to-core transitions between OSS and OSS4SG projects
2. **Systematic dataset** of 280 verified projects with transparent methodology
3. **Community treatment analysis** revealing how different ecosystems support newcomer advancement
4. **Engagement pattern identification** showing diverse pathways to core contributor status

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

**Status**: In Development  
**Dataset**: 375 verified projects  
**Verification Rate**: 98.9%  
**Balance Ratio**: 1:1.03 (OSS:OSS4SG)  
**Target Venue**: FSE 2026