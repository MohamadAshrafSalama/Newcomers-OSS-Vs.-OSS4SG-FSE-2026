# From Newcomer to Core: A Comparative Study of Developer Transitions in OSS and OSS4SG Communities

## 1. Abstract (200 words)
- Problem, gap, contribution, key findings (OSS4SG healthier across 5 metrics)

## 2. Introduction
- Motivation: sustainability depends on newcomer-to-core transitions
- Gap: no comparative study of OSS vs OSS4SG transitions
- Contributions: dataset (375), pipeline (3.5M commits), RQ1 findings, roadmap for RQ2/RQ3

## 3. Dataset and Methodology
- Dataset: 375 projects (185 OSS, 190 OSS4SG), five inclusion criteria
- Pipeline overview: cloning, commit mining, metric extraction
- Core definition: 80% rule, measurement windows
- Normalization: number of code characters (as per prior work)

## 4. RQ1: Community Structure Differences
- Figures: 
  - Fig 1: Box plots — community_structure_boxplots.png (in `paper/figures/`)
  - Fig 2: Violin plots — community_structure_violins.png (in `paper/figures/`)
- Tables:
  - Table 1: Summary stats — summary_table_for_paper.csv (in `paper/tables/`)
  - Table 2: Statistical tests — statistical_test_results.csv (in `paper/tables/`)
- Key results: higher core ratios, lower one-time contributors, lower Gini, higher bus factor, higher recent engagement (all p<0.001)

## 5. Threats to Validity
- Internal: core definition sensitivity, commit-based measurement biases
- External: project selection and generalizability
- Construct: proxy measures for community health

## 6. Related Work
- Newcomer onboarding and barriers
- Long-term contributor prediction
- OSS4SG ecosystem characteristics

## 7. Roadmap for RQ2 and RQ3 (Future Work in this paper)
- RQ2: treatment dynamics via PR/issue response metrics
- RQ3: engagement time-series clustering (Soft-DTW)

## 8. Conclusion
- Summary: OSS4SG exhibits systematically healthier community structures
- Implications: onboarding practices and leadership sustainability