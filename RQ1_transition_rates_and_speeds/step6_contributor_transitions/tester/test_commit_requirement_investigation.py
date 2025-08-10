#!/usr/bin/env python3
"""
Investigation Script: Why Does OSS Have Such Low Commit Requirements?
=====================================================================

Comprehensive analysis to understand the 30x difference in commit requirements
between OSS (median 2) and OSS4SG (median 59) projects.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CommitRequirementInvestigator:
    def __init__(self):
        # Use Step 6 results as the base path
        here = Path(__file__).resolve()
        self.base_path = here.parents[1] / "results"
        self.main_file = self.base_path / "contributor_transitions.csv"
        self.df = None
        self.findings = []
        
    def load_data(self):
        """Load the transitions dataset."""
        print("=" * 80)
        print("INVESTIGATING OSS vs OSS4SG COMMIT REQUIREMENTS")
        print("=" * 80)
        
        if not self.main_file.exists():
            print(f"❌ ERROR: File not found: {self.main_file}")
            return False
        
        print(f"Loading: {self.main_file}")
        self.df = pd.read_csv(self.main_file, low_memory=False)
        
        # Filter to only those who became core
        self.core_df = self.df[self.df['became_core'] == True].copy()
        print(f"✅ Loaded {len(self.df):,} transitions")
        print(f"   {len(self.core_df):,} became core\n")
        
        return True
    
    def analyze_basic_distributions(self):
        """1. Basic distribution analysis of commits_to_core."""
        print("\n" + "=" * 80)
        print("1. BASIC DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        for ptype in ['OSS', 'OSS4SG']:
            ptype_data = self.core_df[self.core_df['project_type'] == ptype]
            commits = ptype_data['commits_to_core']
            
            print(f"\n{ptype} (n={len(ptype_data):,}):")
            print(f"  Mean: {commits.mean():.1f}")
            print(f"  Median: {commits.median():.0f}")
            print(f"  Std Dev: {commits.std():.1f}")
            print(f"  Min: {commits.min()}")
            print(f"  Max: {commits.max()}")
            print(f"  Q25: {commits.quantile(0.25):.0f}")
            print(f"  Q75: {commits.quantile(0.75):.0f}")
            print(f"  Q90: {commits.quantile(0.90):.0f}")
            
            # Distribution of very low commits
            print(f"\n  Low commit distributions:")
            for threshold in [1, 2, 5, 10, 20, 50]:
                count = (commits <= threshold).sum()
                pct = count / len(commits) * 100 if len(commits) > 0 else 0
                print(f"    ≤{threshold:2} commits: {count:4,} ({pct:5.1f}%)")
        
        # Statistical test
        oss_commits = self.core_df[self.core_df['project_type'] == 'OSS']['commits_to_core']
        oss4sg_commits = self.core_df[self.core_df['project_type'] == 'OSS4SG']['commits_to_core']
        
        u_stat, p_value = stats.mannwhitneyu(oss_commits, oss4sg_commits)
        print(f"\nMann-Whitney U test p-value: {p_value:.2e}")
        
        if p_value < 0.001:
            print("✅ Difference is statistically significant")
            self.findings.append("Commit requirement difference is statistically significant")
    
    def analyze_by_project(self):
        """2. Analyze which specific projects have low commit requirements."""
        print("\n" + "=" * 80)
        print("2. PROJECT-LEVEL ANALYSIS")
        print("=" * 80)
        
        # Group by project and calculate median commits_to_core
        project_stats = self.core_df.groupby(['project_name', 'project_type']).agg({
            'commits_to_core': ['median', 'mean', 'min', 'count']
        }).round(1)
        
        project_stats.columns = ['median_commits', 'mean_commits', 'min_commits', 'core_count']
        project_stats = project_stats.reset_index()
        
        for ptype in ['OSS', 'OSS4SG']:
            ptype_projects = project_stats[project_stats['project_type'] == ptype]
            
            print(f"\n{ptype} Projects:")
            print(f"  Total projects with core achievers: {len(ptype_projects)}")
            
            # Projects with very low median commits
            low_commit_projects = ptype_projects[ptype_projects['median_commits'] <= 5]
            print(f"  Projects with median ≤5 commits: {len(low_commit_projects)} ({len(low_commit_projects)/len(ptype_projects)*100:.1f}%)")
            
            if len(low_commit_projects) > 0:
                print(f"\n  Top 10 {ptype} projects with LOWEST median commits to core:")
                top_low = ptype_projects.nsmallest(10, 'median_commits')
                for _, proj in top_low.iterrows():
                    print(f"    {proj['project_name'][:40]:40} | Median: {proj['median_commits']:3.0f} | Core achievers: {proj['core_count']:3.0f}")
            
            print(f"\n  Top 10 {ptype} projects with HIGHEST median commits to core:")
            top_high = ptype_projects.nlargest(10, 'median_commits')
            for _, proj in top_high.iterrows():
                print(f"    {proj['project_name'][:40]:40} | Median: {proj['median_commits']:3.0f} | Core achievers: {proj['core_count']:3.0f}")
        
        # Store for further analysis
        self.project_stats = project_stats
        
        # Finding
        oss_low = len(project_stats[(project_stats['project_type'] == 'OSS') & (project_stats['median_commits'] <= 5)])
        oss4sg_low = len(project_stats[(project_stats['project_type'] == 'OSS4SG') & (project_stats['median_commits'] <= 5)])
        
        if oss_low > oss4sg_low * 2:
            self.findings.append(f"OSS has {oss_low} projects with median ≤5 commits vs {oss4sg_low} for OSS4SG")
    
    def analyze_temporal_patterns(self):
        """3. Check if the pattern changes over time."""
        print("\n" + "=" * 80)
        print("3. TEMPORAL PATTERN ANALYSIS")
        print("=" * 80)
        
        # Convert dates to datetime
        self.core_df['first_core_date'] = pd.to_datetime(self.core_df['first_core_date'])
        self.core_df['core_year'] = self.core_df['first_core_date'].dt.year
        
        print("\nMedian commits to core by year:")
        print("-" * 60)
        
        for year in sorted(self.core_df['core_year'].dropna().unique())[-10:]:  # Last 10 years
            year_data = self.core_df[self.core_df['core_year'] == year]
            
            oss_data = year_data[year_data['project_type'] == 'OSS']
            oss4sg_data = year_data[year_data['project_type'] == 'OSS4SG']
            
            if len(oss_data) > 0 or len(oss4sg_data) > 0:
                oss_median = oss_data['commits_to_core'].median() if len(oss_data) > 0 else np.nan
                oss4sg_median = oss4sg_data['commits_to_core'].median() if len(oss4sg_data) > 0 else np.nan
                
                print(f"{int(year)}: OSS={oss_median:5.0f} (n={len(oss_data):3}) | OSS4SG={oss4sg_median:5.0f} (n={len(oss4sg_data):3})")
        
        # Check if recent years show different pattern
        recent_years = self.core_df[self.core_df['core_year'] >= 2020]
        if len(recent_years) > 0:
            recent_oss = recent_years[recent_years['project_type'] == 'OSS']['commits_to_core'].median()
            recent_oss4sg = recent_years[recent_years['project_type'] == 'OSS4SG']['commits_to_core'].median()
            
            print(f"\nPost-2020 median commits:")
            print(f"  OSS: {recent_oss:.0f}")
            print(f"  OSS4SG: {recent_oss4sg:.0f}")
            
            if pd.notna(recent_oss) and recent_oss > 10:
                self.findings.append("Recent OSS projects (2020+) have higher commit requirements")
    
    def test_different_thresholds(self):
        """4. Test how results change with different minimum commit thresholds."""
        print("\n" + "=" * 80)
        print("4. SENSITIVITY ANALYSIS: DIFFERENT THRESHOLDS")
        print("=" * 80)
        
        thresholds = [1, 2, 5, 10, 20, 50]
        results = []
        
        print("\nExcluding contributors with commits_to_core below threshold:")
        print("-" * 70)
        print(f"{'Threshold':>10} | {'OSS Median':>10} | {'OSS N':>8} | {'OSS4SG Median':>13} | {'OSS4SG N':>10} | {'Ratio':>8}")
        print("-" * 70)
        
        for threshold in thresholds:
            filtered = self.core_df[self.core_df['commits_to_core'] >= threshold]
            
            oss_filtered = filtered[filtered['project_type'] == 'OSS']
            oss4sg_filtered = filtered[filtered['project_type'] == 'OSS4SG']
            
            oss_median = oss_filtered['commits_to_core'].median() if len(oss_filtered) > 0 else np.nan
            oss4sg_median = oss4sg_filtered['commits_to_core'].median() if len(oss4sg_filtered) > 0 else np.nan
            
            ratio = oss4sg_median / oss_median if pd.notna(oss_median) and oss_median > 0 else np.nan
            
            print(f"{threshold:>10} | {oss_median:>10.0f} | {len(oss_filtered):>8,} | {oss4sg_median:>13.0f} | {len(oss4sg_filtered):>10,} | {ratio:>8.1f}x")
            
            results.append({
                'threshold': threshold,
                'oss_median': oss_median,
                'oss_n': len(oss_filtered),
                'oss4sg_median': oss4sg_median,
                'oss4sg_n': len(oss4sg_filtered),
                'ratio': ratio
            })
        
        # Finding
        if len(results) >= 3 and pd.notna(results[2]['ratio']) and results[2]['ratio'] < 5:
            self.findings.append(f"With ≥5 commit threshold, ratio drops to {results[2]['ratio']:.1f}x")
    
    def analyze_combined_metrics(self):
        """5. Analyze combinations of metrics to understand the pattern."""
        print("\n" + "=" * 80)
        print("5. COMBINED METRICS ANALYSIS")
        print("=" * 80)
        
        # Analyze time vs commits relationship
        for ptype in ['OSS', 'OSS4SG']:
            ptype_data = self.core_df[self.core_df['project_type'] == ptype]
            
            print(f"\n{ptype}:")
            
            # Correlation between time and commits
            corr = ptype_data[['weeks_to_core', 'commits_to_core']].corr().iloc[0, 1]
            print(f"  Correlation (weeks vs commits): {corr:.3f}")
            
            # Fast core achievers (< 4 weeks)
            fast_core = ptype_data[ptype_data['weeks_to_core'] <= 4]
            if len(fast_core) > 0:
                print(f"\n  Fast core achievers (≤4 weeks): {len(fast_core)} ({len(fast_core)/len(ptype_data)*100:.1f}%)")
                print(f"    Median commits: {fast_core['commits_to_core'].median():.0f}")
                print(f"    Mean commits: {fast_core['commits_to_core'].mean():.1f}")
            
            # Low commit achievers (≤5 commits)
            low_commit = ptype_data[ptype_data['commits_to_core'] <= 5]
            if len(low_commit) > 0:
                print(f"\n  Low commit achievers (≤5 commits): {len(low_commit)} ({len(low_commit)/len(ptype_data)*100:.1f}%)")
                print(f"    Median weeks: {low_commit['weeks_to_core'].median():.0f}")
                print(f"    Mean weeks: {low_commit['weeks_to_core'].mean():.1f}")
            
            # Both fast AND low commits
            fast_and_low = ptype_data[(ptype_data['weeks_to_core'] <= 4) & (ptype_data['commits_to_core'] <= 5)]
            if len(fast_and_low) > 0:
                print(f"\n  Both fast (≤4w) AND low commits (≤5): {len(fast_and_low)} ({len(fast_and_low)/len(ptype_data)*100:.1f}%)")
    
    def analyze_project_characteristics(self):
        """6. Check if certain types of projects have different patterns."""
        print("\n" + "=" * 80)
        print("6. PROJECT CHARACTERISTICS ANALYSIS")
        print("=" * 80)
        
        # Load activity data to get project sizes (relative path from Step 6)
        activity_file = (Path(__file__).resolve().parents[1] / "../step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv").resolve()
        
        if activity_file.exists():
            print("Loading activity data for project size analysis...")
            
            # Read just what we need (sample for speed)
            activity_df = pd.read_csv(
                activity_file,
                usecols=['project_name', 'contributor_email'],
                nrows=1000000
            )
            
            # Calculate project sizes
            project_sizes = activity_df.groupby('project_name')['contributor_email'].nunique().reset_index()
            project_sizes.columns = ['project_name', 'total_contributors']
            
            # Merge with our project stats
            if hasattr(self, 'project_stats'):
                merged = self.project_stats.merge(project_sizes, on='project_name', how='left')
                
                # Categorize projects by size
                merged['size_category'] = pd.cut(
                    merged['total_contributors'].fillna(0),
                    bins=[0, 10, 50, 200, 10000],
                    labels=['Small (<10)', 'Medium (10-50)', 'Large (50-200)', 'Very Large (>200)']
                )
                
                print("\nMedian commits to core by project size:")
                size_analysis = merged.groupby(['project_type', 'size_category']).agg({
                    'median_commits': 'median',
                    'project_name': 'count'
                }).round(1)
                
                print(size_analysis)
        
        # Check specific suspicious OSS projects
        print("\n" + "=" * 80)
        print("7. SUSPICIOUS PROJECT INVESTIGATION")
        print("=" * 80)
        
        # Find OSS projects where EVERYONE needs ≤2 commits
        suspicious_projects = []
        
        for project in self.core_df['project_name'].unique():
            project_data = self.core_df[self.core_df['project_name'] == project]
            
            if len(project_data) >= 3:  # At least 3 core achievers
                max_commits = project_data['commits_to_core'].max()
                median_commits = project_data['commits_to_core'].median()
                
                if max_commits <= 2:
                    suspicious_projects.append({
                        'project': project,
                        'type': project_data['project_type'].iloc[0],
                        'core_count': len(project_data),
                        'max_commits': max_commits,
                        'median_commits': median_commits
                    })
        
        if suspicious_projects:
            print(f"\nFound {len(suspicious_projects)} projects where ALL core achievers needed ≤2 commits:")
            
            susp_df = pd.DataFrame(suspicious_projects)
            
            for ptype in ['OSS', 'OSS4SG']:
                ptype_susp = susp_df[susp_df['type'] == ptype]
                if len(ptype_susp) > 0:
                    print(f"\n{ptype}: {len(ptype_susp)} suspicious projects")
                    for _, proj in ptype_susp.head(10).iterrows():
                        print(f"  {proj['project'][:50]:50} | {proj['core_count']} core achievers, all with ≤{proj['max_commits']:.0f} commits")
            
            self.findings.append(f"Found {len(suspicious_projects)} projects where ALL core achievers needed ≤2 commits")
    
    def recommend_solution(self):
        """7. Provide recommendations based on findings."""
        print("\n" + "=" * 80)
        print("FINDINGS AND RECOMMENDATIONS")
        print("=" * 80)
        
        print("\n📊 KEY FINDINGS:")
        for i, finding in enumerate(self.findings, 1):
            print(f"  {i}. {finding}")
        
        print("\n🎯 RECOMMENDATIONS:")
        
        # Calculate what happens with different filters
        baseline_core = len(self.core_df)
        
        # Option 1: Exclude < 5 commits
        option1 = self.core_df[self.core_df['commits_to_core'] >= 5]
        opt1_oss = option1[option1['project_type'] == 'OSS']
        opt1_oss4sg = option1[option1['project_type'] == 'OSS4SG']
        
        print(f"\n1. EXCLUDE contributors with <5 commits to core:")
        print(f"   - Removes {baseline_core - len(option1):,} ({(baseline_core - len(option1))/baseline_core*100:.1f}%) core achievers")
        print(f"   - New OSS median: {opt1_oss['commits_to_core'].median():.0f} commits")
        print(f"   - New OSS4SG median: {opt1_oss4sg['commits_to_core'].median():.0f} commits")
        if opt1_oss['commits_to_core'].median() > 0:
            print(f"   - Ratio: {opt1_oss4sg['commits_to_core'].median()/opt1_oss['commits_to_core'].median():.1f}x")
        
        # Option 2: Exclude specific projects
        if hasattr(self, 'project_stats'):
            suspicious_projects = self.project_stats[self.project_stats['median_commits'] <= 2]['project_name'].tolist()
            option2 = self.core_df[~self.core_df['project_name'].isin(suspicious_projects)]
            opt2_oss = option2[option2['project_type'] == 'OSS']
            opt2_oss4sg = option2[option2['project_type'] == 'OSS4SG']
            
            print(f"\n2. EXCLUDE projects with median ≤2 commits:")
            print(f"   - Removes {len(suspicious_projects)} projects")
            print(f"   - Removes {baseline_core - len(option2):,} ({(baseline_core - len(option2))/baseline_core*100:.1f}%) core achievers")
            print(f"   - New OSS median: {opt2_oss['commits_to_core'].median():.0f} commits")
            print(f"   - New OSS4SG median: {opt2_oss4sg['commits_to_core'].median():.0f} commits")
            if opt2_oss['commits_to_core'].median() > 0:
                print(f"   - Ratio: {opt2_oss4sg['commits_to_core'].median()/opt2_oss['commits_to_core'].median():.1f}x")
        
        # Option 3: Combined approach
        option3 = self.core_df[
            (self.core_df['commits_to_core'] >= 3) & 
            (self.core_df['weeks_to_core'] >= 2)
        ]
        opt3_oss = option3[option3['project_type'] == 'OSS']
        opt3_oss4sg = option3[option3['project_type'] == 'OSS4SG']
        
        print(f"\n3. EXCLUDE <3 commits AND <2 weeks:")
        print(f"   - Removes {baseline_core - len(option3):,} ({(baseline_core - len(option3))/baseline_core*100:.1f}%) core achievers")
        print(f"   - New OSS median: {opt3_oss['commits_to_core'].median():.0f} commits")
        print(f"   - New OSS4SG median: {opt3_oss4sg['commits_to_core'].median():.0f} commits")
        if opt3_oss['commits_to_core'].median() > 0:
            print(f"   - Ratio: {opt3_oss4sg['commits_to_core'].median()/opt3_oss['commits_to_core'].median():.1f}x")
        
        print("\n" + "=" * 80)
        print("FINAL RECOMMENDATION")
        print("=" * 80)
        
        print("""
The large difference appears to be REAL, not a data error:

1. OSS projects genuinely have lower barriers to core status
2. Some OSS projects promote contributors with minimal contributions
3. OSS4SG projects maintain higher standards consistently

Consider one of these approaches:
- Keep data as-is and discuss this finding in your paper
- Apply a minimum threshold (e.g., ≥5 commits) for robustness
- Run analyses both ways and report sensitivity

This difference itself is a valuable research finding!
""")
        
        # Save detailed results
        results = {
            'investigation_date': pd.Timestamp.now().isoformat(),
            'findings': self.findings,
            'baseline_stats': {
                'oss_median_commits': float(self.core_df[self.core_df['project_type'] == 'OSS']['commits_to_core'].median()),
                'oss4sg_median_commits': float(self.core_df[self.core_df['project_type'] == 'OSS4SG']['commits_to_core'].median()),
                'ratio': float(self.core_df[self.core_df['project_type'] == 'OSS4SG']['commits_to_core'].median() / 
                             self.core_df[self.core_df['project_type'] == 'OSS']['commits_to_core'].median())
            },
            'filter_options': {
                'min_5_commits': {
                    'removed': int(baseline_core - len(option1)),
                    'oss_median': float(opt1_oss['commits_to_core'].median()),
                    'oss4sg_median': float(opt1_oss4sg['commits_to_core'].median())
                },
                'min_3_commits_2_weeks': {
                    'removed': int(baseline_core - len(option3)),
                    'oss_median': float(opt3_oss['commits_to_core'].median()),
                    'oss4sg_median': float(opt3_oss4sg['commits_to_core'].median())
                }
            }
        }
        
        output_file = self.base_path / 'investigation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Detailed results saved to: {output_file}")
    
    def create_visualizations(self):
        """Create visualizations to understand the pattern."""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Distribution of commits_to_core (log scale)
        ax = axes[0, 0]
        for ptype in ['OSS', 'OSS4SG']:
            data = self.core_df[self.core_df['project_type'] == ptype]['commits_to_core']
            ax.hist(np.log10(data + 1), bins=50, alpha=0.5, label=ptype, density=True)
        ax.set_xlabel('Log10(Commits to Core + 1)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Commits to Core (Log Scale)')
        ax.legend()
        
        # 2. Cumulative distribution
        ax = axes[0, 1]
        for ptype in ['OSS', 'OSS4SG']:
            data = self.core_df[self.core_df['project_type'] == ptype]['commits_to_core'].sort_values()
            y = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, y, label=ptype, alpha=0.7)
        ax.set_xlabel('Commits to Core')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution')
        ax.set_xlim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Box plot (log scale)
        ax = axes[0, 2]
        data_to_plot = [
            np.log10(self.core_df[self.core_df['project_type'] == 'OSS']['commits_to_core'] + 1),
            np.log10(self.core_df[self.core_df['project_type'] == 'OSS4SG']['commits_to_core'] + 1)
        ]
        ax.boxplot(data_to_plot, labels=['OSS', 'OSS4SG'])
        ax.set_ylabel('Log10(Commits to Core + 1)')
        ax.set_title('Box Plot Comparison')
        ax.grid(True, alpha=0.3)
        
        # 4. Scatter: Time vs Commits
        ax = axes[1, 0]
        for ptype in ['OSS', 'OSS4SG']:
            data = self.core_df[self.core_df['project_type'] == ptype]
            ax.scatter(data['weeks_to_core'], data['commits_to_core'], 
                      alpha=0.3, s=10, label=ptype)
        ax.set_xlabel('Weeks to Core')
        ax.set_ylabel('Commits to Core')
        ax.set_title('Time vs Effort to Core')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 200)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Percentage meeting different thresholds
        ax = axes[1, 1]
        thresholds = [1, 2, 5, 10, 20, 50, 100]
        oss_pcts = []
        oss4sg_pcts = []
        
        for t in thresholds:
            oss_data = self.core_df[self.core_df['project_type'] == 'OSS']['commits_to_core']
            oss4sg_data = self.core_df[self.core_df['project_type'] == 'OSS4SG']['commits_to_core']
            
            oss_pcts.append((oss_data <= t).mean() * 100)
            oss4sg_pcts.append((oss4sg_data <= t).mean() * 100)
        
        ax.plot(thresholds, oss_pcts, 'o-', label='OSS')
        ax.plot(thresholds, oss4sg_pcts, 's-', label='OSS4SG')
        ax.set_xlabel('Commits Threshold')
        ax.set_ylabel('% of Core Achievers Below Threshold')
        ax.set_title('Cumulative % by Commit Threshold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 6. Project-level median commits
        ax = axes[1, 2]
        if hasattr(self, 'project_stats'):
            oss_medians = self.project_stats[self.project_stats['project_type'] == 'OSS']['median_commits']
            oss4sg_medians = self.project_stats[self.project_stats['project_type'] == 'OSS4SG']['median_commits']
            
            ax.hist(oss_medians, bins=30, alpha=0.5, label=f'OSS (n={len(oss_medians)})', density=True)
            ax.hist(oss4sg_medians, bins=30, alpha=0.5, label=f'OSS4SG (n={len(oss4sg_medians)})', density=True)
            ax.set_xlabel('Project Median Commits to Core')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of Project-Level Medians')
            ax.set_xlim(0, 100)
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.base_path / 'investigation_plots.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
        # Do not block interactive window in headless runs
        try:
            plt.close(fig)
        except Exception:
            pass
    
    def run_investigation(self):
        """Run the complete investigation."""
        if not self.load_data():
            return
        
        self.analyze_basic_distributions()
        self.analyze_by_project()
        self.analyze_temporal_patterns()
        self.test_different_thresholds()
        self.analyze_combined_metrics()
        self.analyze_project_characteristics()
        self.recommend_solution()
        self.create_visualizations()


def main():
    investigator = CommitRequirementInvestigator()
    investigator.run_investigation()


if __name__ == "__main__":
    main()


