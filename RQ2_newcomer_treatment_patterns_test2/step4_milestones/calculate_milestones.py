#!/usr/bin/env python3
"""
Script: milestone_detection_complete.py
Purpose: Detect all 7 milestones for contributors and create comprehensive visualizations
Author: RQ2 Analysis Pipeline
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MilestoneAnalyzer:
    def __init__(self, base_path):
        """Initialize the milestone analyzer"""
        self.base_path = Path(base_path)
        self.timeline_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step2_timelines" / "from_cache_timelines"
        self.output_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step4_milestones" / "results"
        self.viz_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step4_milestones" / "visualizations"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Load contributor metadata
        self.load_metadata()
        
        # Define colors for consistency
        self.colors = {
            'OSS': '#2E86AB',
            'OSS4SG': '#A23B72'
        }
        
    def load_metadata(self):
        """Load contributor transition data for core contributors"""
        transitions_path = self.base_path / "RQ1_transition_rates_and_speeds" / "step6_contributor_transitions" / "results" / "contributor_transitions.csv"
        self.transitions_df = pd.read_csv(transitions_path)
        self.core_contributors = self.transitions_df[self.transitions_df['became_core'] == True].copy()
        print(f"Loaded {len(self.core_contributors)} core contributors")
        
    def detect_milestones_for_contributor(self, timeline_df, contributor_info):
        """Detect all 7 milestones for a single contributor"""
        
        milestones = {
            'first_accepted': {'achieved': False, 'week': None},
            'sustained_participation': {'achieved': False, 'week': None},
            'returning_contributor': {'achieved': False, 'week': None},
            'trusted_reviewer': {'achieved': False, 'week': None},
            'cross_boundary': {'achieved': False, 'week': None},
            'community_helper': {'achieved': False, 'week': None},
            'failure_recovery': {'achieved': False, 'week': None}
        }
        
        # Sort timeline by timestamp
        timeline_df = timeline_df.sort_values('event_timestamp').reset_index(drop=True)
        
        # Parse JSON data safely
        def safe_json_parse(x):
            try:
                return json.loads(x) if pd.notna(x) else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        
        timeline_df['data'] = timeline_df['event_data'].apply(safe_json_parse)
        
        # Derive contributor identifiers
        contributor_email = contributor_info.get('contributor_email', None)
        try:
            contributor_username = timeline_df['username'].iloc[0]
        except Exception:
            contributor_username = None

        # 1. FIRST ACCEPTED CONTRIBUTION
        prs = timeline_df[timeline_df['event_type'] == 'pull_request']
        def _is_merged(pr_data: dict) -> bool:
            # Support both boolean merged flag and state strings
            if pr_data is None:
                return False
            if isinstance(pr_data.get('merged'), bool):
                return pr_data.get('merged') is True
            return pr_data.get('state') == 'MERGED'

        merged_prs = prs[prs['data'].apply(_is_merged)]
        
        if len(merged_prs) > 0:
            first_merged = merged_prs.iloc[0]
            milestones['first_accepted']['achieved'] = True
            milestones['first_accepted']['week'] = int(first_merged['event_week'])
        
        # 2. SUSTAINED PARTICIPATION (5 contributions in 90 days)
        # Count merged PRs and commits
        commits = timeline_df[timeline_df['event_type'] == 'commit']
        
        contributions_list = []
        
        # Add merged PRs
        if len(merged_prs) > 0:
            contributions_list.append(merged_prs[['event_timestamp', 'event_week']])
        
        # Add commits
        if len(commits) > 0:
            contributions_list.append(commits[['event_timestamp', 'event_week']])
        
        if contributions_list:
            contributions = pd.concat(contributions_list).sort_values('event_timestamp')
        else:
            contributions = pd.DataFrame(columns=['event_timestamp', 'event_week'])
        
        if len(contributions) >= 5:
            contributions['event_date'] = pd.to_datetime(contributions['event_timestamp']).dt.date
            
            for i in range(len(contributions) - 4):
                window = contributions.iloc[i:i+5]
                days_span = (window.iloc[-1]['event_date'] - window.iloc[0]['event_date']).days
                
                if days_span <= 90:
                    milestones['sustained_participation']['achieved'] = True
                    milestones['sustained_participation']['week'] = int(window.iloc[-1]['event_week'])
                    break
        
        # 3. RETURNING CONTRIBUTOR (30+ day gap then return)
        if len(timeline_df) > 1:
            timeline_df['timestamp_dt'] = pd.to_datetime(timeline_df['event_timestamp'])
            timeline_df['days_since_prev'] = timeline_df['timestamp_dt'].diff().dt.days
            
            returns = timeline_df[timeline_df['days_since_prev'] > 30]
            if len(returns) > 0:
                milestones['returning_contributor']['achieved'] = True
                milestones['returning_contributor']['week'] = int(returns.iloc[0]['event_week'])
        
        # 4. TRUSTED REVIEWER (reviews others' PRs)
        # Look for review events where contributor reviewed someone else's PR
        for _, event in timeline_df.iterrows():
            if event['event_type'] != 'pull_request':
                continue

            pr_data = event['data'] or {}

            # Determine PR author login (if available)
            pr_author_login = None
            pr_author = pr_data.get('author')
            if isinstance(pr_author, dict):
                pr_author_login = pr_author.get('login')
            elif isinstance(pr_author, str):
                pr_author_login = pr_author

            # Normalize reviews into a flat list of dicts
            reviews_field = pr_data.get('reviews', [])
            review_nodes = []
            if isinstance(reviews_field, dict):
                review_nodes = reviews_field.get('nodes', []) or []
            elif isinstance(reviews_field, list):
                review_nodes = reviews_field

            for review in review_nodes:
                reviewer_login = None
                if isinstance(review, dict):
                    # GraphQL shape: { author: { login }, state, ... }
                    author_obj = review.get('author')
                    if isinstance(author_obj, dict):
                        reviewer_login = author_obj.get('login')
                    # Fallbacks
                    if reviewer_login is None:
                        reviewer_login = review.get('reviewer')
                elif isinstance(review, str):
                    reviewer_login = review

                # Match by username primarily; fallback to email if stored as string
                if reviewer_login and (reviewer_login == contributor_username or reviewer_login == contributor_email):
                    # Ensure reviewing someone else's PR
                    if pr_author_login and pr_author_login != reviewer_login:
                        milestones['trusted_reviewer']['achieved'] = True
                        milestones['trusted_reviewer']['week'] = int(event['event_week'])
                        break

            if milestones['trusted_reviewer']['achieved']:
                break
        
        # 5. CROSS-BOUNDARY CONTRIBUTION
        # Track contribution areas based on file types or PR/commit messages
        contribution_areas = []
        
        for _, event in timeline_df.iterrows():
            area = None
            
            if event['event_type'] == 'commit':
                msg = event['data'].get('message', '')
                # Handle case where message might be NaN/float
                if isinstance(msg, str):
                    msg = msg.lower()
                    if any(word in msg for word in ['doc', 'readme', 'tutorial']):
                        area = 'documentation'
                    elif any(word in msg for word in ['test', 'spec', 'coverage']):
                        area = 'testing'
                    elif any(word in msg for word in ['fix', 'bug', 'patch']):
                        area = 'bugfix'
                    else:
                        area = 'feature'
                else:
                    area = 'feature'  # Default for non-string messages
            
            elif event['event_type'] == 'pull_request':
                title = event['data'].get('title', '')
                # Handle case where title might be NaN/float
                if isinstance(title, str):
                    title = title.lower()
                    if 'doc' in title:
                        area = 'documentation'
                    elif 'test' in title:
                        area = 'testing'
                    elif 'fix' in title or 'bug' in title:
                        area = 'bugfix'
                    else:
                        area = 'feature'
                else:
                    area = 'feature'  # Default for non-string titles
            
            if area and area not in contribution_areas:
                if len(contribution_areas) > 0:  # This is a boundary crossing
                    milestones['cross_boundary']['achieved'] = True
                    milestones['cross_boundary']['week'] = int(event['event_week'])
                    break
                contribution_areas.append(area)
        
        # 6. COMMUNITY HELPER (helps another contributor)
        # Look for helpful comments on others' PRs/issues
        helpful_keywords = ['try', 'should', 'could', 'suggest', 'recommend',
                            "here's how", 'example', 'solution', 'fix']

        for _, event in timeline_df.iterrows():
            if event['event_type'] not in ['pull_request', 'issue']:
                continue

            data = event['data'] or {}

            # Determine original author login (if available)
            original_author_login = None
            original_author = data.get('author')
            if isinstance(original_author, dict):
                original_author_login = original_author.get('login')
            elif isinstance(original_author, str):
                original_author_login = original_author

            # Normalize comments
            comments_field = data.get('comments', {})
            comment_nodes = []
            if isinstance(comments_field, dict):
                comment_nodes = comments_field.get('nodes', []) or []
            elif isinstance(comments_field, list):
                comment_nodes = comments_field

            for conv in comment_nodes:
                if not isinstance(conv, dict):
                    continue

                # Identify comment author
                comment_author_login = None
                author_obj = conv.get('author')
                if isinstance(author_obj, dict):
                    comment_author_login = author_obj.get('login')
                elif isinstance(author_obj, str):
                    comment_author_login = author_obj

                # Only count if made by the contributor on someone else's thread
                if comment_author_login and (comment_author_login == contributor_username or comment_author_login == contributor_email):
                    if original_author_login and original_author_login != comment_author_login:
                        # Check if the comment text appears helpful
                        text = conv.get('body') or conv.get('text') or ''
                        if isinstance(text, str) and any(k in text.lower() for k in helpful_keywords):
                            milestones['community_helper']['achieved'] = True
                            milestones['community_helper']['week'] = int(event['event_week'])
                            break

            if milestones['community_helper']['achieved']:
                break
        
        # 7. FAILURE RECOVERY (success after rejection)
        pr_history = []
        for _, pr in prs.iterrows():
            state = pr['data'].get('state')
            week = int(pr['event_week'])
            pr_history.append((week, state))
        
        # Look for CLOSED followed by MERGED
        for i in range(len(pr_history) - 1):
            if pr_history[i][1] == 'CLOSED':
                # Check subsequent PRs for success
                for j in range(i + 1, len(pr_history)):
                    if pr_history[j][1] == 'MERGED':
                        milestones['failure_recovery']['achieved'] = True
                        milestones['failure_recovery']['week'] = pr_history[j][0]
                        break
                
                if milestones['failure_recovery']['achieved']:
                    break
        
        return milestones
    
    def process_all_contributors(self):
        """Process all contributors and detect milestones"""
        
        all_results = []
        
        for _, contributor in tqdm(self.core_contributors.iterrows(), 
                                  total=len(self.core_contributors),
                                  desc="Detecting milestones"):
            
            # Build timeline filename (matches the actual format in the directory)
            # Only replace @ with _at_, keep dots as-is
            email_safe = contributor['contributor_email'].replace('@', '_at_')
            project_safe = contributor['project_name'].replace('/', '_')
            timeline_file = self.timeline_dir / f"timeline_{project_safe}_{email_safe}.csv"
            
            if not timeline_file.exists():
                continue
            
            # Load timeline
            try:
                timeline_df = pd.read_csv(timeline_file)
                
                if len(timeline_df) == 0:
                    continue
                
                # Check required columns
                required_cols = ['event_timestamp', 'event_week', 'event_type', 'event_data', 'is_pre_core']
                missing_cols = [col for col in required_cols if col not in timeline_df.columns]
                if missing_cols:
                    print(f"Warning: Skipping {timeline_file.name} - missing columns: {missing_cols}")
                    continue
                
                # CRITICAL FIX: Filter to only pre-core events (newcomer transition period)
                pre_core_df = timeline_df[timeline_df['is_pre_core'] == True].copy()
                if len(pre_core_df) == 0:
                    continue  # Skip if no pre-core events
                
                # Use the filtered timeline for milestone detection
                timeline_df = pre_core_df
                    
            except Exception as e:
                print(f"Error loading {timeline_file.name}: {e}")
                continue
            
            # Detect milestones
            milestones = self.detect_milestones_for_contributor(timeline_df, contributor)
            
            # Create result record
            result = {
                'contributor_email': contributor['contributor_email'],
                'project_name': contributor['project_name'],
                'project_type': contributor['project_type'],
                'weeks_to_core': contributor['weeks_to_core']
            }
            
            # Add milestone data
            for milestone_name, milestone_data in milestones.items():
                result[f'{milestone_name}_achieved'] = milestone_data['achieved']
                result[f'{milestone_name}_week'] = milestone_data['week']
            
            all_results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_file = self.output_dir / "milestone_detection_results.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\nProcessed {len(results_df)} contributors")
        print(f"Results saved to {output_file}")
        
        return results_df
    
    def remove_outliers(self, data, method='iqr', threshold=1.5):
        """Remove outliers using IQR or Z-score method"""
        if len(data) == 0:
            return data
            
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            return data[z_scores < threshold]
        
        return data
    
    def create_timing_visualizations(self, results_df):
        """Create box plots and violin plots for milestone timing"""
        
        milestones = [
            'first_accepted', 'sustained_participation', 'returning_contributor',
            'trusted_reviewer', 'cross_boundary', 'community_helper', 'failure_recovery'
        ]
        
        milestone_labels = [
            'First\nAccepted', 'Sustained\nParticipation', 'Returning\nContributor',
            'Trusted\nReviewer', 'Cross\nBoundary', 'Community\nHelper', 'Failure\nRecovery'
        ]
        
        # Create two versions: with and without outliers
        for remove_outliers in [False, True]:
            suffix = '_no_outliers' if remove_outliers else '_with_outliers'
            
            # 1. BOX PLOTS
            fig, axes = plt.subplots(2, 4, figsize=(16, 10))
            axes = axes.flatten()
            
            for idx, (milestone, label) in enumerate(zip(milestones, milestone_labels)):
                ax = axes[idx]
                
                # Get data for each project type
                oss_data = results_df[
                    (results_df['project_type'] == 'OSS') & 
                    (results_df[f'{milestone}_achieved'] == True)
                ][f'{milestone}_week'].dropna()
                
                oss4sg_data = results_df[
                    (results_df['project_type'] == 'OSS4SG') & 
                    (results_df[f'{milestone}_achieved'] == True)
                ][f'{milestone}_week'].dropna()
                
                # Remove outliers if requested
                if remove_outliers and len(oss_data) > 0:
                    oss_data = self.remove_outliers(oss_data)
                if remove_outliers and len(oss4sg_data) > 0:
                    oss4sg_data = self.remove_outliers(oss4sg_data)
                
                # Prepare data for plotting
                plot_data = []
                labels_plot = []
                
                if len(oss_data) > 0:
                    plot_data.append(oss_data)
                    labels_plot.append(f'OSS\n(n={len(oss_data)})')
                
                if len(oss4sg_data) > 0:
                    plot_data.append(oss4sg_data)
                    labels_plot.append(f'OSS4SG\n(n={len(oss4sg_data)})')
                
                if plot_data:
                    bp = ax.boxplot(plot_data, labels=labels_plot, patch_artist=True,
                                   showmeans=True, meanline=True)
                    
                    # Color the boxes
                    colors_list = []
                    if len(oss_data) > 0:
                        colors_list.append(self.colors['OSS'])
                    if len(oss4sg_data) > 0:
                        colors_list.append(self.colors['OSS4SG'])
                    
                    for patch, color in zip(bp['boxes'], colors_list):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Add statistical test
                    if len(oss_data) > 0 and len(oss4sg_data) > 0:
                        statistic, p_value = stats.mannwhitneyu(oss_data, oss4sg_data, alternative='two-sided')
                        
                        # Add significance stars
                        if p_value < 0.001:
                            sig_text = '***'
                        elif p_value < 0.01:
                            sig_text = '**'
                        elif p_value < 0.05:
                            sig_text = '*'
                        else:
                            sig_text = 'ns'
                        
                        ax.text(0.5, 0.95, f'p={p_value:.3f} {sig_text}',
                               transform=ax.transAxes, ha='center', va='top')
                
                ax.set_title(label, fontsize=10, fontweight='bold')
                ax.set_ylabel('Weeks to Achievement', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
            
            # Remove empty subplot
            fig.delaxes(axes[-1])
            
            title = 'Time to Achieve Milestones: OSS vs OSS4SG'
            if remove_outliers:
                title += ' (Outliers Removed)'
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            boxplot_file = self.viz_dir / f'milestone_timing_boxplots{suffix}.png'
            plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {boxplot_file}")
            
            # 2. VIOLIN PLOTS
            fig, axes = plt.subplots(2, 4, figsize=(16, 10))
            axes = axes.flatten()
            
            for idx, (milestone, label) in enumerate(zip(milestones, milestone_labels)):
                ax = axes[idx]
                
                # Prepare data
                plot_df = []
                
                for project_type in ['OSS', 'OSS4SG']:
                    data = results_df[
                        (results_df['project_type'] == project_type) & 
                        (results_df[f'{milestone}_achieved'] == True)
                    ][f'{milestone}_week'].dropna()
                    
                    if remove_outliers and len(data) > 0:
                        data = self.remove_outliers(data)
                    
                    for val in data:
                        plot_df.append({'Project Type': project_type, 'Weeks': val})
                
                if plot_df:
                    df_plot = pd.DataFrame(plot_df)
                    
                    # Prepare data for violin plot
                    oss_violin_data = df_plot[df_plot['Project Type'] == 'OSS']['Weeks'].values
                    oss4sg_violin_data = df_plot[df_plot['Project Type'] == 'OSS4SG']['Weeks'].values
                    
                    # Only create violin plot if we have data for at least one group
                    violin_data = []
                    positions = []
                    colors_for_violins = []
                    
                    if len(oss_violin_data) > 0:
                        violin_data.append(oss_violin_data)
                        positions.append(0)
                        colors_for_violins.append(self.colors['OSS'])
                    
                    if len(oss4sg_violin_data) > 0:
                        violin_data.append(oss4sg_violin_data)
                        positions.append(1 if len(oss_violin_data) > 0 else 0)
                        colors_for_violins.append(self.colors['OSS4SG'])
                    
                    if violin_data:
                        # Create violin plot
                        parts = ax.violinplot(
                            violin_data,
                            positions=positions,
                            showmeans=True,
                            showmedians=True
                        )
                        
                        # Color the violins
                        for pc, color in zip(parts['bodies'], colors_for_violins):
                            pc.set_facecolor(color)
                            pc.set_alpha(0.7)
                    
                    # Set appropriate x-axis labels
                    if len(oss_violin_data) > 0 and len(oss4sg_violin_data) > 0:
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels(['OSS', 'OSS4SG'])
                    elif len(oss_violin_data) > 0:
                        ax.set_xticks([0])
                        ax.set_xticklabels(['OSS'])
                    elif len(oss4sg_violin_data) > 0:
                        ax.set_xticks([0])
                        ax.set_xticklabels(['OSS4SG'])
                    
                    # Add sample sizes
                    n_oss = len(df_plot[df_plot['Project Type'] == 'OSS'])
                    n_oss4sg = len(df_plot[df_plot['Project Type'] == 'OSS4SG'])
                    ax.text(0.5, 0.05, f'n: {n_oss} | {n_oss4sg}',
                           transform=ax.transAxes, ha='center')
                
                ax.set_title(label, fontsize=10, fontweight='bold')
                ax.set_ylabel('Weeks to Achievement', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
            
            # Remove empty subplot
            fig.delaxes(axes[-1])
            
            title = 'Time to Achieve Milestones Distribution: OSS vs OSS4SG'
            if remove_outliers:
                title += ' (Outliers Removed)'
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            violinplot_file = self.viz_dir / f'milestone_timing_violins{suffix}.png'
            plt.savefig(violinplot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {violinplot_file}")
    
    def create_achievement_rate_comparison(self, results_df):
        """Create achievement rate comparison visualization"""
        
        milestones = [
            'first_accepted', 'sustained_participation', 'returning_contributor',
            'trusted_reviewer', 'cross_boundary', 'community_helper', 'failure_recovery'
        ]
        
        milestone_labels = [
            'First Accepted', 'Sustained\nParticipation', 'Returning\nContributor',
            'Trusted Reviewer', 'Cross Boundary', 'Community Helper', 'Failure Recovery'
        ]
        
        # Calculate achievement rates
        oss_rates = []
        oss4sg_rates = []
        p_values = []
        
        for milestone in milestones:
            oss_achieved = results_df[results_df['project_type'] == 'OSS'][f'{milestone}_achieved'].mean()
            oss4sg_achieved = results_df[results_df['project_type'] == 'OSS4SG'][f'{milestone}_achieved'].mean()
            
            oss_rates.append(oss_achieved * 100)
            oss4sg_rates.append(oss4sg_achieved * 100)
            
            # Chi-square test
            contingency = pd.crosstab(
                results_df['project_type'],
                results_df[f'{milestone}_achieved']
            )
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            p_values.append(p_value)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(milestone_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, oss_rates, width, label='OSS', 
                      color=self.colors['OSS'], alpha=0.8)
        bars2 = ax.bar(x + width/2, oss4sg_rates, width, label='OSS4SG',
                      color=self.colors['OSS4SG'], alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Add significance stars
        for i, p_value in enumerate(p_values):
            if p_value < 0.05:
                sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                y_pos = max(oss_rates[i], oss4sg_rates[i]) + 5
                ax.text(i, y_pos, sig_text, ha='center', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Milestones', fontsize=12)
        ax.set_ylabel('Achievement Rate (%)', fontsize=12)
        ax.set_title('Milestone Achievement Rates: OSS vs OSS4SG', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(milestone_labels, rotation=0, ha='center')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        # Save figure
        achievement_file = self.viz_dir / 'milestone_achievement_rates.png'
        plt.savefig(achievement_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {achievement_file}")
    
    def create_radar_plots(self, results_df):
        """Create radar (spider) plots for achievement rates and median timing for 5 working milestones"""
        import numpy as np
        import matplotlib.pyplot as plt

        # Only the 5 working milestones
        milestones = [
            'first_accepted', 'sustained_participation', 'returning_contributor',
            'cross_boundary', 'failure_recovery'
        ]
        labels = ['First Accepted', 'Sustained', 'Returning', 'Cross-Boundary', 'Failure Recovery']

        # Achievement rates
        oss_rates = []
        oss4sg_rates = []
        oss_medians = []
        oss4sg_medians = []

        for m in milestones:
            oss_rates.append(results_df[results_df['project_type'] == 'OSS'][f'{m}_achieved'].mean() * 100)
            oss4sg_rates.append(results_df[results_df['project_type'] == 'OSS4SG'][f'{m}_achieved'].mean() * 100)

            # Timing medians (use 1-based weeks for presentation)
            oss_t = results_df[(results_df['project_type'] == 'OSS') & (results_df[f'{m}_achieved'] == True)][f'{m}_week'].dropna()
            oss4_t = results_df[(results_df['project_type'] == 'OSS4SG') & (results_df[f'{m}_achieved'] == True)][f'{m}_week'].dropna()
            if len(oss_t) > 0:
                oss_t = oss_t + 1
            if len(oss4_t) > 0:
                oss4_t = oss4_t + 1
            # Use cleaned medians for robustness
            oss_t_clean = self.remove_outliers(oss_t) if len(oss_t) > 0 else oss_t
            oss4_t_clean = self.remove_outliers(oss4_t) if len(oss4_t) > 0 else oss4_t
            oss_medians.append(float(oss_t_clean.median()) if len(oss_t_clean) > 0 else np.nan)
            oss4sg_medians.append(float(oss4_t_clean.median()) if len(oss4_t_clean) > 0 else np.nan)

        def _radar(ax, values1, values2, categories, title, color1, color2, rlim=None):
            N = len(categories)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            values1 = np.array(values1)
            values2 = np.array(values2)
            # close the polygon
            angles = np.concatenate((angles, [angles[0]]))
            v1 = np.concatenate((values1, [values1[0]]))
            v2 = np.concatenate((values2, [values2[0]]))

            ax.plot(angles, v1, color=color1, linewidth=2, label='OSS')
            ax.fill(angles, v1, color=color1, alpha=0.25)
            ax.plot(angles, v2, color=color2, linewidth=2, label='OSS4SG')
            ax.fill(angles, v2, color=color2, alpha=0.25)
            ax.set_thetagrids(angles[:-1] * 180/np.pi, categories)
            if rlim is not None:
                ax.set_rlim(rlim)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # Radar for achievement rates (0-100)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)
        _radar(ax, oss_rates, oss4sg_rates, labels, 'Milestone Achievement Rates (%)', self.colors['OSS'], self.colors['OSS4SG'], rlim=(0, 100))
        plt.tight_layout()
        out1 = self.viz_dir / 'milestone_achievement_radar.png'
        plt.savefig(out1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out1}")

        # Radar for median weeks (auto-scale)
        all_vals = [v for v in oss_medians + oss4sg_medians if not np.isnan(v)]
        rmax = max(all_vals) if all_vals else 10
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, polar=True)
        _radar(ax, oss_medians, oss4sg_medians, labels, 'Median Weeks to Milestone (Cleaned)', self.colors['OSS'], self.colors['OSS4SG'], rlim=(0, rmax * 1.1))
        plt.tight_layout()
        out2 = self.viz_dir / 'milestone_timing_radar.png'
        plt.savefig(out2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out2}")
    
    def generate_summary_statistics(self, results_df):
        """Generate summary statistics table"""
        
        milestones = [
            'first_accepted', 'sustained_participation', 'returning_contributor',
            'trusted_reviewer', 'cross_boundary', 'community_helper', 'failure_recovery'
        ]
        
        summary_data = []
        
        for milestone in milestones:
            # Achievement rates
            oss_achieved = results_df[results_df['project_type'] == 'OSS'][f'{milestone}_achieved']
            oss4sg_achieved = results_df[results_df['project_type'] == 'OSS4SG'][f'{milestone}_achieved']
            
            # Timing for achievers (with and without outliers)
            oss_timing = results_df[
                (results_df['project_type'] == 'OSS') & 
                (results_df[f'{milestone}_achieved'] == True)
            ][f'{milestone}_week'].dropna()
            
            oss4sg_timing = results_df[
                (results_df['project_type'] == 'OSS4SG') & 
                (results_df[f'{milestone}_achieved'] == True)
            ][f'{milestone}_week'].dropna()
            
            # REPORTING OFFSET: convert week indices (0-based) to 1-based for presentation
            if len(oss_timing) > 0:
                oss_timing = oss_timing + 1
            if len(oss4sg_timing) > 0:
                oss4sg_timing = oss4sg_timing + 1

            # Remove outliers for robust statistics
            oss_timing_clean = self.remove_outliers(oss_timing) if len(oss_timing) > 0 else oss_timing
            oss4sg_timing_clean = self.remove_outliers(oss4sg_timing) if len(oss4sg_timing) > 0 else oss4sg_timing
            
            summary_data.append({
                'Milestone': milestone.replace('_', ' ').title(),
                'OSS_Rate_%': f"{oss_achieved.mean()*100:.1f}",
                'OSS4SG_Rate_%': f"{oss4sg_achieved.mean()*100:.1f}",
                'OSS_Median_Weeks': f"{oss_timing.median():.1f}" if len(oss_timing) > 0 else "N/A",
                'OSS4SG_Median_Weeks': f"{oss4sg_timing.median():.1f}" if len(oss4sg_timing) > 0 else "N/A",
                'OSS_Median_Clean': f"{oss_timing_clean.median():.1f}" if len(oss_timing_clean) > 0 else "N/A",
                'OSS4SG_Median_Clean': f"{oss4sg_timing_clean.median():.1f}" if len(oss4sg_timing_clean) > 0 else "N/A",
                'OSS_N': len(oss_achieved),
                'OSS4SG_N': len(oss4sg_achieved)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = self.output_dir / 'milestone_summary_statistics.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary statistics to {summary_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("MILESTONE ACHIEVEMENT SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        return summary_df

def main():
    """Main execution function"""
    
    # Set base path
    base_path = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"
    
    print("="*80)
    print("MILESTONE DETECTION AND ANALYSIS PIPELINE")
    print("="*80)
    
    # Initialize analyzer
    analyzer = MilestoneAnalyzer(base_path)
    
    # Step 1: Detect milestones
    print("\n📍 STEP 1: Detecting milestones for all contributors...")
    results_df = analyzer.process_all_contributors()
    
    # Step 2: Generate summary statistics
    print("\n📊 STEP 2: Generating summary statistics...")
    summary_df = analyzer.generate_summary_statistics(results_df)
    
    # Step 3: Create visualizations
    print("\n📈 STEP 3: Creating visualizations...")
    print("  Creating timing visualizations (with and without outliers)...")
    analyzer.create_timing_visualizations(results_df)
    
    print("  Creating achievement rate comparison...")
    analyzer.create_achievement_rate_comparison(results_df)
    
    print("\n" + "="*80)
    print("✅ MILESTONE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {analyzer.output_dir}")
    print(f"📊 Visualizations saved to: {analyzer.viz_dir}")
    print("\nGenerated files:")
    print("  - milestone_detection_results.csv")
    print("  - milestone_summary_statistics.csv")
    print("  - milestone_timing_boxplots_with_outliers.png")
    print("  - milestone_timing_boxplots_no_outliers.png")
    print("  - milestone_timing_violins_with_outliers.png")
    print("  - milestone_timing_violins_no_outliers.png")
    print("  - milestone_achievement_rates.png")

if __name__ == "__main__":
    main()