import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ContributionIndexTimeSeries:
    """
    Creates weekly and monthly time series of contribution indices for each contributor.
    Based on the paper's methodology but adapted for temporal analysis.
    """
    
    def __init__(self, timeline_dir):
        """
        Initialize with directory containing timeline CSV files.
        
        Args:
            timeline_dir: Path to directory with timeline_{project}_{email}.csv files
        """
        self.timeline_dir = Path(timeline_dir)
        self.weights = {
            'commits': 0.25,
            'prs_merged': 0.20,
            'comments': 0.15,
            'issues_opened': 0.15,
            'active_days': 0.15,
            'duration': 0.10
        }
        
    def load_timeline(self, filepath):
        """Load and parse a single timeline CSV file, filtering to pre-core events only."""
        df = pd.read_csv(filepath)
        
        # CRITICAL FIX: Filter to only pre-core events (newcomer transition period)
        if 'is_pre_core' in df.columns:
            pre_core_df = df[df['is_pre_core'] == True].copy()
            if len(pre_core_df) == 0:
                # Return empty DataFrame if no pre-core events
                return pd.DataFrame(columns=df.columns)
            df = pre_core_df
        else:
            print(f"Warning: {filepath.name} missing is_pre_core column - using all events")
        
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        # Parse event_data JSON
        df['event_data'] = df['event_data'].apply(lambda x: json.loads(x) if pd.notna(x) else {})
        return df
    
    def extract_weekly_metrics(self, timeline_df):
        """
        Extract metrics for each week from a contributor's timeline.
        
        Returns DataFrame with columns:
        - week_num: Week number from first activity
        - commits: Number of commits
        - prs_merged: Number of merged PRs
        - comments: Total comments (issues + PRs)
        - issues_opened: Number of issues created
        - active_days: Number of unique days with activity (0-7)
        """
        metrics = []
        
        # Get unique weeks
        weeks = timeline_df['event_week'].unique()
        min_week = int(timeline_df['event_week'].min())
        max_week = int(timeline_df['event_week'].max())
        
        # Process each week from first to last activity
        for week_num in range(min_week, max_week + 1):
            week_data = timeline_df[timeline_df['event_week'] == week_num]
            
            # Count commits
            commits = len(week_data[week_data['event_type'] == 'commit'])
            
            # Count merged PRs
            pr_data = week_data[week_data['event_type'] == 'pull_request']
            prs_merged = 0
            for _, pr in pr_data.iterrows():
                try:
                    if isinstance(pr['event_data'], dict) and pr['event_data'].get('merged', False):
                        prs_merged += 1
                except Exception as e:
                    # Skip problematic PR events
                    continue
            
            # Count comments (from issues and PRs)
            comments = 0
            contributor_username = timeline_df['username'].iloc[0] if len(timeline_df) > 0 else None
            
            for _, event in week_data.iterrows():
                try:
                    if event['event_type'] in ['issue', 'pull_request']:
                        # Count comments in the event
                        event_comments = event['event_data'].get('comments', {})
                        if isinstance(event_comments, dict):
                            comment_nodes = event_comments.get('nodes', [])
                            # Only count comments by this contributor
                            for comment in comment_nodes:
                                if isinstance(comment, dict):
                                    author = comment.get('author', {})
                                    if isinstance(author, dict) and author.get('login') == contributor_username:
                                        comments += 1
                        
                        # Also count reviews as comments for PRs
                        if event['event_type'] == 'pull_request':
                            reviews = event['event_data'].get('reviews', {})
                            if isinstance(reviews, dict):
                                review_nodes = reviews.get('nodes', [])
                                for review in review_nodes:
                                    if isinstance(review, dict):
                                        author = review.get('author', {})
                                        if isinstance(author, dict) and author.get('login') == contributor_username:
                                            comments += 1
                except Exception as e:
                    # Skip problematic events and continue processing
                    continue
            
            # Count issues opened
            issue_data = week_data[week_data['event_type'] == 'issue']
            issues_opened = len(issue_data)
            
            # Count active days in the week
            if len(week_data) > 0:
                # Get unique days within this week
                unique_days = week_data['event_timestamp'].dt.date.nunique()
                active_days = min(unique_days, 7)  # Cap at 7 days per week
            else:
                active_days = 0
            
            metrics.append({
                'week_num': week_num,
                'commits': commits,
                'prs_merged': prs_merged,
                'comments': comments,
                'issues_opened': issues_opened,
                'active_days': active_days
            })
        
        return pd.DataFrame(metrics)
    
    def calculate_duration_metrics(self, weekly_metrics_df, method='rolling_4week'):
        """
        Calculate duration/consistency metric for each week.
        
        Methods:
        - 'rolling_4week': Active weeks in last 4 weeks / 4
        - 'rolling_8week': Active weeks in last 8 weeks / 8
        - 'cumulative_ratio': Active weeks so far / total weeks so far
        - 'binary': 1 if active this week, 0 otherwise
        - 'weeks_since_start': Normalized cumulative weeks since first activity
        """
        duration_values = []
        
        for idx, week in weekly_metrics_df.iterrows():
            week_num = week['week_num']
            
            # Check if week is active (has any activity)
            is_active = (week['commits'] + week['prs_merged'] + 
                        week['comments'] + week['issues_opened']) > 0
            
            if method == 'rolling_4week':
                # Look at last 4 weeks including current
                window_start = max(0, idx - 3)
                window_data = weekly_metrics_df.iloc[window_start:idx+1]
                active_weeks = ((window_data['commits'] + window_data['prs_merged'] + 
                               window_data['comments'] + window_data['issues_opened']) > 0).sum()
                duration = active_weeks / 4.0
                
            elif method == 'rolling_8week':
                # Look at last 8 weeks including current
                window_start = max(0, idx - 7)
                window_data = weekly_metrics_df.iloc[window_start:idx+1]
                active_weeks = ((window_data['commits'] + window_data['prs_merged'] + 
                               window_data['comments'] + window_data['issues_opened']) > 0).sum()
                duration = active_weeks / 8.0
                
            elif method == 'cumulative_ratio':
                # Ratio of active weeks to total weeks so far
                past_data = weekly_metrics_df.iloc[:idx+1]
                active_weeks = ((past_data['commits'] + past_data['prs_merged'] + 
                               past_data['comments'] + past_data['issues_opened']) > 0).sum()
                total_weeks = idx + 1
                duration = active_weeks / total_weeks
                
            elif method == 'binary':
                # Simple binary: 1 if active, 0 if not
                duration = 1.0 if is_active else 0.0
                
            elif method == 'weeks_since_start':
                # Cumulative weeks since first activity (will normalize later)
                duration = idx + 1  # Week position since start
                
            else:
                raise ValueError(f"Unknown duration method: {method}")
            
            duration_values.append(duration)
        
        weekly_metrics_df['duration'] = duration_values
        
        # Normalize weeks_since_start if that method was used
        if method == 'weeks_since_start' and len(duration_values) > 0:
            max_duration = max(duration_values)
            if max_duration > 0:
                weekly_metrics_df['duration'] = weekly_metrics_df['duration'] / max_duration
        
        return weekly_metrics_df
    
    def calculate_contribution_index(self, metrics_df):
        """
        Calculate contribution index for each week using the paper's formula.
        
        Formula:
        CI = 0.25*commits + 0.20*prs_merged + 0.15*comments + 
             0.15*issues_opened + 0.15*normalized_active_days + 0.10*duration
        """
        # Normalize active_days (0-7 days per week)
        metrics_df['active_days_norm'] = metrics_df['active_days'] / 7.0
        
        # Calculate raw contribution index
        metrics_df['contribution_index'] = (
            self.weights['commits'] * metrics_df['commits'] +
            self.weights['prs_merged'] * metrics_df['prs_merged'] +
            self.weights['comments'] * metrics_df['comments'] +
            self.weights['issues_opened'] * metrics_df['issues_opened'] +
            self.weights['active_days'] * metrics_df['active_days_norm'] +
            self.weights['duration'] * metrics_df['duration']
        )
        
        return metrics_df
    
    def aggregate_to_monthly(self, weekly_metrics_df):
        """
        Aggregate weekly metrics to monthly.
        Months are defined as 4-week periods.
        """
        monthly_metrics = []
        
        # Group weeks into months (4 weeks per month)
        weekly_metrics_df['month_num'] = weekly_metrics_df['week_num'] // 4
        
        for month in weekly_metrics_df['month_num'].unique():
            month_data = weekly_metrics_df[weekly_metrics_df['month_num'] == month]
            
            # Sum activity metrics
            monthly_metric = {
                'month_num': month,
                'commits': month_data['commits'].sum(),
                'prs_merged': month_data['prs_merged'].sum(),
                'comments': month_data['comments'].sum(),
                'issues_opened': month_data['issues_opened'].sum(),
                'active_days': month_data['active_days'].sum(),  # Total active days in month
                'duration': month_data['duration'].mean()  # Average duration metric
            }
            
            monthly_metrics.append(monthly_metric)
        
        monthly_df = pd.DataFrame(monthly_metrics)
        
        # Normalize active_days for monthly (max 28 days per 4-week month)
        monthly_df['active_days_norm'] = monthly_df['active_days'] / 28.0
        
        # Calculate monthly contribution index
        monthly_df['contribution_index'] = (
            self.weights['commits'] * monthly_df['commits'] +
            self.weights['prs_merged'] * monthly_df['prs_merged'] +
            self.weights['comments'] * monthly_df['comments'] +
            self.weights['issues_opened'] * monthly_df['issues_opened'] +
            self.weights['active_days'] * monthly_df['active_days_norm'] +
            self.weights['duration'] * monthly_df['duration']
        )
        
        return monthly_df
    
    def process_all_contributors(self, duration_method='rolling_4week', verbose=True):
        """
        Process all contributor timeline files and create time series.
        
        Returns:
            - weekly_timeseries: Dict of {contributor_id: weekly_df}
            - monthly_timeseries: Dict of {contributor_id: monthly_df}
            - metadata: Summary statistics
        """
        weekly_timeseries = {}
        monthly_timeseries = {}
        metadata = {
            'total_contributors': 0,
            'duration_method': duration_method,
            'min_weeks': float('inf'),
            'max_weeks': 0,
            'avg_weeks': 0,
            'contributors_processed': []
        }
        
        # Get all timeline files
        timeline_files = list(self.timeline_dir.glob('timeline_*.csv'))
        
        # Optional: limit files for testing (uncomment to test with smaller dataset)
        # if len(timeline_files) > 100:
        #     if verbose:
        #         print(f"Large dataset detected ({len(timeline_files)} files). Processing first 10 for testing...")
        #     timeline_files = timeline_files[:10]
        
        if verbose:
            print(f"Found {len(timeline_files)} timeline files to process")
            print(f"Using duration method: {duration_method}")
            print("-" * 50)
        
        all_weeks = []
        
        for i, filepath in enumerate(timeline_files):
            if verbose and i % 100 == 0:
                print(f"Processing contributor {i+1}/{len(timeline_files)}")
            
            try:
                # Extract contributor ID from filename
                filename = filepath.name
                # Format: timeline_{project}_{email}.csv
                parts = filename.replace('timeline_', '').replace('.csv', '')
                contributor_id = parts
                
                # Load timeline
                timeline_df = self.load_timeline(filepath)
                
                if len(timeline_df) == 0:
                    continue
                
                # Extract weekly metrics
                weekly_metrics = self.extract_weekly_metrics(timeline_df)
                
                # Calculate duration metric
                weekly_metrics = self.calculate_duration_metrics(
                    weekly_metrics, method=duration_method
                )
                
                # Calculate contribution index
                weekly_metrics = self.calculate_contribution_index(weekly_metrics)
                
                # Add contributor info
                weekly_metrics['contributor_id'] = contributor_id
                weekly_metrics['project'] = timeline_df['project_name'].iloc[0]
                weekly_metrics['project_type'] = timeline_df['project_type'].iloc[0]
                
                # Create monthly aggregation
                monthly_metrics = self.aggregate_to_monthly(weekly_metrics)
                monthly_metrics['contributor_id'] = contributor_id
                monthly_metrics['project'] = timeline_df['project_name'].iloc[0]
                monthly_metrics['project_type'] = timeline_df['project_type'].iloc[0]
                
                # Store in dictionaries
                weekly_timeseries[contributor_id] = weekly_metrics
                monthly_timeseries[contributor_id] = monthly_metrics
                
                # Update metadata
                num_weeks = len(weekly_metrics)
                all_weeks.append(num_weeks)
                metadata['min_weeks'] = min(metadata['min_weeks'], num_weeks)
                metadata['max_weeks'] = max(metadata['max_weeks'], num_weeks)
                metadata['contributors_processed'].append(contributor_id)
                
            except Exception as e:
                if verbose:
                    print(f"Error processing {filepath.name}: {e}")
                continue
        
        metadata['total_contributors'] = len(weekly_timeseries)
        metadata['avg_weeks'] = np.mean(all_weeks) if all_weeks else 0
        
        if verbose:
            print("\n" + "="*50)
            print("PROCESSING COMPLETE")
            print(f"Total contributors processed: {metadata['total_contributors']}")
            print(f"Week range: {metadata['min_weeks']} to {metadata['max_weeks']} weeks")
            print(f"Average weeks per contributor: {metadata['avg_weeks']:.1f}")
        
        return weekly_timeseries, monthly_timeseries, metadata
    
    def save_timeseries(self, weekly_timeseries, monthly_timeseries, output_dir):
        """
        Save time series data to CSV files.
        
        Creates:
        - all_weekly_timeseries.csv: All weekly data combined
        - all_monthly_timeseries.csv: All monthly data combined
        - individual/: Directory with individual contributor files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Combine all weekly data
        all_weekly = []
        for contributor_id, df in weekly_timeseries.items():
            all_weekly.append(df)
        
        if all_weekly:
            combined_weekly = pd.concat(all_weekly, ignore_index=True)
            combined_weekly.to_csv(output_path / 'all_weekly_timeseries.csv', index=False)
            print(f"Saved combined weekly data: {len(combined_weekly)} rows")
        
        # Combine all monthly data
        all_monthly = []
        for contributor_id, df in monthly_timeseries.items():
            all_monthly.append(df)
        
        if all_monthly:
            combined_monthly = pd.concat(all_monthly, ignore_index=True)
            combined_monthly.to_csv(output_path / 'all_monthly_timeseries.csv', index=False)
            print(f"Saved combined monthly data: {len(combined_monthly)} rows")
        
        # Save individual files if needed
        individual_dir = output_path / 'individual'
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        for contributor_id, df in weekly_timeseries.items():
            safe_id = contributor_id.replace('/', '_').replace('@', '_at_')
            df.to_csv(individual_dir / f'weekly_{safe_id}.csv', index=False)
        
        print(f"Saved {len(weekly_timeseries)} individual contributor files")
    
    def create_pivot_timeseries(self, timeseries_dict, value_column='contribution_index'):
        """
        Create pivot table format where rows are contributors and columns are time periods.
        This format is ideal for DTW clustering.
        
        Returns DataFrame with:
        - Rows: Contributors
        - Columns: Week/Month numbers
        - Values: Contribution index (or specified metric)
        """
        pivot_data = []
        
        for contributor_id, df in timeseries_dict.items():
            # Get time column name (week_num or month_num)
            time_col = 'week_num' if 'week_num' in df.columns else 'month_num'
            
            # Create a series for this contributor
            series_dict = {'contributor_id': contributor_id}
            series_dict['project'] = df['project'].iloc[0] if 'project' in df.columns else None
            series_dict['project_type'] = df['project_type'].iloc[0] if 'project_type' in df.columns else None
            
            # Add values for each time period
            for _, row in df.iterrows():
                time_key = f't_{int(row[time_col])}'
                series_dict[time_key] = row[value_column]
            
            pivot_data.append(series_dict)
        
        # Create DataFrame
        pivot_df = pd.DataFrame(pivot_data)
        
        # Sort columns to ensure time periods are in order
        time_cols = [col for col in pivot_df.columns if col.startswith('t_')]
        time_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        # Reorder columns
        meta_cols = [col for col in pivot_df.columns if not col.startswith('t_')]
        pivot_df = pivot_df[meta_cols + time_cols]
        
        # Fill NaN with 0 (for contributors who didn't have activity in certain periods)
        pivot_df = pivot_df.fillna(0)
        
        return pivot_df


# Example usage and testing code
def main():
    """
    Main function to demonstrate usage.
    """
    # Configuration
    base_path = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"
    timeline_dir = f"{base_path}/RQ2_newcomer_treatment_patterns_test2/step2_timelines/from_cache_timelines/"
    output_dir = f"{base_path}/RQ3_engagement_patterns/step1/results/"
    
    # Initialize processor
    processor = ContributionIndexTimeSeries(timeline_dir)
    
    # Test different duration methods
    duration_methods = [
        'rolling_4week',     # Recommended: 4-week rolling window
        'rolling_8week',     # Alternative: 8-week rolling window
        'cumulative_ratio',  # Consistency ratio
        'binary',           # Simple active/inactive
        # 'weeks_since_start'  # Not recommended due to high variance
    ]
    
    print("="*60)
    print("CONTRIBUTION INDEX TIME SERIES GENERATOR")
    print("="*60)
    
    for method in duration_methods[:1]:  # Process first method for now
        print(f"\nProcessing with duration method: {method}")
        print("-"*40)
        
        # Process all contributors
        weekly, monthly, metadata = processor.process_all_contributors(
            duration_method=method,
            verbose=True
        )
        
        # Save results
        method_output_dir = f"{output_dir}/{method}/"
        processor.save_timeseries(weekly, monthly, method_output_dir)
        
        # Create pivot format for DTW clustering
        weekly_pivot = processor.create_pivot_timeseries(weekly)
        monthly_pivot = processor.create_pivot_timeseries(monthly)
        
        # Save pivot tables
        weekly_pivot.to_csv(f"{method_output_dir}/weekly_pivot_for_dtw.csv", index=False)
        monthly_pivot.to_csv(f"{method_output_dir}/monthly_pivot_for_dtw.csv", index=False)
        
        print(f"\nPivot table shapes:")
        print(f"Weekly: {weekly_pivot.shape} (contributors x time periods)")
        print(f"Monthly: {monthly_pivot.shape} (contributors x time periods)")
        
        # Show sample of the data
        if len(weekly) > 0:
            sample_contributor = list(weekly.keys())[0]
            sample_df = weekly[sample_contributor]
            print(f"\nSample weekly data for contributor: {sample_contributor}")
            print(sample_df[['week_num', 'commits', 'prs_merged', 'comments', 
                           'issues_opened', 'active_days', 'duration', 
                           'contribution_index']].head(10))
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("Next steps: Use the pivot tables for DTW clustering")
    print("="*60)


if __name__ == "__main__":
    main()