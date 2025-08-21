#!/usr/bin/env python3
"""
SIMPLE & COMPREHENSIVE ML Pipeline for RQ1 Step 10
================================================
Simple approach:
1. Extract ALL core contributors from monthly data (by name)
2. Use ALL contributors from commits dataset  
3. Simple filters: >1 commit AND >=3 months activity
4. Extract features with NaN handling (don't drop contributors)
5. Train models on the full dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class SimpleMLPipeline:
    """
    Simple, effective ML pipeline that doesn't lose data due to complex matching
    """
    
    def __init__(self, output_dir="RQ1_transition_rates_and_speeds/step10_ml_modeling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'simple_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*80)
        self.logger.info("SIMPLE COMPREHENSIVE ML PIPELINE FOR RQ1 STEP 10")
        self.logger.info("="*80)
        
    def extract_all_core_contributors(self):
        """Extract ALL core contributors from monthly data"""
        self.logger.info("🎯 Extracting ALL core contributors from monthly data...")
        
        # Load monthly transitions
        monthly_path = "RQ1_transition_rates_and_speeds/step4_newcomer_transition_rates/corrected_transition_results/monthly_transitions.csv"
        monthly_df = pd.read_csv(monthly_path)
        
        self.logger.info(f"Loaded {len(monthly_df):,} project-month records")
        
        # Extract all core names
        all_core_names = set()
        core_records = []
        
        for _, row in tqdm(monthly_df.iterrows(), total=len(monthly_df), desc="Processing monthly data"):
            project_name = row['project_name']
            project_type = row['project_type']
            month = row['month']
            
            if pd.notna(row['truly_new_core_names']) and row['truly_new_core_names'] not in ['[]', '']:
                try:
                    new_cores = eval(row['truly_new_core_names'])
                    for name in new_cores:
                        clean_name = name.strip()
                        all_core_names.add(clean_name)
                        core_records.append({
                            'project_name': project_name,
                            'project_type': project_type,
                            'author_name': clean_name,
                            'became_core_month': month,
                            'became_core_date': pd.to_datetime(month + '-01')
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to parse: {row['truly_new_core_names']}")
        
        self.core_contributors_df = pd.DataFrame(core_records)
        
        self.logger.info(f"Total core contributor records: {len(self.core_contributors_df):,}")
        self.logger.info(f"Unique core names: {len(all_core_names):,}")
        
        # Save for inspection
        self.core_contributors_df.to_csv(self.output_dir / 'all_core_contributors.csv', index=False)
        
        return all_core_names
    
    def load_and_process_commits(self, core_names):
        """Load commits and identify core contributors"""
        self.logger.info("📂 Loading master commits dataset...")
        
        # Load commits data in chunks to handle memory
        commits_path = "RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
        
        chunk_size = 500000
        chunks = []
        
        for chunk in tqdm(pd.read_csv(commits_path, chunksize=chunk_size, low_memory=False), 
                         desc="Loading commits"):
            # Convert dates
            chunk['commit_date'] = pd.to_datetime(chunk['commit_date'], utc=True)
            chunks.append(chunk)
        
        self.commits_df = pd.concat(chunks, ignore_index=True)
        del chunks  # Free memory
        
        self.logger.info(f"Loaded {len(self.commits_df):,} commits")
        self.logger.info(f"Unique projects: {self.commits_df['project_name'].nunique()}")
        self.logger.info(f"Unique author names: {self.commits_df['author_name'].nunique()}")
        
        # Check how many core names are in commits
        commits_names = set(self.commits_df['author_name'].unique())
        matched_cores = core_names.intersection(commits_names)
        
        self.logger.info(f"Core names found in commits: {len(matched_cores):,}/{len(core_names):,} ({len(matched_cores)/len(core_names)*100:.1f}%)")
        
        if len(matched_cores) / len(core_names) < 0.9:
            self.logger.warning(f"Only {len(matched_cores)/len(core_names)*100:.1f}% of core names matched!")
        else:
            self.logger.info("✅ Good matching rate!")
        
        # Mark core contributors in commits
        self.commits_df['is_core'] = self.commits_df['author_name'].isin(matched_cores)
        
        self.logger.info(f"Commits by core contributors: {self.commits_df['is_core'].sum():,}/{len(self.commits_df):,}")
        
        return matched_cores
    
    def create_contributor_dataset(self):
        """Create dataset of all contributors with basic filters"""
        self.logger.info("👥 Creating comprehensive contributor dataset...")
        
        # Aggregate by project and author
        contributor_stats = self.commits_df.groupby(['project_name', 'author_name']).agg({
            'project_type': 'first',
            'author_email': 'first', 
            'commit_date': ['min', 'max', 'count'],
            'total_lines_changed': 'sum',
            'files_modified_count': 'sum',
            'is_core': 'any'  # If any commit is by core contributor
        }).reset_index()
        
        # Flatten column names
        contributor_stats.columns = [
            'project_name', 'author_name', 'project_type', 'author_email',
            'first_commit_date', 'last_commit_date', 'total_commits',
            'total_lines_changed', 'total_files_modified', 'became_core'
        ]
        
        # Calculate activity duration
        contributor_stats['activity_duration_days'] = (
            contributor_stats['last_commit_date'] - contributor_stats['first_commit_date']
        ).dt.days + 1
        
        self.logger.info(f"Total contributors: {len(contributor_stats):,}")
        
        # Apply filters: >1 commit AND >=3 months activity
        before_filter = len(contributor_stats)
        filtered_contributors = contributor_stats[
            (contributor_stats['total_commits'] > 1) &
            (contributor_stats['activity_duration_days'] >= 90)  # 3 months
        ].copy()
        
        self.logger.info(f"After filtering (>1 commit, >=3 months): {len(filtered_contributors):,}")
        self.logger.info(f"Filtered out: {before_filter - len(filtered_contributors):,} contributors")
        
        # Summary stats
        cores = filtered_contributors['became_core'].sum()
        self.logger.info(f"Core contributors: {cores:,} ({cores/len(filtered_contributors)*100:.1f}%)")
        
        # By project type
        for ptype in ['OSS', 'OSS4SG']:
            subset = filtered_contributors[filtered_contributors['project_type'] == ptype]
            subset_cores = subset['became_core'].sum()
            if len(subset) > 0:
                self.logger.info(f"{ptype}: {len(subset):,} contributors, {subset_cores:,} cores ({subset_cores/len(subset)*100:.1f}%)")
        
        self.contributor_dataset = filtered_contributors
        self.contributor_dataset.to_csv(self.output_dir / 'filtered_contributors.csv', index=False)
        
        return filtered_contributors
    
    def extract_90day_features(self):
        """Extract 90-day features for all contributors"""
        self.logger.info("⚙️ Extracting 90-day features for all contributors...")
        
        features_list = []
        
        for _, contributor in tqdm(self.contributor_dataset.iterrows(), 
                                 total=len(self.contributor_dataset), 
                                 desc="Extracting features"):
            
            # Get this contributor's commits
            contrib_commits = self.commits_df[
                (self.commits_df['project_name'] == contributor['project_name']) &
                (self.commits_df['author_name'] == contributor['author_name'])
            ].copy()
            
            if len(contrib_commits) == 0:
                continue
                
            # Sort by date
            contrib_commits = contrib_commits.sort_values('commit_date')
            first_commit_date = contrib_commits['commit_date'].iloc[0]
            
            # Define 90-day window
            window_end = first_commit_date + timedelta(days=90)
            window_commits = contrib_commits[contrib_commits['commit_date'] <= window_end].copy()
            
            # Extract features (with NaN handling)
            features = self._calculate_features(contributor, window_commits, first_commit_date)
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        self.logger.info(f"Extracted features for {len(self.features_df):,} contributors")
        
        # Save features
        self.features_df.to_csv(self.output_dir / 'features_90day_comprehensive.csv', index=False)
        
        return self.features_df
    
    def _calculate_features(self, contributor, window_commits, first_commit_date):
        """Calculate features for one contributor (robust to missing data)"""
        
        features = {
            # Basic info
            'project_name': contributor['project_name'],
            'project_type': contributor['project_type'],
            'author_name': contributor['author_name'],
            'author_email': contributor['author_email'],
            'became_core': contributor['became_core'],
            'first_commit_date': first_commit_date,
            
            # Basic 90-day features
            'commits_90d': len(window_commits),
            'lines_changed_90d': window_commits['total_lines_changed'].sum() if not window_commits.empty else 0,
            'files_modified_90d': window_commits['files_modified_count'].sum() if not window_commits.empty else 0,
        }
        
        if len(window_commits) > 0:
            # Temporal features
            features['active_days_90d'] = window_commits['commit_date'].dt.date.nunique()
            features['days_span_90d'] = (window_commits['commit_date'].max() - window_commits['commit_date'].min()).days + 1
            features['avg_commits_per_day'] = len(window_commits) / 90
            
            # Daily activity
            daily_commits = window_commits.groupby(window_commits['commit_date'].dt.date).size()
            features['max_commits_single_day'] = daily_commits.max()
            features['avg_commits_per_active_day'] = daily_commits.mean()
            features['commit_frequency_std'] = daily_commits.std() if len(daily_commits) > 1 else 0
            
            # Monthly breakdown
            month1_end = first_commit_date + timedelta(days=30)
            month2_end = first_commit_date + timedelta(days=60)
            
            features['month1_commits'] = len(window_commits[window_commits['commit_date'] <= month1_end])
            features['month2_commits'] = len(window_commits[
                (window_commits['commit_date'] > month1_end) & 
                (window_commits['commit_date'] <= month2_end)
            ])
            features['month3_commits'] = len(window_commits[
                window_commits['commit_date'] > month2_end
            ])
            
            # Growth patterns
            total_commits = len(window_commits)
            features['early_commits_pct'] = features['month1_commits'] / total_commits if total_commits > 0 else 0
            features['late_commits_pct'] = features['month3_commits'] / total_commits if total_commits > 0 else 0
            
            # Gaps analysis
            if len(window_commits) > 1:
                sorted_dates = window_commits['commit_date'].sort_values()
                gaps = sorted_dates.diff().dt.days.dropna()
                features['avg_gap_days'] = gaps.mean() if len(gaps) > 0 else 0
                features['max_gap_days'] = gaps.max() if len(gaps) > 0 else 0
                features['gap_consistency'] = 1 / (gaps.std() + 1) if len(gaps) > 1 else 1
            else:
                features['avg_gap_days'] = 0
                features['max_gap_days'] = 0
                features['gap_consistency'] = 1
                
        else:
            # Default values for empty window
            for key in ['active_days_90d', 'days_span_90d', 'avg_commits_per_day', 
                       'max_commits_single_day', 'avg_commits_per_active_day', 'commit_frequency_std',
                       'month1_commits', 'month2_commits', 'month3_commits',
                       'early_commits_pct', 'late_commits_pct', 
                       'avg_gap_days', 'max_gap_days', 'gap_consistency']:
                features[key] = 0
        
        # Handle any remaining NaN/inf values
        for key, value in features.items():
            if pd.isna(value) or (isinstance(value, float) and np.isinf(value)):
                features[key] = 0 if key not in ['project_name', 'project_type', 'author_name', 'author_email', 'first_commit_date'] else value
        
        return features
    
    def train_models(self):
        """Train ML models on the comprehensive dataset"""
        self.logger.info("🤖 Training ML models on comprehensive dataset...")
        
        # Prepare features and labels
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['project_name', 'project_type', 'author_name', 'author_email', 
                                     'first_commit_date', 'became_core']]
        
        X = self.features_df[feature_cols]
        y = self.features_df['became_core']
        
        self.logger.info(f"Feature matrix: {X.shape}")
        self.logger.info(f"Positive class rate: {y.mean()*100:.1f}%")
        
        # Handle missing values with imputation
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Define models
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                roc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                pr_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='average_precision', n_jobs=-1)
                
                results[name] = {
                    'roc_auc_mean': roc_scores.mean(),
                    'roc_auc_std': roc_scores.std(),
                    'pr_auc_mean': pr_scores.mean(),
                    'pr_auc_std': pr_scores.std()
                }
                
                self.logger.info(f"{name} - ROC AUC: {roc_scores.mean():.3f} ± {roc_scores.std():.3f}")
                self.logger.info(f"{name} - PR AUC: {pr_scores.mean():.3f} ± {pr_scores.std():.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                results[name] = {'roc_auc_mean': 0, 'roc_auc_std': 0, 'pr_auc_mean': 0, 'pr_auc_std': 0}
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(self.output_dir / 'model_results_comprehensive.csv')
        
        # Feature importance
        self._analyze_feature_importance(X_imputed, y, feature_cols)
        
        return results_df
    
    def _analyze_feature_importance(self, X, y, feature_cols):
        """Analyze feature importance using Random Forest"""
        self.logger.info("📊 Analyzing feature importance...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(self.output_dir / 'feature_importance_comprehensive.csv', index=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return importance_df
    
    def generate_summary_report(self, results_df):
        """Generate comprehensive summary report"""
        
        # Calculate statistics
        total_contributors = len(self.features_df)
        total_cores = self.features_df['became_core'].sum()
        core_rate = total_cores / total_contributors * 100
        
        # By project type
        oss_contributors = len(self.features_df[self.features_df['project_type'] == 'OSS'])
        oss_cores = self.features_df[self.features_df['project_type'] == 'OSS']['became_core'].sum()
        
        oss4sg_contributors = len(self.features_df[self.features_df['project_type'] == 'OSS4SG'])
        oss4sg_cores = self.features_df[self.features_df['project_type'] == 'OSS4SG']['became_core'].sum()
        
        report = f"""
SIMPLE COMPREHENSIVE ML PIPELINE SUMMARY
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY:
- Total contributors analyzed: {total_contributors:,}
- Core contributors: {total_cores:,} ({core_rate:.1f}%)
- Projects: {self.features_df['project_name'].nunique()}

PROJECT TYPE BREAKDOWN:
- OSS: {oss_contributors:,} contributors, {oss_cores:,} cores ({oss_cores/oss_contributors*100:.1f}%)
- OSS4SG: {oss4sg_contributors:,} contributors, {oss4sg_cores:,} cores ({oss4sg_cores/oss4sg_contributors*100:.1f}%)

MODEL PERFORMANCE:
"""
        
        for model in results_df.index:
            roc = results_df.loc[model, 'roc_auc_mean']
            roc_std = results_df.loc[model, 'roc_auc_std']
            pr = results_df.loc[model, 'pr_auc_mean']
            pr_std = results_df.loc[model, 'pr_auc_std']
            report += f"- {model}: ROC AUC = {roc:.3f} ± {roc_std:.3f}, PR AUC = {pr:.3f} ± {pr_std:.3f}\n"
        
        report += f"""
FILES GENERATED:
- features_90day_comprehensive.csv: All contributor features
- filtered_contributors.csv: Basic contributor stats
- all_core_contributors.csv: Core contributor records
- model_results_comprehensive.csv: Model performance metrics
- feature_importance_comprehensive.csv: Feature rankings
- feature_importance_comprehensive.png: Feature importance plot
"""
        
        # Save and print report
        with open(self.output_dir / 'comprehensive_summary.txt', 'w') as f:
            f.write(report)
        
        print(report)
        
        return report
    
    def run_pipeline(self):
        """Run the complete simple pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: Extract core contributors
            core_names = self.extract_all_core_contributors()
            
            # Step 2: Load commits and match
            matched_cores = self.load_and_process_commits(core_names)
            
            # Step 3: Create contributor dataset
            self.create_contributor_dataset()
            
            # Step 4: Extract features
            self.extract_90day_features()
            
            # Step 5: Train models
            results_df = self.train_models()
            
            # Step 6: Generate report
            self.generate_summary_report(results_df)
            
            # Execution time
            execution_time = (datetime.now() - start_time).total_seconds() / 60
            
            self.logger.info("="*80)
            self.logger.info("✅ SIMPLE COMPREHENSIVE PIPELINE COMPLETED!")
            self.logger.info(f"Execution time: {execution_time:.1f} minutes")
            self.logger.info("="*80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    pipeline = SimpleMLPipeline()
    success = pipeline.run_pipeline()
    
    if not success:
        exit(1)
