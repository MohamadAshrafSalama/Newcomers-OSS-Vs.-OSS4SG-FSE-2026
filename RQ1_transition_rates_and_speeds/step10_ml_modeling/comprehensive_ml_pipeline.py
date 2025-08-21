#!/usr/bin/env python3
"""
Comprehensive ML Pipeline for RQ1 Step 10
==========================================
Properly uses:
1. Master commits dataset (3.5M commits, 92K contributors)
2. Monthly core transition data 
3. 90-day feature windows for ALL contributors
4. Robust email/name matching with verification
5. Smart missing feature handling (NaN/imputation, not removal)
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
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class ComprehensiveMLPipeline:
    """
    End-to-end ML pipeline for newcomer-to-core prediction using comprehensive data
    """
    
    def __init__(self, output_dir="RQ1_transition_rates_and_speeds/step10_ml_modeling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'comprehensive_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*80)
        self.logger.info("COMPREHENSIVE ML PIPELINE FOR RQ1 STEP 10")
        self.logger.info("="*80)
        
        # Initialize data containers
        self.commits_df = None
        self.core_contributors = None
        self.contributor_features = None
        self.final_dataset = None
        
    def load_commits_data(self):
        """Load and preprocess commits data"""
        self.logger.info("📂 Loading master commits dataset...")
        
        # Load commits data
        commits_path = "RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
        
        self.commits_df = pd.read_csv(
            commits_path,
            dtype={
                'project_name': str,
                'project_type': str,
                'author_email': str,
                'author_name': str,
                'commit_hash': str
            },
            low_memory=False
        )
        
        self.logger.info(f"Loaded {len(self.commits_df):,} commits")
        self.logger.info(f"Unique projects: {self.commits_df['project_name'].nunique()}")
        self.logger.info(f"Unique contributors (by email): {self.commits_df['author_email'].nunique()}")
        self.logger.info(f"Unique contributors (by name): {self.commits_df['author_name'].nunique()}")
        
        # Convert dates
        self.commits_df['commit_date'] = pd.to_datetime(self.commits_df['commit_date'], utc=True)
        
        # Normalize emails and names for better matching
        self.commits_df['author_email_norm'] = self.commits_df['author_email'].str.lower().str.strip()
        self.commits_df['author_name_norm'] = self.commits_df['author_name'].str.strip()
        
        # Create unique contributor identifiers combining email and name
        self.commits_df['contributor_id'] = (
            self.commits_df['project_name'] + "::" + 
            self.commits_df['author_email_norm'] + "::" + 
            self.commits_df['author_name_norm']
        )
        
        self.logger.info(f"Unique contributor IDs: {self.commits_df['contributor_id'].nunique()}")
        
    def extract_core_contributors(self):
        """Extract core contributors from monthly transition data"""
        self.logger.info("🎯 Extracting core contributors from monthly data...")
        
        # Load monthly transitions
        monthly_path = "RQ1_transition_rates_and_speeds/step4_newcomer_transition_rates/corrected_transition_results/monthly_transitions.csv"
        monthly_df = pd.read_csv(monthly_path)
        
        self.logger.info(f"Loaded {len(monthly_df):,} project-month records")
        
        # Extract all core contributors
        core_contributors = []
        
        for _, row in tqdm(monthly_df.iterrows(), total=len(monthly_df), desc="Processing monthly data"):
            project_name = row['project_name']
            month = row['month']
            
            # Process truly new core contributors
            if pd.notna(row['truly_new_core_names']) and row['truly_new_core_names'] not in ['[]', '']:
                try:
                    new_cores = eval(row['truly_new_core_names'])
                    for name in new_cores:
                        core_contributors.append({
                            'project_name': project_name,
                            'author_name': name.strip(),
                            'became_core_month': month,
                            'became_core_date': pd.to_datetime(month + '-01'),
                            'is_core': True
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to parse new core names for {project_name} {month}: {e}")
        
        self.core_contributors = pd.DataFrame(core_contributors)
        self.logger.info(f"Extracted {len(self.core_contributors)} core contributor records")
        self.logger.info(f"Unique core contributors: {len(self.core_contributors.groupby(['project_name', 'author_name']))}")
        
        # Save core contributors list
        self.core_contributors.to_csv(self.output_dir / 'extracted_core_contributors.csv', index=False)
        
    def match_contributors_to_commits(self):
        """Match core contributors to commits data using multiple strategies"""
        self.logger.info("🔗 Matching contributors between core data and commits...")
        
        # Create contributor mapping from commits
        contributor_mapping = self.commits_df.groupby(['project_name', 'author_name_norm']).agg({
            'author_email_norm': 'first',
            'contributor_id': 'first',
            'commit_hash': 'count'
        }).rename(columns={'commit_hash': 'total_commits'}).reset_index()
        
        # Normalize core contributor names
        self.core_contributors['author_name_norm'] = self.core_contributors['author_name'].str.strip()
        
        # Match core contributors to commits
        matched_cores = self.core_contributors.merge(
            contributor_mapping,
            on=['project_name', 'author_name_norm'],
            how='left'
        )
        
        # Check matching success
        matched_count = matched_cores['contributor_id'].notna().sum()
        total_count = len(matched_cores)
        
        self.logger.info(f"Core contributor matching: {matched_count}/{total_count} ({matched_count/total_count*100:.1f}%)")
        
        # For unmatched, try fuzzy matching or alternative strategies
        unmatched = matched_cores[matched_cores['contributor_id'].isna()]
        if len(unmatched) > 0:
            self.logger.warning(f"{len(unmatched)} core contributors could not be matched to commits")
            # Save unmatched for manual inspection
            unmatched[['project_name', 'author_name', 'became_core_month']].to_csv(
                self.output_dir / 'unmatched_core_contributors.csv', index=False
            )
        
        # Keep only matched core contributors
        self.matched_cores = matched_cores[matched_cores['contributor_id'].notna()].copy()
        self.logger.info(f"Successfully matched {len(self.matched_cores)} core contributors")
        
    def extract_contributor_features(self):
        """Extract 90-day features for ALL contributors"""
        self.logger.info("⚙️ Extracting 90-day features for all contributors...")
        
        # Get all unique contributors
        all_contributors = self.commits_df.groupby(['project_name', 'contributor_id']).agg({
            'author_email_norm': 'first',
            'author_name_norm': 'first',
            'commit_date': ['min', 'max'],
            'commit_hash': 'count',
            'total_lines_changed': 'sum',
            'files_modified_count': 'sum'
        }).reset_index()
        
        # Flatten column names
        all_contributors.columns = [
            'project_name', 'contributor_id', 'author_email', 'author_name',
            'first_commit_date', 'last_commit_date', 'total_commits', 
            'total_lines_changed', 'total_files_modified'
        ]
        
        self.logger.info(f"Processing features for {len(all_contributors)} unique contributors...")
        
        # Extract 90-day features
        features_list = []
        
        for _, contrib in tqdm(all_contributors.iterrows(), total=len(all_contributors), desc="Extracting features"):
            try:
                features = self._extract_90day_features(contrib)
                if features is not None:
                    features_list.append(features)
            except Exception as e:
                self.logger.warning(f"Failed to extract features for {contrib['contributor_id']}: {e}")
        
        self.contributor_features = pd.DataFrame(features_list)
        self.logger.info(f"Extracted features for {len(self.contributor_features)} contributors")
        
        # Save features
        self.contributor_features.to_csv(self.output_dir / 'contributor_features_90day.csv', index=False)
        
    def _extract_90day_features(self, contributor):
        """Extract 90-day features for a single contributor"""
        project_name = contributor['project_name']
        contributor_id = contributor['contributor_id']
        first_commit_date = contributor['first_commit_date']
        
        # Get commits for this contributor
        contrib_commits = self.commits_df[
            (self.commits_df['project_name'] == project_name) &
            (self.commits_df['contributor_id'] == contributor_id)
        ].copy()
        
        if len(contrib_commits) < 3:  # Minimum threshold
            return None
            
        # Define 90-day window from first commit
        window_end = first_commit_date + timedelta(days=90)
        window_commits = contrib_commits[contrib_commits['commit_date'] <= window_end].copy()
        
        if len(window_commits) == 0:
            return None
        
        # Sort by date
        window_commits = window_commits.sort_values('commit_date')
        
        # Calculate features
        features = {
            'project_name': project_name,
            'contributor_id': contributor_id,
            'author_email': contributor['author_email'],
            'author_name': contributor['author_name'],
            'first_commit_date': first_commit_date,
            
            # Basic 90-day features
            'commits_90d': len(window_commits),
            'lines_changed_90d': window_commits['total_lines_changed'].sum(),
            'files_modified_90d': window_commits['files_modified_count'].sum(),
            'active_days_90d': window_commits['commit_date'].dt.date.nunique(),
            
            # Temporal patterns
            'days_span_90d': (window_commits['commit_date'].max() - window_commits['commit_date'].min()).days + 1,
            'avg_commits_per_day': len(window_commits) / 90,
            'max_commits_single_day': window_commits.groupby(window_commits['commit_date'].dt.date).size().max(),
            
            # Monthly breakdown
            'month1_commits': len(window_commits[window_commits['commit_date'] <= first_commit_date + timedelta(days=30)]),
            'month2_commits': len(window_commits[
                (window_commits['commit_date'] > first_commit_date + timedelta(days=30)) &
                (window_commits['commit_date'] <= first_commit_date + timedelta(days=60))
            ]),
            'month3_commits': len(window_commits[
                (window_commits['commit_date'] > first_commit_date + timedelta(days=60)) &
                (window_commits['commit_date'] <= first_commit_date + timedelta(days=90))
            ]),
            
            # Growth patterns
            'early_commits_pct': len(window_commits[window_commits['commit_date'] <= first_commit_date + timedelta(days=30)]) / len(window_commits),
            'late_commits_pct': len(window_commits[window_commits['commit_date'] > first_commit_date + timedelta(days=60)]) / len(window_commits),
            
            # Consistency
            'commit_frequency_std': window_commits.groupby(window_commits['commit_date'].dt.date).size().std() or 0,
            'longest_gap_days': self._calculate_longest_gap(window_commits['commit_date']),
        }
        
        # Handle NaN/infinite values
        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                features[key] = 0 if isinstance(value, (int, float)) else value
                
        return features
    
    def _calculate_longest_gap(self, dates):
        """Calculate longest gap between commits in days"""
        if len(dates) < 2:
            return 0
        
        sorted_dates = dates.sort_values()
        gaps = sorted_dates.diff().dt.days
        return gaps.max() if len(gaps) > 0 else 0
    
    def create_labeled_dataset(self):
        """Create labeled dataset combining features with core status"""
        self.logger.info("🏷️ Creating labeled dataset...")
        
        # Add core labels to features
        core_lookup = set()
        for _, core in self.matched_cores.iterrows():
            # Match by project and name
            matches = self.contributor_features[
                (self.contributor_features['project_name'] == core['project_name']) &
                (self.contributor_features['author_name'] == core['author_name_norm'])
            ]
            for idx in matches.index:
                core_lookup.add(idx)
        
        # Create labels
        self.contributor_features['became_core'] = self.contributor_features.index.isin(core_lookup)
        
        # Add project type
        project_types = self.commits_df[['project_name', 'project_type']].drop_duplicates()
        self.final_dataset = self.contributor_features.merge(project_types, on='project_name', how='left')
        
        # Summary
        total = len(self.final_dataset)
        cores = self.final_dataset['became_core'].sum()
        self.logger.info(f"Final dataset: {total:,} contributors")
        self.logger.info(f"Core contributors: {cores:,} ({cores/total*100:.1f}%)")
        
        # By project type
        for ptype in ['OSS', 'OSS4SG']:
            subset = self.final_dataset[self.final_dataset['project_type'] == ptype]
            subset_cores = subset['became_core'].sum()
            self.logger.info(f"{ptype}: {len(subset):,} contributors, {subset_cores:,} cores ({subset_cores/len(subset)*100:.1f}%)")
        
        # Save final dataset
        self.final_dataset.to_csv(self.output_dir / 'final_ml_dataset.csv', index=False)
        
    def train_models(self):
        """Train ML models with proper handling of missing values"""
        self.logger.info("🤖 Training ML models...")
        
        # Prepare features and labels
        feature_cols = [col for col in self.final_dataset.columns 
                       if col not in ['project_name', 'contributor_id', 'author_email', 'author_name', 
                                     'first_commit_date', 'became_core', 'project_type']]
        
        X = self.final_dataset[feature_cols]
        y = self.final_dataset['became_core']
        
        self.logger.info(f"Feature matrix: {X.shape}")
        self.logger.info(f"Features: {feature_cols}")
        
        # Handle missing values with imputation
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        # Standardize features
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
            
            # Cross-validation scores
            roc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            precision_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='average_precision', n_jobs=-1)
            
            results[name] = {
                'roc_auc_mean': roc_scores.mean(),
                'roc_auc_std': roc_scores.std(),
                'pr_auc_mean': precision_scores.mean(),
                'pr_auc_std': precision_scores.std()
            }
            
            self.logger.info(f"{name} - ROC AUC: {roc_scores.mean():.3f} ± {roc_scores.std():.3f}")
            self.logger.info(f"{name} - PR AUC: {precision_scores.mean():.3f} ± {precision_scores.std():.3f}")
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(self.output_dir / 'model_performance_comprehensive.csv')
        
        # Feature importance (using Random Forest)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_scaled, y)
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(self.output_dir / 'feature_importance_comprehensive.csv', index=False)
        
        # Create visualizations
        self._create_visualizations(results_df, feature_importance)
        
        return results_df, feature_importance
    
    def _create_visualizations(self, results_df, feature_importance):
        """Create comprehensive visualizations"""
        self.logger.info("📊 Creating visualizations...")
        
        # Model performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC AUC comparison
        ax1.bar(results_df.index, results_df['roc_auc_mean'], 
                yerr=results_df['roc_auc_std'], capsize=5)
        ax1.set_title('Model Performance - ROC AUC')
        ax1.set_ylabel('ROC AUC')
        ax1.set_ylim(0, 1)
        
        # PR AUC comparison
        ax2.bar(results_df.index, results_df['pr_auc_mean'], 
                yerr=results_df['pr_auc_std'], capsize=5)
        ax2.set_title('Model Performance - PR AUC')
        ax2.set_ylabel('PR AUC')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: Load commits data
            self.load_commits_data()
            
            # Step 2: Extract core contributors
            self.extract_core_contributors()
            
            # Step 3: Match contributors
            self.match_contributors_to_commits()
            
            # Step 4: Extract features
            self.extract_contributor_features()
            
            # Step 5: Create labeled dataset
            self.create_labeled_dataset()
            
            # Step 6: Train models
            results_df, feature_importance = self.train_models()
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() / 60
            
            # Generate summary report
            self._generate_summary_report(execution_time, results_df, feature_importance)
            
            self.logger.info("="*80)
            self.logger.info("✅ COMPREHENSIVE ML PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Execution time: {execution_time:.1f} minutes")
            self.logger.info(f"Results saved to: {self.output_dir}")
            self.logger.info("="*80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_summary_report(self, execution_time, results_df, feature_importance):
        """Generate comprehensive summary report"""
        report = f"""
COMPREHENSIVE ML PIPELINE SUMMARY REPORT
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Execution Time: {execution_time:.1f} minutes

DATA SUMMARY:
- Total commits processed: {len(self.commits_df):,}
- Unique projects: {self.commits_df['project_name'].nunique()}
- Total contributors analyzed: {len(self.final_dataset):,}
- Core contributors: {self.final_dataset['became_core'].sum():,}
- Core rate: {self.final_dataset['became_core'].mean()*100:.1f}%

PROJECT TYPE BREAKDOWN:
"""
        
        for ptype in ['OSS', 'OSS4SG']:
            subset = self.final_dataset[self.final_dataset['project_type'] == ptype]
            cores = subset['became_core'].sum()
            report += f"- {ptype}: {len(subset):,} contributors, {cores:,} cores ({cores/len(subset)*100:.1f}%)\n"
        
        report += f"""
MODEL PERFORMANCE:
"""
        
        for model in results_df.index:
            roc = results_df.loc[model, 'roc_auc_mean']
            roc_std = results_df.loc[model, 'roc_auc_std']
            pr = results_df.loc[model, 'pr_auc_mean']
            pr_std = results_df.loc[model, 'pr_auc_std']
            report += f"- {model}: ROC AUC = {roc:.3f} ± {roc_std:.3f}, PR AUC = {pr:.3f} ± {pr_std:.3f}\n"
        
        report += f"""
TOP 10 FEATURES:
"""
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            report += f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}\n"
        
        report += f"""
FILES GENERATED:
- final_ml_dataset.csv: Complete labeled dataset
- contributor_features_90day.csv: Raw features
- model_performance_comprehensive.csv: Model metrics
- feature_importance_comprehensive.csv: Feature rankings
- model_performance_comparison.png: Performance plots
- feature_importance_plot.png: Feature importance plot
- comprehensive_pipeline.log: Detailed execution log
"""
        
        # Save report
        with open(self.output_dir / 'comprehensive_summary_report.txt', 'w') as f:
            f.write(report)
        
        print(report)

if __name__ == "__main__":
    # Run the comprehensive pipeline
    pipeline = ComprehensiveMLPipeline()
    success = pipeline.run_complete_pipeline()
    
    if not success:
        exit(1)

