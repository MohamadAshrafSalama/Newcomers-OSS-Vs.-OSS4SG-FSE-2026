#!/usr/bin/env python3
"""
Complete ML Pipeline for Predicting Core Contributors from 90-Day Activity
===========================================================================
Research Question: What behaviors in the first 90 days predict becoming a core contributor?

Based on:
- Zhou & Mockus (2012): "What Make Long Term Contributors"
- Begel & Zimmermann (2014): "Analyze This! 145 Questions for Data Scientists"

Author: FSE 2026 Research Team
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr

# ML imports
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
except ImportError:
    print("Warning: XGBoost not installed. Some models will be skipped.")
    XGBClassifier = None

warnings.filterwarnings('ignore')

# Configure paths
BASE_DIR = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
TRANSITIONS_PATH = BASE_DIR / "RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv"
WEEKLY_ACTIVITY_PATH = BASE_DIR / "RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv"
OUTPUT_DIR = BASE_DIR / "RQ1_transition_rates_and_speeds/step11/python"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class NinetyDayFeatureExtractor:
    """
    Extract interpretable features from first 90 days (13 weeks) of contributor activity
    """
    
    def __init__(self, window_weeks: int = 13):
        self.window_weeks = window_weeks
        self.feature_names = []
        
    def extract_features(self, contributor_weekly_data: pd.DataFrame, 
                        project_stats: Dict) -> Optional[Dict]:
        """
        Extract ~20 interpretable features from contributor's first 90 days
        """
        # Filter to first 90 days (prefer weeks since contributor's first commit)
        if 'weeks_since_first_commit' in contributor_weekly_data.columns:
            early_mask = contributor_weekly_data['weeks_since_first_commit'] < self.window_weeks
        else:
            # fallback to absolute week_number (1-indexed)
            early_mask = contributor_weekly_data['week_number'] <= self.window_weeks
        early_data = contributor_weekly_data[early_mask].copy()
        
        if len(early_data) == 0:
            return None
            
        features = {}
        
        # === CONSISTENCY FEATURES (How regular is their participation?) ===
        active_weeks = (early_data['commits_this_week'] > 0)
        features['active_weeks_count'] = active_weeks.sum()
        features['activity_rate'] = features['active_weeks_count'] / self.window_weeks
        
        # Longest inactive streak
        inactive = (~active_weeks).astype(int)
        streak_groups = (inactive != inactive.shift()).cumsum()
        streaks = inactive.groupby(streak_groups).sum()
        features['longest_inactive_streak'] = streaks[streaks > 0].max() if any(streaks > 0) else 0
        
        # Activity regularity (coefficient of variation inverted)
        if features['active_weeks_count'] > 1:
            active_commits = early_data.loc[active_weeks, 'commits_this_week']
            cv = active_commits.std() / (active_commits.mean() + 1e-6)
            features['activity_regularity'] = 1 / (1 + cv)  # Higher = more regular
        else:
            features['activity_regularity'] = 0
            
        # === MOMENTUM FEATURES (Are they accelerating or fading?) ===
        # Split into 3 months
        month_boundaries = [0, 4, 8, 13]
        monthly_commits = []
        for i in range(3):
            start, end = month_boundaries[i], month_boundaries[i+1]
            month_data = early_data.iloc[start:end] if len(early_data) > start else pd.DataFrame()
            monthly_commits.append(month_data['commits_this_week'].sum() if len(month_data) > 0 else 0)
            
        features['month1_commits'] = monthly_commits[0]
        features['month2_commits'] = monthly_commits[1]
        features['month3_commits'] = monthly_commits[2]
        
        # Acceleration metrics
        features['acceleration_rate'] = (monthly_commits[2] - monthly_commits[0]) / (monthly_commits[0] + 1)
        features['sustained_growth'] = int(
            monthly_commits[1] > monthly_commits[0] and 
            monthly_commits[2] > monthly_commits[1]
        )
        
        # Linear trend
        if features['active_weeks_count'] >= 3:
            weeks = np.arange(len(early_data))
            commits = early_data['commits_this_week'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, commits)
            features['trend_slope'] = slope
            features['trend_r2'] = r_value ** 2
        else:
            features['trend_slope'] = 0
            features['trend_r2'] = 0
            
        # === INTENSITY FEATURES (How much do they contribute?) ===
        features['total_commits_90d'] = early_data['commits_this_week'].sum()
        features['peak_week_commits'] = early_data['commits_this_week'].max()
        
        # Use cumulative lines changed diff to get weekly changes
        early_data['lines_changed_weekly'] = early_data.groupby('contributor_email')['cumulative_lines_changed'].diff()
        early_data.loc[early_data.index[0], 'lines_changed_weekly'] = early_data.iloc[0]['cumulative_lines_changed']
        features['total_lines_changed_90d'] = early_data['lines_changed_weekly'].sum()
        
        # Average intensity when active
        if features['active_weeks_count'] > 0:
            features['avg_commits_when_active'] = features['total_commits_90d'] / features['active_weeks_count']
            features['avg_lines_per_commit'] = features['total_lines_changed_90d'] / max(features['total_commits_90d'], 1)
        else:
            features['avg_commits_when_active'] = 0
            features['avg_lines_per_commit'] = 0
            
        # Burst ratio
        avg_weekly = features['total_commits_90d'] / self.window_weeks
        features['burst_ratio'] = features['peak_week_commits'] / (avg_weekly + 1)
        
        # === INTEGRATION FEATURES (How well are they integrating?) ===
        # Time to first commit
        first_active = early_data[early_data['commits_this_week'] > 0]['week_number'].min()
        features['weeks_to_first_commit'] = first_active if pd.notna(first_active) else self.window_weeks
        
        # Relative productivity (normalized by project)
        features['relative_productivity'] = features['total_commits_90d'] / (project_stats.get('median_90d_commits', 10) + 1)
        
        # Final rank/contribution percentage
        if 'contribution_percentage' in early_data.columns:
            features['final_contribution_pct'] = early_data.iloc[-1]['contribution_percentage']
            features['max_contribution_pct'] = early_data['contribution_percentage'].max()
        else:
            features['final_contribution_pct'] = 0
            features['max_contribution_pct'] = 0
            
        # Early vs late activity
        first_half = early_data.iloc[:7]['commits_this_week'].sum()
        second_half = early_data.iloc[7:]['commits_this_week'].sum()
        features['early_vs_late_ratio'] = first_half / (second_half + 1)
        
        return features


class CorePredictionPipeline:
    """
    Complete ML pipeline for core contributor prediction
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.feature_extractor = NinetyDayFeatureExtractor()
        self.results = {}
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and filter data for analysis
        """
        print("=" * 80)
        print("LOADING AND PREPARING DATA")
        print("=" * 80)
        
        # Load datasets
        transitions = pd.read_csv(TRANSITIONS_PATH)
        weekly_activity = pd.read_csv(WEEKLY_ACTIVITY_PATH)
        
        # Filter for meaningful contributors
        # Require: ≥3 commits and ≥13 weeks of observation
        valid_contributors = transitions[
            (transitions['total_commits'] >= 3) & 
            (transitions['total_weeks_observed'] >= 13)
        ].copy()
        
        print(f"Total contributors: {len(transitions):,}")
        print(f"After filtering (≥3 commits, ≥13 weeks): {len(valid_contributors):,}")
        print(f"Core contributors: {valid_contributors['became_core'].sum():,} "
              f"({100*valid_contributors['became_core'].mean():.1f}%)")
        
        # Print by project type
        for ptype in ['OSS', 'OSS4SG']:
            subset = valid_contributors[valid_contributors['project_type'] == ptype]
            print(f"  {ptype}: {len(subset):,} contributors, "
                  f"{subset['became_core'].sum():,} core "
                  f"({100*subset['became_core'].mean():.1f}%)")
        
        return valid_contributors, weekly_activity
    
    def extract_all_features(self, transitions: pd.DataFrame, 
                            weekly_activity: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for all valid contributors
        """
        print("\n" + "=" * 80)
        print("EXTRACTING 90-DAY FEATURES")
        print("=" * 80)
        
        all_features = []
        
        # Calculate project-level statistics for normalization
        project_stats = {}
        for project in transitions['project_name'].unique():
            project_contributors = transitions[transitions['project_name'] == project]
            project_stats[project] = {
                'median_90d_commits': project_contributors['total_commits'].median(),
                'mean_90d_commits': project_contributors['total_commits'].mean(),
            }
        
        # Extract features for each contributor
        for idx, contributor in transitions.iterrows():
            if idx % 1000 == 0:
                print(f"  Processing contributor {idx+1}/{len(transitions)}...")
                
            # Get this contributor's weekly activity
            contributor_weekly = weekly_activity[
                (weekly_activity['project_name'] == contributor['project_name']) &
                (weekly_activity['contributor_email'] == contributor['contributor_email'])
            ].sort_values('week_number')
            
            # Extract features
            features = self.feature_extractor.extract_features(
                contributor_weekly, 
                project_stats.get(contributor['project_name'], {})
            )
            
            if features:
                # Add metadata
                features['contributor_email'] = contributor['contributor_email']
                features['project_name'] = contributor['project_name']
                features['project_type'] = contributor['project_type']
                features['became_core'] = int(contributor['became_core'])
                all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        print(f"\nExtracted features for {len(features_df):,} contributors")
        print(f"Number of features: {len([c for c in features_df.columns if c not in ['contributor_email', 'project_name', 'project_type', 'became_core']])}")
        
        # Save features
        features_df.to_csv(self.output_dir / "extracted_features_90day.csv", index=False)
        
        return features_df
    
    def analyze_feature_importance(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze feature importance using correlation and mutual information
        """
        print("\n" + "=" * 80)
        print("ANALYZING FEATURE IMPORTANCE")
        print("=" * 80)
        
        # Prepare feature matrix and labels
        feature_cols = [c for c in features_df.columns 
                       if c not in ['contributor_email', 'project_name', 'project_type', 'became_core']]
        X = features_df[feature_cols]
        y = features_df['became_core']
        
        results = []
        
        for col in feature_cols:
            # Spearman correlation
            corr, p_value = spearmanr(X[col].fillna(0), y)
            
            # Mutual information
            mi_score = mutual_info_classif(
                X[[col]].fillna(0), y, random_state=42
            )[0]
            
            # Mann-Whitney U test (difference between groups)
            core_values = X[y == 1][col].dropna()
            non_core_values = X[y == 0][col].dropna()
            if len(core_values) > 0 and len(non_core_values) > 0:
                mw_stat, mw_p = mannwhitneyu(core_values, non_core_values)
                # Effect size (rank-biserial correlation)
                n1, n2 = len(core_values), len(non_core_values)
                effect_size = 1 - (2*mw_stat) / (n1 * n2)
            else:
                mw_p = 1.0
                effect_size = 0
            
            results.append({
                'feature': col,
                'correlation': corr,
                'corr_p_value': p_value,
                'mutual_info': mi_score,
                'mann_whitney_p': mw_p,
                'effect_size': effect_size,
                'abs_correlation': abs(corr),
                'mean_core': core_values.mean() if len(core_values) > 0 else 0,
                'mean_non_core': non_core_values.mean() if len(non_core_values) > 0 else 0,
            })
        
        importance_df = pd.DataFrame(results)
        importance_df = importance_df.sort_values('abs_correlation', ascending=False)
        
        # Save importance analysis
        importance_df.to_csv(self.output_dir / "feature_importance_analysis.csv", index=False)
        
        # Print top features
        print("\nTOP 10 MOST PREDICTIVE FEATURES:")
        print("-" * 80)
        for idx, row in importance_df.head(10).iterrows():
            direction = "↑" if row['correlation'] > 0 else "↓"
            print(f"{row['feature']:30s} | r={row['correlation']:+.3f} {direction} | "
                  f"MI={row['mutual_info']:.3f} | p={row['corr_p_value']:.3e}")
        
        return importance_df
    
    def train_models(self, features_df: pd.DataFrame) -> Dict:
        """
        Train multiple models with cross-validation
        """
        print("\n" + "=" * 80)
        print("TRAINING MODELS WITH 5-FOLD CROSS-VALIDATION")
        print("=" * 80)
        
        # Prepare data
        feature_cols = [c for c in features_df.columns 
                       if c not in ['contributor_email', 'project_name', 'project_type', 'became_core']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['became_core']
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGBClassifier is not None:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Scoring metrics
        scoring = {
            'roc_auc': 'roc_auc',
            'pr_auc': 'average_precision',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
        }
        
        cv_results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            scores = cross_validate(
                model, X_scaled, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Calculate mean and std for each metric
            results = {}
            for metric in scoring.keys():
                train_key = f'train_{metric}'
                test_key = f'test_{metric}'
                results[f'{metric}_train_mean'] = scores[train_key].mean()
                results[f'{metric}_train_std'] = scores[train_key].std()
                results[f'{metric}_test_mean'] = scores[test_key].mean()
                results[f'{metric}_test_std'] = scores[test_key].std()
            
            cv_results[name] = results
            
            # Print results
            print(f"  ROC-AUC: {results['roc_auc_test_mean']:.3f} ± {results['roc_auc_test_std']:.3f}")
            print(f"  PR-AUC:  {results['pr_auc_test_mean']:.3f} ± {results['pr_auc_test_std']:.3f}")
            print(f"  F1:      {results['f1_test_mean']:.3f} ± {results['f1_test_std']:.3f}")
        
        # Save results
        results_df = pd.DataFrame(cv_results).T
        results_df.to_csv(self.output_dir / "model_performance_results.csv")
        
        # Train final models on full data for feature importance
        final_models = {}
        for name, model in models.items():
            model.fit(X_scaled, y)
            final_models[name] = model
        
        self.results['cv_results'] = cv_results
        self.results['models'] = final_models
        self.results['scaler'] = scaler
        self.results['feature_names'] = feature_cols
        
        return cv_results
    
    def analyze_by_project_type(self, features_df: pd.DataFrame) -> Dict:
        """
        Comprehensive analysis: Combined, OSS-only, and OSS4SG-only models
        """
        print("\n" + "=" * 80)
        print("COMPARATIVE ANALYSIS: COMBINED vs OSS vs OSS4SG")
        print("=" * 80)
        
        comprehensive_results = {}
        feature_importance_by_type = {}
        
        # Define subsets
        subsets = {
            'Combined': features_df,
            'OSS': features_df[features_df['project_type'] == 'OSS'],
            'OSS4SG': features_df[features_df['project_type'] == 'OSS4SG']
        }
        
        # Feature columns
        feature_cols = [c for c in features_df.columns 
                       if c not in ['contributor_email', 'project_name', 'project_type', 'became_core']]
        
        for subset_name, subset_df in subsets.items():
            print(f"\n{'='*60}")
            print(f"ANALYZING: {subset_name}")
            print(f"{'='*60}")
            
            X = subset_df[feature_cols].fillna(0)
            y = subset_df['became_core']
            
            # Basic stats
            print(f"Samples: {len(subset_df):,}")
            print(f"Core: {y.sum():,} ({100*y.mean():.1f}%)")
            
            # === 1. FEATURE IMPORTANCE ANALYSIS ===
            print(f"\nFeature Importance for {subset_name}:")
            importance_results = []
            
            for col in feature_cols:
                # Correlation
                corr, p_value = spearmanr(X[col], y)
                
                # Mutual information
                mi_score = mutual_info_classif(X[[col]], y, random_state=42)[0]
                
                # Effect size
                core_values = X[y == 1][col].dropna()
                non_core_values = X[y == 0][col].dropna()
                
                if len(core_values) > 0 and len(non_core_values) > 0:
                    mw_stat, mw_p = mannwhitneyu(core_values, non_core_values)
                    n1, n2 = len(core_values), len(non_core_values)
                    effect_size = 1 - (2*mw_stat) / (n1 * n2)
                else:
                    effect_size = 0
                    mw_p = 1.0
                
                importance_results.append({
                    'feature': col,
                    'correlation': corr,
                    'p_value': p_value,
                    'mutual_info': mi_score,
                    'effect_size': effect_size,
                    'abs_correlation': abs(corr),
                    'mean_core': core_values.mean() if len(core_values) > 0 else 0,
                    'mean_non_core': non_core_values.mean() if len(non_core_values) > 0 else 0,
                })
            
            importance_df = pd.DataFrame(importance_results).sort_values('abs_correlation', ascending=False)
            feature_importance_by_type[subset_name] = importance_df
            
            # Save feature importance
            importance_df.to_csv(self.output_dir / f"feature_importance_{subset_name.lower()}.csv", index=False)
            
            # Print top 5 features
            print(f"\nTop 5 Predictive Features for {subset_name}:")
            for idx, row in importance_df.head(5).iterrows():
                direction = "↑" if row['correlation'] > 0 else "↓"
                print(f"  {row['feature']:25s} | r={row['correlation']:+.3f} {direction} | ES={row['effect_size']:.2f}")
            
            # === 2. MODEL TRAINING ===
            print(f"\nTraining Models for {subset_name}:")

            # If the subset has fewer than 2 classes, skip training to avoid failures
            if y.nunique() < 2:
                print(f"  Skipping model training for {subset_name}: only one class present ({y.unique()})")
                comprehensive_results[subset_name] = {
                    'stats': {
                        'n_samples': len(subset_df),
                        'n_core': int(y.sum()),
                        'pct_core': 100 * float(y.mean()) if len(subset_df) > 0 else 0.0,
                    },
                    'models': {},
                    'top_features': importance_df.head(10).to_dict('records')
                }
                continue

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Define models
            models = {
                'LogisticReg': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
                'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                                      class_weight='balanced', random_state=42, n_jobs=-1),
            }

            # Add XGBoost if available
            if XGBClassifier is not None:
                models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                                 random_state=42, use_label_encoder=False, eval_metric='logloss')
            else:
                models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                                       learning_rate=0.1, random_state=42)

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model_results = {}

            for model_name, model in models.items():
                scores = cross_validate(
                    model, X_scaled, y,
                    cv=cv,
                    scoring=['roc_auc', 'average_precision', 'f1'],
                    n_jobs=-1,
                    error_score='raise'
                )

                model_results[model_name] = {
                    'roc_auc': scores['test_roc_auc'].mean(),
                    'roc_auc_std': scores['test_roc_auc'].std(),
                    'pr_auc': scores['test_average_precision'].mean(),
                    'pr_auc_std': scores['test_average_precision'].std(),
                    'f1': scores['test_f1'].mean(),
                    'f1_std': scores['test_f1'].std(),
                }

                print(f"  {model_name:12s}: ROC={model_results[model_name]['roc_auc']:.3f}, "
                      f"PR={model_results[model_name]['pr_auc']:.3f}, "
                      f"F1={model_results[model_name]['f1']:.3f}")

            comprehensive_results[subset_name] = {
                'stats': {
                    'n_samples': len(subset_df),
                    'n_core': int(y.sum()),
                    'pct_core': 100 * float(y.mean()),
                },
                'models': model_results,
                'top_features': importance_df.head(10).to_dict('records')
            }
        
        # === 3. COMPARATIVE ANALYSIS ===
        print("\n" + "=" * 80)
        print("COMPARATIVE FEATURE IMPORTANCE")
        print("=" * 80)
        
        # Find unique important features for each type
        oss_top10 = set(feature_importance_by_type['OSS'].head(10)['feature'])
        oss4sg_top10 = set(feature_importance_by_type['OSS4SG'].head(10)['feature'])
        combined_top10 = set(feature_importance_by_type['Combined'].head(10)['feature'])
        
        oss_unique = oss_top10 - oss4sg_top10
        oss4sg_unique = oss4sg_top10 - oss_top10
        shared = oss_top10 & oss4sg_top10
        
        print(f"\nFeatures unique to OSS (top 10): {oss_unique if oss_unique else 'None'}")
        print(f"Features unique to OSS4SG (top 10): {oss4sg_unique if oss4sg_unique else 'None'}")
        print(f"Shared important features: {len(shared)}")
        
        # Create comparison table
        comparison_data = []
        all_features = set(feature_cols)
        
        for feature in all_features:
            row = {'feature': feature}
            for subset_name in ['Combined', 'OSS', 'OSS4SG']:
                feat_data = feature_importance_by_type[subset_name]
                feat_row = feat_data[feat_data['feature'] == feature].iloc[0]
                row[f'{subset_name}_corr'] = feat_row['correlation']
                row[f'{subset_name}_rank'] = feat_data[feat_data['feature'] == feature].index[0] + 1
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['rank_diff_OSS_OSS4SG'] = comparison_df['OSS_rank'] - comparison_df['OSS4SG_rank']
        comparison_df = comparison_df.sort_values('Combined_rank')
        comparison_df.to_csv(self.output_dir / "feature_comparison_all_types.csv", index=False)
        
        # Save comprehensive results
        with open(self.output_dir / "comprehensive_type_analysis.json", 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        self.results['type_comparison'] = comprehensive_results
        self.results['feature_importance_by_type'] = feature_importance_by_type
        
        return comprehensive_results
    
    def create_comparative_visualizations(self, features_df: pd.DataFrame):
        """
        Create visualizations comparing OSS vs OSS4SG feature importance
        """
        if 'feature_importance_by_type' not in self.results:
            return
            
        print("\nCreating comparative visualizations...")
        
        fig = plt.figure(figsize=(18, 10))
        
        # Get feature importance for each type
        oss_importance = self.results['feature_importance_by_type']['OSS']
        oss4sg_importance = self.results['feature_importance_by_type']['OSS4SG']
        combined_importance = self.results['feature_importance_by_type']['Combined']
        
        # 1. Side-by-side feature importance comparison
        ax1 = plt.subplot(2, 3, 1)
        top_features = combined_importance.head(10)['feature'].tolist()
        
        oss_corrs = []
        oss4sg_corrs = []
        for feat in top_features:
            oss_corrs.append(oss_importance[oss_importance['feature'] == feat]['correlation'].iloc[0])
            oss4sg_corrs.append(oss4sg_importance[oss4sg_importance['feature'] == feat]['correlation'].iloc[0])
        
        x = np.arange(len(top_features))
        width = 0.35
        
        bars1 = ax1.barh(x - width/2, oss_corrs, width, label='OSS', color='lightblue')
        bars2 = ax1.barh(x + width/2, oss4sg_corrs, width, label='OSS4SG', color='lightgreen')
        
        ax1.set_yticks(x)
        ax1.set_yticklabels([f[:20] for f in top_features], fontsize=8)
        ax1.set_xlabel('Correlation with Becoming Core')
        ax1.set_title('A. Feature Importance: OSS vs OSS4SG', fontweight='bold')
        ax1.legend()
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # 2. Unique important features
        ax2 = plt.subplot(2, 3, 2)
        
        # Get top 15 for better comparison
        oss_top15 = set(oss_importance.head(15)['feature'])
        oss4sg_top15 = set(oss4sg_importance.head(15)['feature'])
        
        # Venn diagram data
        oss_only = len(oss_top15 - oss4sg_top15)
        oss4sg_only = len(oss4sg_top15 - oss_top15)
        shared = len(oss_top15 & oss4sg_top15)
        
        # Simple bar chart for Venn data
        categories = ['OSS\nUnique', 'Shared', 'OSS4SG\nUnique']
        values = [oss_only, shared, oss4sg_only]
        colors = ['lightblue', 'purple', 'lightgreen']
        
        bars = ax2.bar(categories, values, color=colors)
        ax2.set_ylabel('Number of Features')
        ax2.set_title('B. Top-15 Feature Overlap', fontweight='bold')
        ax2.set_ylim(0, max(values) + 2)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 3. Model performance comparison
        ax3 = plt.subplot(2, 3, 3)
        
        if 'type_comparison' in self.results:
            model_types = ['Combined', 'OSS', 'OSS4SG']
            model_colors = ['gray', 'lightblue', 'lightgreen']
            
            # Get Random Forest performance for each (fall back if missing)
            rf_roc = []
            rf_pr = []
            for mtype in model_types:
                models_dict = self.results['type_comparison'].get(mtype, {}).get('models', {})
                if 'RandomForest' in models_dict:
                    rf_roc.append(models_dict['RandomForest'].get('roc_auc', float('nan')))
                    rf_pr.append(models_dict['RandomForest'].get('pr_auc', float('nan')))
                elif len(models_dict) > 0:
                    # use the first available model's metrics
                    first_model = next(iter(models_dict.values()))
                    rf_roc.append(first_model.get('roc_auc', float('nan')))
                    rf_pr.append(first_model.get('pr_auc', float('nan')))
                else:
                    rf_roc.append(float('nan'))
                    rf_pr.append(float('nan'))
            
            x = np.arange(len(model_types))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, rf_roc, width, label='ROC-AUC', color='steelblue')
            bars2 = ax3.bar(x + width/2, rf_pr, width, label='PR-AUC', color='coral')
            
            ax3.set_ylabel('Score')
            ax3.set_title('C. Model Performance by Type', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(model_types)
            ax3.legend()
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Feature rank changes
        ax4 = plt.subplot(2, 3, 4)
        
        # Calculate rank changes
        rank_changes = []
        feature_names = []
        for feat in combined_importance.head(10)['feature']:
            oss_rank = oss_importance[oss_importance['feature'] == feat].index[0] + 1
            oss4sg_rank = oss4sg_importance[oss4sg_importance['feature'] == feat].index[0] + 1
            rank_change = oss_rank - oss4sg_rank  # Negative means better in OSS
            rank_changes.append(rank_change)
            feature_names.append(feat[:20])
        
        colors = ['lightblue' if x < 0 else 'lightgreen' for x in rank_changes]
        bars = ax4.barh(range(len(rank_changes)), rank_changes, color=colors)
        ax4.set_yticks(range(len(feature_names)))
        ax4.set_yticklabels(feature_names, fontsize=8)
        ax4.set_xlabel('← Better for OSS | Better for OSS4SG →')
        ax4.set_title('D. Feature Rank Difference', fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(True, alpha=0.3)
        
        # 5. Top unique features for each type
        ax5 = plt.subplot(2, 3, 5)
        
        # Get features that have very different correlations
        all_features = set(oss_importance['feature']) & set(oss4sg_importance['feature'])
        diff_data = []
        for feat in all_features:
            oss_corr = oss_importance[oss_importance['feature'] == feat]['correlation'].iloc[0]
            oss4sg_corr = oss4sg_importance[oss4sg_importance['feature'] == feat]['correlation'].iloc[0]
            diff = oss4sg_corr - oss_corr
            diff_data.append({'feature': feat, 'diff': diff, 'oss': oss_corr, 'oss4sg': oss4sg_corr})
        
        diff_df = pd.DataFrame(diff_data).sort_values('diff', key=abs, ascending=False)
        top_diff = diff_df.head(8)
        
        x = np.arange(len(top_diff))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, top_diff['oss'], width, label='OSS', color='lightblue')
        bars2 = ax5.bar(x + width/2, top_diff['oss4sg'], width, label='OSS4SG', color='lightgreen')
        
        ax5.set_xlabel('Feature')
        ax5.set_ylabel('Correlation')
        ax5.set_title('E. Features with Largest Correlation Difference', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f[:12] for f in top_diff['feature']], rotation=45, ha='right', fontsize=8)
        ax5.legend()
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.grid(True, alpha=0.3)
        
        # 6. Success patterns comparison
        ax6 = plt.subplot(2, 3, 6)
        
        key_features = ['total_commits_90d', 'activity_rate', 'sustained_growth', 'month3_commits']
        
        data_comparison = []
        for feat in key_features:
            for ptype in ['OSS', 'OSS4SG']:
                subset = features_df[features_df['project_type'] == ptype]
                core_mean = subset[subset['became_core'] == 1][feat].mean()
                non_core_mean = subset[subset['became_core'] == 0][feat].mean()
                ratio = core_mean / (non_core_mean + 0.001)
                data_comparison.append({
                    'feature': feat.replace('_', ' ').title()[:15],
                    'type': ptype,
                    'ratio': ratio
                })
        
        comp_df = pd.DataFrame(data_comparison)
        pivot_df = comp_df.pivot(index='feature', columns='type', values='ratio')
        
        x = np.arange(len(key_features))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, pivot_df['OSS'], width, label='OSS', color='lightblue')
        bars2 = ax6.bar(x + width/2, pivot_df['OSS4SG'], width, label='OSS4SG', color='lightgreen')
        
        ax6.set_xlabel('Feature')
        ax6.set_ylabel('Core/Non-Core Ratio')
        ax6.set_title('F. Core vs Non-Core Feature Ratios', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(pivot_df.index, rotation=45, ha='right', fontsize=8)
        ax6.legend()
        ax6.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('OSS vs OSS4SG: Comparative Feature Analysis for Core Prediction', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        plt.savefig(self.output_dir / 'oss_vs_oss4sg_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'oss_vs_oss4sg_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  Saved comparative visualizations")
    
    def create_visualizations(self, features_df: pd.DataFrame, importance_df: pd.DataFrame):
        """
        Create academic-quality visualizations
        """
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Feature Importance Bar Chart
        ax1 = plt.subplot(3, 3, 1)
        top_features = importance_df.head(10)
        colors = ['green' if x > 0 else 'red' for x in top_features['correlation']]
        bars = ax1.barh(range(len(top_features)), top_features['correlation'], color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=8)
        ax1.set_xlabel('Correlation with Becoming Core')
        ax1.set_title('A. Top 10 Predictive Features', fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution Comparison for Top 3 Features
        for i, (idx, feat) in enumerate(importance_df.head(3).iterrows()):
            ax = plt.subplot(3, 3, i+2)
            feature_name = feat['feature']
            
            core_values = features_df[features_df['became_core'] == 1][feature_name].dropna()
            non_core_values = features_df[features_df['became_core'] == 0][feature_name].dropna()
            
            # Create violin plot
            parts = ax.violinplot([non_core_values, core_values], positions=[0, 1], 
                                  widths=0.6, showmeans=True, showmedians=True)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Non-Core', 'Core'])
            ax.set_ylabel(feature_name.replace('_', ' ').title(), fontsize=8)
            ax.set_title(f'{chr(66+i)}. {feature_name.replace("_", " ").title()}', fontweight='bold', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add effect size annotation
            es = feat['effect_size']
            ax.text(0.5, ax.get_ylim()[1]*0.9, f'Effect Size: {es:.2f}', 
                   ha='center', fontsize=8, style='italic')
        
        # 3. Model Performance Comparison
        ax3 = plt.subplot(3, 3, 5)
        if 'cv_results' in self.results:
            model_names = list(self.results['cv_results'].keys())
            roc_scores = [self.results['cv_results'][m]['roc_auc_test_mean'] for m in model_names]
            pr_scores = [self.results['cv_results'][m]['pr_auc_test_mean'] for m in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, roc_scores, width, label='ROC-AUC', color='steelblue')
            bars2 = ax3.bar(x + width/2, pr_scores, width, label='PR-AUC', color='coral')
            
            ax3.set_xlabel('Model')
            ax3.set_ylabel('Score')
            ax3.set_title('E. Model Performance Comparison', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=7)
        
        # 4. Temporal Pattern Analysis
        ax4 = plt.subplot(3, 3, 6)
        core_df = features_df[features_df['became_core'] == 1]
        non_core_df = features_df[features_df['became_core'] == 0]
        
        months = ['month1_commits', 'month2_commits', 'month3_commits']
        core_means = [core_df[m].mean() for m in months]
        non_core_means = [non_core_df[m].mean() for m in months]
        
        x = np.arange(3)
        ax4.plot(x, core_means, marker='o', label='Core', linewidth=2, markersize=8)
        ax4.plot(x, non_core_means, marker='s', label='Non-Core', linewidth=2, markersize=8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Month 1', 'Month 2', 'Month 3'])
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Average Commits')
        ax4.set_title('F. Temporal Activity Patterns', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. OSS vs OSS4SG Comparison
        ax5 = plt.subplot(3, 3, 7)
        oss_df = features_df[features_df['project_type'] == 'OSS']
        oss4sg_df = features_df[features_df['project_type'] == 'OSS4SG']
        
        comparison_features = ['total_commits_90d', 'activity_rate', 'sustained_growth']
        oss_core_means = [oss_df[oss_df['became_core'] == 1][f].mean() for f in comparison_features]
        oss4sg_core_means = [oss4sg_df[oss4sg_df['became_core'] == 1][f].mean() for f in comparison_features]
        
        x = np.arange(len(comparison_features))
        width = 0.35
        
        ax5.bar(x - width/2, oss_core_means, width, label='OSS', color='lightblue')
        ax5.bar(x + width/2, oss4sg_core_means, width, label='OSS4SG', color='lightgreen')
        
        ax5.set_xlabel('Feature')
        ax5.set_ylabel('Mean Value (Core Contributors)')
        ax5.set_title('G. OSS vs OSS4SG Core Patterns', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f.replace('_', '\n') for f in comparison_features], fontsize=8)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Feature Correlation Heatmap
        ax6 = plt.subplot(3, 3, 8)
        feature_cols = importance_df.head(10)['feature'].tolist()
        corr_matrix = features_df[feature_cols].corr()
        
        im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax6.set_xticks(np.arange(len(feature_cols)))
        ax6.set_yticks(np.arange(len(feature_cols)))
        ax6.set_xticklabels([f[:15] for f in feature_cols], rotation=45, ha='right', fontsize=7)
        ax6.set_yticklabels([f[:15] for f in feature_cols], fontsize=7)
        ax6.set_title('H. Feature Correlation Matrix', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        
        # 7. Success Rate by Activity Level
        ax7 = plt.subplot(3, 3, 9)
        activity_bins = pd.qcut(features_df['total_commits_90d'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        success_by_quartile = features_df.groupby(activity_bins)['became_core'].mean() * 100
        
        bars = ax7.bar(range(4), success_by_quartile.values, color='skyblue', edgecolor='navy')
        ax7.set_xticks(range(4))
        ax7.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        ax7.set_xlabel('Activity Quartile (Total Commits)')
        ax7.set_ylabel('Success Rate (%)')
        ax7.set_title('I. Success Rate by Activity Level', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('90-Day Core Contributor Prediction Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'comprehensive_analysis_figure.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'comprehensive_analysis_figure.png', dpi=150, bbox_inches='tight')
        print(f"  Saved comprehensive figure to {self.output_dir}")
        
        # Create additional focused plots
        self.create_focused_plots(features_df, importance_df)
    
    def create_focused_plots(self, features_df: pd.DataFrame, importance_df: pd.DataFrame):
        """
        Create additional focused plots for specific insights
        """
        # Plot 1: ROC and PR curves for best model
        if 'models' in self.results and 'Random Forest' in self.results['models']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Prepare data
            feature_cols = self.results['feature_names']
            X = features_df[feature_cols].fillna(0)
            y = features_df['became_core']
            X_scaled = self.results['scaler'].transform(X)
            
            # Get predictions
            model = self.results['models']['Random Forest']
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc = roc_auc_score(y, y_pred_proba)
            ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('Receiver Operating Characteristic', fontweight='bold')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            pr_auc = average_precision_score(y, y_pred_proba)
            ax2.plot(recall, precision, color='green', lw=2, 
                    label=f'PR curve (AUC = {pr_auc:.2f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve', fontweight='bold')
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'roc_pr_curves.pdf', dpi=300, bbox_inches='tight')
            print("  Saved ROC/PR curves")
    
    def generate_latex_table(self, cv_results: Dict):
        """
        Generate LaTeX table for academic paper
        """
        print("\n" + "=" * 80)
        print("GENERATING LATEX TABLE")
        print("=" * 80)
        
        latex_lines = []
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{Model Performance for 90-Day Core Contributor Prediction}")
        latex_lines.append(r"\label{tab:model_performance}")
        latex_lines.append(r"\begin{tabular}{lcccc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"Model & ROC-AUC & PR-AUC & F1 Score & Precision \\")
        latex_lines.append(r"\midrule")
        
        for model_name, results in cv_results.items():
            roc_auc = f"{results['roc_auc_test_mean']:.3f} ± {results['roc_auc_test_std']:.3f}"
            pr_auc = f"{results['pr_auc_test_mean']:.3f} ± {results['pr_auc_test_std']:.3f}"
            f1 = f"{results['f1_test_mean']:.3f} ± {results['f1_test_std']:.3f}"
            precision = f"{results['precision_test_mean']:.3f} ± {results['precision_test_std']:.3f}"
            
            latex_lines.append(f"{model_name} & {roc_auc} & {pr_auc} & {f1} & {precision} \\\\")
        
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")
        
        latex_table = '\n'.join(latex_lines)
        
        # Save to file
        with open(self.output_dir / 'model_performance_table.tex', 'w') as f:
            f.write(latex_table)
        
        print("LaTeX table saved to model_performance_table.tex")
        print("\nTable preview:")
        print(latex_table)
        
        return latex_table
    
    def generate_summary_report(self, features_df: pd.DataFrame, 
                               importance_df: pd.DataFrame,
                               cv_results: Dict):
        """
        Generate comprehensive summary report with type comparisons
        """
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("90-DAY CORE CONTRIBUTOR PREDICTION: SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset Summary
        report_lines.append("DATASET SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total contributors analyzed: {len(features_df):,}")
        report_lines.append(f"Core contributors: {features_df['became_core'].sum():,} "
                          f"({100*features_df['became_core'].mean():.1f}%)")
        report_lines.append(f"Features extracted: {len(importance_df)}")
        report_lines.append("")
        
        # By Project Type
        report_lines.append("BY PROJECT TYPE")
        report_lines.append("-" * 40)
        for ptype in ['OSS', 'OSS4SG']:
            subset = features_df[features_df['project_type'] == ptype]
            report_lines.append(f"{ptype}:")
            report_lines.append(f"  Contributors: {len(subset):,}")
            report_lines.append(f"  Core: {subset['became_core'].sum():,} "
                              f"({100*subset['became_core'].mean():.1f}%)")
        report_lines.append("")
        
        # Top Predictive Features - COMBINED
        report_lines.append("TOP 10 PREDICTIVE FEATURES (COMBINED MODEL)")
        report_lines.append("-" * 40)
        for idx, row in importance_df.head(10).iterrows():
            direction = "increases" if row['correlation'] > 0 else "decreases"
            report_lines.append(f"{row['feature']:30s} | r={row['correlation']:+.3f} | "
                              f"{direction} likelihood")
        report_lines.append("")
        
        # Type-Specific Feature Importance
        if 'feature_importance_by_type' in self.results:
            report_lines.append("TYPE-SPECIFIC TOP FEATURES")
            report_lines.append("-" * 40)
            
            for ptype in ['OSS', 'OSS4SG']:
                report_lines.append(f"\n{ptype} Top 5 Predictive Features:")
                type_importance = self.results['feature_importance_by_type'][ptype]
                for idx, row in type_importance.head(5).iterrows():
                    direction = "↑" if row['correlation'] > 0 else "↓"
                    report_lines.append(f"  {row['feature']:25s} | r={row['correlation']:+.3f} {direction}")
            
            # Find unique features
            oss_top10 = set(self.results['feature_importance_by_type']['OSS'].head(10)['feature'])
            oss4sg_top10 = set(self.results['feature_importance_by_type']['OSS4SG'].head(10)['feature'])
            
            unique_oss = oss_top10 - oss4sg_top10
            unique_oss4sg = oss4sg_top10 - oss_top10
            
            report_lines.append(f"\nUnique to OSS top-10: {', '.join(unique_oss) if unique_oss else 'None'}")
            report_lines.append(f"Unique to OSS4SG top-10: {', '.join(unique_oss4sg) if unique_oss4sg else 'None'}")
            report_lines.append("")
        
        # Model Performance - ALL MODELS
        report_lines.append("MODEL PERFORMANCE COMPARISON")
        report_lines.append("-" * 40)
        
        if 'type_comparison' in self.results:
            for model_type in ['Combined', 'OSS', 'OSS4SG']:
                report_lines.append(f"\n{model_type} Models:")
                type_data = self.results['type_comparison'][model_type]
                for model_name, metrics in type_data['models'].items():
                    report_lines.append(f"  {model_name:12s}: ROC={metrics['roc_auc']:.3f}, "
                                      f"PR={metrics['pr_auc']:.3f}, F1={metrics['f1']:.3f}")
        else:
            # Fallback to original cv_results
            for model_name, results in cv_results.items():
                report_lines.append(f"\n{model_name}:")
                report_lines.append(f"  ROC-AUC:  {results['roc_auc_test_mean']:.3f} ± {results['roc_auc_test_std']:.3f}")
                report_lines.append(f"  PR-AUC:   {results['pr_auc_test_mean']:.3f} ± {results['pr_auc_test_std']:.3f}")
                report_lines.append(f"  F1 Score: {results['f1_test_mean']:.3f} ± {results['f1_test_std']:.3f}")
        report_lines.append("")
        
        # Key Insights
        report_lines.append("KEY INSIGHTS")
        report_lines.append("-" * 40)
        
        # Analyze key patterns
        core_df = features_df[features_df['became_core'] == 1]
        non_core_df = features_df[features_df['became_core'] == 0]
        
        insights = []
        
        # Activity level insight
        core_commits = core_df['total_commits_90d'].median()
        non_core_commits = non_core_df['total_commits_90d'].median()
        ratio = core_commits / max(non_core_commits, 1)
        insights.append(f"• Core contributors make {ratio:.1f}x more commits in first 90 days "
                       f"(median: {core_commits:.0f} vs {non_core_commits:.0f})")
        
        # Consistency insight
        core_rate = core_df['activity_rate'].mean()
        non_core_rate = non_core_df['activity_rate'].mean()
        insights.append(f"• Core contributors are active {100*core_rate:.0f}% of weeks "
                       f"vs {100*non_core_rate:.0f}% for non-core")
        
        # Growth pattern
        core_growth = core_df['sustained_growth'].mean()
        non_core_growth = non_core_df['sustained_growth'].mean()
        insights.append(f"• {100*core_growth:.0f}% of core contributors show sustained growth "
                       f"vs {100*non_core_growth:.0f}% of non-core")
        
        # Project type differences
        oss_core_rate = features_df[features_df['project_type'] == 'OSS']['became_core'].mean()
        oss4sg_core_rate = features_df[features_df['project_type'] == 'OSS4SG']['became_core'].mean()
        insights.append(f"• OSS4SG projects have {100*oss4sg_core_rate:.1f}% core rate "
                       f"vs {100*oss_core_rate:.1f}% for conventional OSS")
        
        # Type-specific insights
        if 'feature_importance_by_type' in self.results:
            oss_df = features_df[features_df['project_type'] == 'OSS']
            oss4sg_df = features_df[features_df['project_type'] == 'OSS4SG']
            
            # Compare key metrics
            oss_core = oss_df[oss_df['became_core'] == 1]
            oss4sg_core = oss4sg_df[oss4sg_df['became_core'] == 1]
            
            insights.append(f"• OSS core contributors: {oss_core['total_commits_90d'].median():.0f} median commits")
            insights.append(f"• OSS4SG core contributors: {oss4sg_core['total_commits_90d'].median():.0f} median commits")
        
        for insight in insights:
            report_lines.append(insight)
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        report_text = '\n'.join(report_lines)
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\nReport saved to summary_report.txt")
        print("\n" + report_text)
        
        return report_text
    
    def run_complete_pipeline(self):
        """
        Execute the complete analysis pipeline with comparative analysis
        """
        print("\n" + "=" * 80)
        print("STARTING 90-DAY CORE CONTRIBUTOR PREDICTION PIPELINE")
        print("WITH OSS vs OSS4SG COMPARATIVE ANALYSIS")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        
        # 1. Load and prepare data
        transitions, weekly_activity = self.load_and_prepare_data()
        
        # 2. Extract features
        features_df = self.extract_all_features(transitions, weekly_activity)
        
        # 3. Analyze feature importance (for combined model)
        importance_df = self.analyze_feature_importance(features_df)
        
        # 4. Train models (combined)
        cv_results = self.train_models(features_df)
        
        # 5. COMPREHENSIVE ANALYSIS: Combined, OSS, OSS4SG
        type_results = self.analyze_by_project_type(features_df)
        
        # 6. Create visualizations
        self.create_visualizations(features_df, importance_df)
        
        # 7. Create comparative visualizations
        self.create_comparative_visualizations(features_df)
        
        # 8. Generate LaTeX table
        latex_table = self.generate_latex_table(cv_results)
        
        # 9. Generate summary report with comparisons
        summary_report = self.generate_summary_report(features_df, importance_df, cv_results)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"All results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.iterdir()):
            print(f"  - {file.name}")
        
        return {
            'features_df': features_df,
            'importance_df': importance_df,
            'cv_results': cv_results,
            'type_results': type_results,
            'latex_table': latex_table,
            'summary_report': summary_report
        }


def main():
    """
    Main execution function
    """
    # Create pipeline instance
    pipeline = CorePredictionPipeline(OUTPUT_DIR)
    
    # Run complete analysis
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "=" * 80)
    print("CITATIONS FOR ACADEMIC PAPER")
    print("=" * 80)
    print("""
@inproceedings{zhou2012makes,
  title={What make long term contributors: Willingness and opportunity in OSS community},
  author={Zhou, Minghui and Mockus, Audris},
  booktitle={2012 34th International Conference on Software Engineering (ICSE)},
  pages={518--528},
  year={2012},
  organization={IEEE}
}

@inproceedings{begel2014analyze,
  title={Analyze this! 145 questions for data scientists in software engineering},
  author={Begel, Andrew and Zimmermann, Thomas},
  booktitle={Proceedings of the 36th International Conference on Software Engineering},
  pages={12--23},
  year={2014}
}
    """)
    
    return results


if __name__ == "__main__":
    results = main()