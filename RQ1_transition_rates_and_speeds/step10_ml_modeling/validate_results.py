#!/usr/bin/env python3
"""
Validate the comprehensive ML pipeline results
"""

import pandas as pd
import numpy as np

def validate_results():
    print("=== COMPREHENSIVE DATASET VALIDATION ===")
    
    # Load the comprehensive dataset
    df = pd.read_csv('RQ1_transition_rates_and_speeds/step10_ml_modeling/features_90day_comprehensive.csv')
    
    print(f"Total contributors: {len(df):,}")
    print(f"Core contributors: {df['became_core'].sum():,}")
    print(f"Core rate: {df['became_core'].mean()*100:.1f}%")
    print(f"Projects: {df['project_name'].nunique()}")
    
    print("\nBy project type:")
    for ptype in ['OSS', 'OSS4SG']:
        subset = df[df['project_type'] == ptype]
        cores = subset['became_core'].sum()
        print(f"{ptype}: {len(subset):,} contributors, {cores:,} cores ({cores/len(subset)*100:.1f}%)")
    
    print("\nFeature statistics:")
    print(f"Commits 90d - Mean: {df['commits_90d'].mean():.1f}, Median: {df['commits_90d'].median():.1f}")
    print(f"Lines changed 90d - Mean: {df['lines_changed_90d'].mean():.0f}, Median: {df['lines_changed_90d'].median():.0f}")
    print(f"Active days 90d - Mean: {df['active_days_90d'].mean():.1f}, Median: {df['active_days_90d'].median():.1f}")
    
    # Top projects by contributor count
    print("\nTop 10 projects by contributor count:")
    top_projects = df['project_name'].value_counts().head(10)
    for project, count in top_projects.items():
        cores = df[df['project_name'] == project]['became_core'].sum()
        print(f"  {project}: {count:,} contributors, {cores:,} cores ({cores/count*100:.1f}%)")
    
    # Model results
    print("\n=== MODEL PERFORMANCE ===")
    results = pd.read_csv('RQ1_transition_rates_and_speeds/step10_ml_modeling/model_results_comprehensive.csv', index_col=0)
    for model in results.index:
        roc = results.loc[model, 'roc_auc_mean']
        pr = results.loc[model, 'pr_auc_mean']
        print(f"{model}: ROC AUC = {roc:.3f}, PR AUC = {pr:.3f}")
    
    # Feature importance
    print("\n=== TOP 10 FEATURES ===")
    features = pd.read_csv('RQ1_transition_rates_and_speeds/step10_ml_modeling/feature_importance_comprehensive.csv')
    for i, (_, row) in enumerate(features.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n=== COMPARISON WITH PREVIOUS STEP10 ===")
    print(f"Previous: 61 contributors, 2 cores (3.3%)")
    print(f"Current:  {len(df):,} contributors, {df['became_core'].sum():,} cores ({df['became_core'].mean()*100:.1f}%)")
    print(f"Improvement: {len(df)/61:.0f}x more contributors, {df['became_core'].sum()/2:.0f}x more cores")

if __name__ == "__main__":
    validate_results()

