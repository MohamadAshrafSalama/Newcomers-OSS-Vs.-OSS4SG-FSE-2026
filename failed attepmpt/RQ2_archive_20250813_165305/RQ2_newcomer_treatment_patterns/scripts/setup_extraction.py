#!/usr/bin/env python3
"""
Setup script for RQ2 GitHub treatment data extraction.
Creates candidate lists WITHOUT arbitrary date filtering.
"""

import sys
from pathlib import Path
import pandas as pd
import json


def setup_extraction():
    """Setup extraction with proper candidate selection."""
    
    print("🚀 RQ2 GitHub Treatment Data Extraction Setup")
    print("="*70)
    
    # Check for transitions dataset
    transitions_file = Path("RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv")
    if not transitions_file.exists():
        # Try alternative location
        transitions_file = Path("RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions_no_instant.csv")
    
    if not transitions_file.exists():
        print("❌ Cannot find transitions dataset. Please run Step 6 first.")
        sys.exit(1)
    
    print(f"✅ Found transitions dataset: {transitions_file}")
    
    # Create directory structure
    base_dir = Path("RQ2_newcomer_treatment_patterns")
    base_dir.mkdir(exist_ok=True)
    
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    scripts_dir = base_dir / "scripts"
    
    for dir_path in [data_dir, results_dir, scripts_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Load transitions data
    print("\n📊 Loading transitions data...")
    df = pd.read_csv(transitions_file)
    
    # Basic statistics
    total_contributors = len(df)
    core_achievers = df[df['became_core'] == True]
    total_core = len(core_achievers)
    
    print(f"  Total contributors: {total_contributors:,}")
    print(f"  Core achievers: {total_core:,}")
    print(f"  Core rate: {total_core/total_contributors*100:.1f}%")
    
    # Filter for reasonable extraction
    print("\n🔍 Filtering candidates...")
    
    # Start with core achievers
    candidates = core_achievers.copy()
    print(f"  Starting with {len(candidates):,} core achievers")
    
    # Remove instant cores (weeks_to_core = 0)
    instant_cores = candidates['weeks_to_core'] == 0
    candidates = candidates[~instant_cores]
    print(f"  Removed {instant_cores.sum()} instant cores")
    
    # Remove extremely long transitions (>5 years)
    extremely_long = candidates['weeks_to_core'] > 260
    candidates = candidates[~extremely_long]
    print(f"  Removed {extremely_long.sum()} with >5 year transitions")
    
    # Ensure we have required fields
    required_fields = ['project_name', 'contributor_email', 'first_commit_date', 
                      'first_core_date', 'weeks_to_core', 'project_type']
    
    for field in required_fields:
        if field not in candidates.columns:
            print(f"  ⚠️ Missing field: {field}")
            continue
        missing = candidates[field].isna()
        if missing.any():
            candidates = candidates[~missing]
            print(f"  Removed {missing.sum()} with missing {field}")
    
    print(f"\n✅ Final candidate count: {len(candidates):,}")
    
    # Analyze candidates
    print("\n📈 Candidate Analysis:")
    print(f"  By project type:")
    for ptype in candidates['project_type'].unique():
        count = len(candidates[candidates['project_type'] == ptype])
        pct = count / len(candidates) * 100
        print(f"    {ptype}: {count:,} ({pct:.1f}%)")
    
    print(f"\n  Transition duration (weeks):")
    print(f"    Min: {candidates['weeks_to_core'].min():.0f}")
    print(f"    25%: {candidates['weeks_to_core'].quantile(0.25):.0f}")
    print(f"    Median: {candidates['weeks_to_core'].median():.0f}")
    print(f"    75%: {candidates['weeks_to_core'].quantile(0.75):.0f}")
    print(f"    Max: {candidates['weeks_to_core'].max():.0f}")
    
    print(f"\n  Unique projects: {candidates['project_name'].nunique()}")
    
    # Save full candidate list
    full_file = data_dir / "candidates_all.csv"
    candidates.to_csv(full_file, index=False)
    print(f"\n💾 Saved {len(candidates):,} candidates to {full_file}")
    
    # Create balanced sample for testing
    print("\n🎲 Creating balanced test samples...")
    
    # Small sample (10 each)
    small_sample = []
    for ptype in candidates['project_type'].unique():
        type_candidates = candidates[candidates['project_type'] == ptype]
        sample = type_candidates.sample(n=min(10, len(type_candidates)), random_state=42)
        small_sample.append(sample)
    
    small_df = pd.concat(small_sample)
    small_file = data_dir / "candidates_test_20.csv"
    small_df.to_csv(small_file, index=False)
    print(f"  Created test sample: {len(small_df)} contributors")
    
    # Medium sample (50 each)
    medium_sample = []
    for ptype in candidates['project_type'].unique():
        type_candidates = candidates[candidates['project_type'] == ptype]
        sample = type_candidates.sample(n=min(50, len(type_candidates)), random_state=42)
        medium_sample.append(sample)
    
    medium_df = pd.concat(medium_sample)
    medium_file = data_dir / "candidates_test_100.csv"
    medium_df.to_csv(medium_file, index=False)
    print(f"  Created medium sample: {len(medium_df)} contributors")
    
    # Create token file template if doesn't exist
    token_file = base_dir / "github_tokens.txt"
    if not token_file.exists():
        with open(token_file, 'w') as f:
            f.write("# GitHub Personal Access Tokens\n")
            f.write("# One token per line\n")
            f.write("# Get from: https://github.com/settings/tokens\n")
            f.write("# Required scopes: repo (full), read:user\n")
            f.write("#\n")
            f.write("# Example:\n")
            f.write("# ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        print(f"\n📝 Created token template: {token_file}")
    else:
        print(f"\n✅ Token file exists: {token_file}")
    
    # Save metadata
    metadata = {
        'setup_date': pd.Timestamp.now().isoformat(),
        'transitions_file': str(transitions_file),
        'total_candidates': len(candidates),
        'by_type': {
            ptype: int(len(candidates[candidates['project_type'] == ptype]))
            for ptype in candidates['project_type'].unique()
        },
        'samples': {
            'test_20': len(small_df),
            'test_100': len(medium_df),
            'all': len(candidates)
        }
    }
    
    metadata_file = base_dir / "extraction_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("📋 NEXT STEPS")
    print("="*70)
    
    print("\n1️⃣ Add GitHub tokens:")
    print(f"   vim {token_file}")
    print("   Add at least 3-5 tokens for good performance")
    
    print("\n2️⃣ Test with 20 contributors:")
    print(f"""
python3 RQ2_newcomer_treatment_patterns/scripts/robust_extractor.py \
    --tokens RQ2_newcomer_treatment_patterns/github_tokens.txt \
    --contributors RQ2_newcomer_treatment_patterns/data/candidates_test_20.csv \
    --output-dir RQ2_newcomer_treatment_patterns/results/test_20
""")
    
    print("\n3️⃣ Run medium test (100 contributors):")
    print(f"""
python3 RQ2_newcomer_treatment_patterns/scripts/robust_extractor.py \
    --tokens RQ2_newcomer_treatment_patterns/github_tokens.txt \
    --contributors RQ2_newcomer_treatment_patterns/data/candidates_test_100.csv \
    --output-dir RQ2_newcomer_treatment_patterns/results/test_100
""")
    
    print("\n4️⃣ Run full extraction:")
    print(f"""
python3 RQ2_newcomer_treatment_patterns/scripts/robust_extractor.py \
    --tokens RQ2_newcomer_treatment_patterns/github_tokens.txt \
    --contributors RQ2_newcomer_treatment_patterns/data/candidates_all.csv \
    --output-dir RQ2_newcomer_treatment_patterns/results/full
""")
    
    print("\n5️⃣ Monitor progress (in another terminal):")
    print(f"""
python3 RQ2_newcomer_treatment_patterns/scripts/monitor.py \
    --output-dir RQ2_newcomer_treatment_patterns/results/full \
    --watch
""")
    
    # Time estimates
    print("\n⏱️ TIME ESTIMATES:")
    requests_per_contributor = 5  # Conservative estimate
    total_requests = len(candidates) * requests_per_contributor
    
    print(f"  Total candidates: {len(candidates):,}")
    print(f"  Estimated API calls: {total_requests:,}")
    print(f"  With 1 token: ~{total_requests/5000:.1f} hours")
    print(f"  With 5 tokens: ~{total_requests/25000:.1f} hours")
    print(f"  With 10 tokens: ~{total_requests/50000:.1f} hours")
    
    print("\n💡 TIPS:")
    print("  • The extractor automatically resumes if interrupted")
    print("  • Username resolution is cached for efficiency")
    print("  • Failed extractions are logged for debugging")
    print("  • Check extraction.log for detailed progress")
    
    return metadata


def check_dependencies():
    """Check required Python packages."""
    required = ['pandas', 'numpy', 'requests', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies installed\n")
    
    # Run setup
    metadata = setup_extraction()
    
    print("\n✅ Setup complete!")
    print(f"📊 Ready to extract data for {metadata['total_candidates']:,} contributors")


