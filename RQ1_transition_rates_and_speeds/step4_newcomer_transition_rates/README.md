# Step 4: Newcomer-to-Core Transition Rate Analysis

## Objective
Analyze the **dynamic rate** at which newcomers become core contributors over time, comparing OSS vs OSS4SG projects.

## Research Question
**RQ1.2**: How frequently do newcomers transition to core contributor status in OSS vs OSS4SG projects?

## Key Metric
**Monthly Core Transition Rate** = (New core contributors this month) / (Existing core contributors before this month)

## Methodology

### Step 1: Temporal Core Contributor Detection
For each project:
1. Sort all commits chronologically
2. For each month, apply the 80% rule based on commits up to that point
3. Track who are the core contributors at each time point
4. Identify **transition events** (when someone becomes core for the first time)

### Step 2: Transition Rate Calculation
For each project-month:
1. Count new core contributors this month
2. Count existing core contributors before this month
3. Calculate transition rate (handle edge cases for first month)
4. Only include months with at least 1 existing core contributor

### Step 3: Aggregation and Comparison
1. Calculate average monthly transition rates per project
2. Compare OSS vs OSS4SG using statistical tests
3. Analyze temporal patterns (seasonal effects, project age effects)

## Expected Outputs
1. **Transition rate dataset**: Project-level monthly transition rates
2. **Statistical comparison**: OSS vs OSS4SG transition rate differences
3. **Time series plots**: Showing transition patterns over time
4. **Summary statistics**: Average rates, peak periods, etc.

## Algorithm Details

### Core Contributors at Time T
```python
def get_core_contributors_at_time(commits_up_to_t):
    """Apply 80% rule to commits up to time T"""
    contributor_commits = commits_up_to_t.groupby('author_name').size()
    contributor_commits = contributor_commits.sort_values(ascending=False)
    
    total_commits = contributor_commits.sum()
    cumulative = contributor_commits.cumsum()
    cumulative_pct = cumulative / total_commits
    
    # Find core contributors (80% rule)
    core_mask = cumulative_pct <= 0.80
    if core_mask.sum() == 0:
        return [contributor_commits.index[0]]
    
    core_contributors = contributor_commits[core_mask].index.tolist()
    # Add the contributor that pushes us over 80%
    if cumulative_pct[core_contributors[-1]] < 0.80:
        next_idx = len(core_contributors)
        if next_idx < len(contributor_commits):
            core_contributors.append(contributor_commits.index[next_idx])
    
    return core_contributors
```

### Monthly Transition Detection
```python
def calculate_monthly_transitions(project_commits):
    """Calculate monthly core transition rates"""
    # Group by year-month
    project_commits['year_month'] = project_commits['commit_date'].dt.to_period('M')
    
    transitions = []
    previous_core = set()
    
    for month in sorted(project_commits['year_month'].unique()):
        # Get commits up to this month
        commits_up_to_month = project_commits[
            project_commits['year_month'] <= month
        ]
        
        # Find current core contributors
        current_core = set(get_core_contributors_at_time(commits_up_to_month))
        
        # Find new core contributors
        new_core = current_core - previous_core
        existing_core_count = len(previous_core)
        
        # Calculate transition rate
        if existing_core_count > 0:  # Skip first month
            transition_rate = len(new_core) / existing_core_count
            transitions.append({
                'month': month,
                'new_core_count': len(new_core),
                'existing_core_count': existing_core_count,
                'transition_rate': transition_rate,
                'total_core_after': len(current_core)
            })
        
        previous_core = current_core
    
    return pd.DataFrame(transitions)
```

## Data Requirements
- Input: `master_commits_dataset.csv` (3.5M commits)
- Minimum project requirements:
  - At least 12 months of activity
  - At least 10 core contributors over lifetime
  - At least 50 total commits

## Expected Results
- **OSS4SG Hypothesis**: Higher transition rates due to mission-driven attraction
- **OSS Hypothesis**: More stable core teams with lower transition rates
- **Temporal Patterns**: Seasonal effects, project lifecycle patterns

## Statistical Tests
- Mann-Whitney U test for OSS vs OSS4SG comparison
- Time series analysis for temporal patterns
- Effect size calculation (Cliff's Delta)