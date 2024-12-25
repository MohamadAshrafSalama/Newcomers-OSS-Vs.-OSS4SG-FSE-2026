## Step 5: Weekly Datasets – Documentation

This step generates two weekly datasets:
- Dataset 1: Project core timeline (weekly core contributor sets)
- Dataset 2: Contributor activity per project per week

### What we tested

- Dataset 1 validator (quick): `tester/test_core_timeline_quick.py`
  - Schema, nulls, JSON, week sequence, threshold logic, basic cross-check with commits
  - Result: Passed (reasonable stats, sequential weeks, non-empty core lists)

- Dataset 2 validator (original): `tester/test_contributor_activity_weekly.py`
  - Found all zeros for commits/cumulative; invalid emails; core mismatches

- Diagnostic (Dataset 2): `tester/diagnose_contributor_activity.py`
  - Confirmed zeros across metrics; no hashes/lines/files; sampled project files also zeros

- Fixed Dataset 2 (test-mode) quick test: `tester/test_contributor_activity_fixed_quick.py`
  - Confirms non-zero weekly commits, cumulative growth, core flags, and hashes/lines/files

### Issues encountered (Dataset 2)

1. All activity measures were zero
   - Root cause: week key misalignment and timezone issues causing groupby lookups to miss
   - Symptom: `commits_this_week == 0`, `cumulative_commits == 0`, `is_core_this_week == False`

2. Core status mismatches
   - Root cause: using tz-aware timestamps vs string dates for lookup into the core timeline

3. Invalid/blank contributor emails
   - Some entries lacked `@` or were parsed as `nan`

4. Checkpoint reuse masking fixes
   - Old zero-data outputs were re-consolidated due to checkpoint reuse

### Fixes applied

- Week calculation and alignment
  - Use Monday 00:00 UTC week starts without dropping timezone
  - Build contributor-week metrics keyed by `(email, week_start_utc)`
  - Generate weeks with `tz='UTC'` and match keys exactly

- Core lookup
  - Use `(project_name, week_date.strftime('%Y-%m-%d'))` for robust lookup

- Email handling
  - Keep emails lowercased/stripped; drop only empty/`'nan'` values

- Checkpoint hygiene
  - Clear test output directory before regenerating the test slice

### Files and scripts

- Core timeline generator: `project_core_timeline_weekly.py`
- Contributor activity (fixed, full run): `contributor_activity_weekly_fixed.py`
- Contributor activity (fixed, test-mode): `contributor_activity_weekly_fixed_test.py` (archived)
- Validators/diagnostics in `tester/`

### How to run (Dataset 2)

1) Fixed full generator

```
python3 RQ1_transition_rates_and_speeds/step5_weekly_datasets/contributor_activity_weekly_fixed.py
```

Outputs to: `RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/`

2) Validate (optional quick checks)

```
python3 RQ1_transition_rates_and_speeds/step5_weekly_datasets/tester/test_contributor_activity_weekly.py
```

3) Validate (fixed test-mode quick)

```
python3 RQ1_transition_rates_and_speeds/step5_weekly_datasets/tester/test_contributor_activity_fixed_quick.py
```

### Notes

- The full run can take around an hour and generate tens of millions of rows. Ensure enough disk space and RAM.
- Project-level results are saved under `dataset2_contributor_activity/project_results/` and then consolidated.

