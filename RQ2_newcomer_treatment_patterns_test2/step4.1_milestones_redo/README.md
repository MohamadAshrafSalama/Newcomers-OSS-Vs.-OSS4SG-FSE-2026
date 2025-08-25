# Evidence-Based OSS Contributor Progression Milestones (Step 4.1)

Data policy (Option B)
- Scope: Use only core contributors who have a timeline CSV under `step2_timelines/from_cache_timelines/`.
- Period: From each contributor’s first interaction to the first time they became core.
- Pre-core only: All milestone computations use `is_pre_core == True` events.
- Rationale: This is a focused re-implementation for milestones using the existing time-series subset. Missing timelines will be documented separately and are not used here.

Milestones

1. First Merged Pull Request (FMPR)
- Supporting paper: Zhou & Mockus (2012) — "What Makes Long Term Contributors: Willingness and Opportunity in OSS Community"
- Threshold: First PR with status = "merged" and author ≠ merger
- Rationale: Initial social + technical validation; at least one accepted contribution strongly predicts long-term engagement.

2. Sustained Participation Over 12 Weeks (SP12W)
- Supporting paper: Zhou & Mockus (2012) — theoretical extension to 12 weeks
- Threshold: Maintain activity (commits, PRs, issues, or comments) in ≥ 9 of 12 consecutive weeks (75% consistency)
- Rationale: Distinguishes sustained commitment from drive-by contributions.

3. Cross-Component Contribution Breadth (CCCB)
- Not used in this run (Option B timelines lack file paths). We explicitly skip CCCB and document this limitation.

4. Failure Recovery Resilience (FRR)
- Supporting paper: Legay, Decan & Mens (2018) — "On the Impact of Pull Request Decisions on Future Contributions"
- Threshold: Make a new contribution after ≥ 1 pre-core PR rejection (closed and not merged)
- Rationale: Persistence following rejection signals resilience against typical dropout trends.

5. Return After Extended Absence (RAEA)
- Supporting paper: Calefato et al. (2022) — "Will You Come Back to Contribute? Investigating the Inactivity of OSS Core Developers in GitHub"
- Threshold: Return to contribution after 3–6 months of inactivity (pre-core)
- Rationale: Early evidence of the break-and-return pattern common among core developers.

6. High Acceptance Rate Trajectory (HART)
- Supporting paper: Legay, Decan & Mens (2018)
- Variants (evaluate in order; first to satisfy sets milestone time and is recorded):
  - Classic: running acceptance ≥ 0.67 within first 10 pre-core PRs
  - Adjusted (small-sample): running acceptance Wilson 95% lower bound ≥ 0.67 with at least 3 PRs
  - Simple-count fallback: achieve when total pre-core PRs ≥ N_min (default 3) and acceptance ≥ 0.67 (plain proportion)
- Rationale: Classic mirrors prior work; Adjusted reduces small-N bias; Simple-count provides an interpretable fallback when counts are very low.

7. Direct Commit Access Proxy (DCA)
- Purpose: Proxy for “commit access gained” without committer fields.
- Pre-core variant (default): Find the longest contiguous pre-core run of commit events with no intervening PR events; milestone time = timestamp of the first commit in that run. Parameter M (minimum run length) defaults to 3; we will also report the run length.
- Optional post-core variant: Starting at first core week, find the longest contiguous run of commit events without PRs; timestamp = first commit in that run. Reported separately as sensitivity (not used in pre-core Markov flows).
- Caveat: Without committer identity or PR–commit linkage, this is a heuristic; merges can appear as commits. Treat as a proxy, not definitive commit rights.

Notes
- All milestones are assessed strictly before the first core week.
- Implementation scripts will live in this directory, with outputs clearly labeled and reproducible.

Clarifications and parameters
- Week definition: `event_week` is computed in timelines as floor((event_timestamp - first_commit_dt)/7 days). We treat the earliest pre-core `event_week` as the starting point and slide windows over this integer index.
- SP12W timing: milestone time is the end of the first 12-week window that reaches ≥ 9 active weeks. A week is active if it contains at least one event (commit, PR, issue, or comment embedded in PR/issue timelines). The milestone captures when consistency is first demonstrated.
- FRR contributions: any later pre-core event counts (commit, PR, or issue) after the first rejected PR. Issues do count here.
- RAEA gap window: 90–180 days follows Calefato et al. (2022) guidance (3–6 months is a meaningful absence with reasonable return probability). This window will be parameterized (defaults: min_gap_days=90, max_gap_days=180).
- HART evaluation order: Classic → Adjusted → Simple-count fallback. Record which variant triggered and at which PR index (or final count for Simple).
- Optional “Commit Access Gained” milestone (CA): Tan et al. (2024) detect this via the first time a developer appears as Git committer (Author==Committer). Our current pre-core timelines store commit payloads with author fields only (no committer). True CA detection requires enriching timelines with committer_name/committer_email from git logs or the GitHub commit API. As a proxy (when available in PR JSON), we may mark “Self-merge” when `mergedBy == author`, but this is weaker than commit rights.

Analysis cohorts (run both)
- All timelines: use every core contributor with a pre-core timeline (Option B).
- Min15 filter: exclude contributors with < 15 pre-core events to reduce extremely short histories.
- We will compute milestone detection, sequences, Markov graphs, top-5 flows, and plots for both cohorts, and compare Overall/OSS/OSS4SG within each cohort.

Markov Graph Experiment (Progression Flows)

Goal
- Map how contributors typically progress across the milestones (pre-core) using a first-order Markov model, and compare flows for Overall vs OSS vs OSS4SG.

Inputs (Option B)
- Timelines: `RQ2_newcomer_treatment_patterns_test2/step2_timelines/from_cache_timelines/timeline_*.csv` (use only files that exist)
- Use only rows where `is_pre_core == True` and only contributors who became core (as per RQ1 transitions).

States
- START: initial interaction (the first pre-core event)
- FMPR: First Merged Pull Request
- SP12W: Sustained Participation over 12 weeks (≥ 9/12 active weeks)
- CCCB: Cross-Component Contribution Breadth (SKIPPED in this run)
- FRR: Failure Recovery Resilience (new contribution after ≥ 1 rejected PR)
- RAEA: Return After Extended Absence (return after a 3–6 month gap)
- HART: High Acceptance Rate Trajectory (≥ 67% acceptance within first 10 PRs)
- END: termination at first core week

Metric extraction (from timelines)
- General preprocessing:
  - Sort by `event_timestamp` and filter `is_pre_core == True`.
  - Weekly activity: group by `event_week`; a week is active if it has ≥ 1 event of any type (commit, PR, issue, comment if present in PR/Issue timelines).
  - PR fields: parse `event_data` JSON for `merged`/`state`/`mergedAt` and for file paths when present (`files.nodes[].path` or similar); fallback to `changedFiles` counts when paths are missing.

- FMPR
  - Detection: first `pull_request` with merged == True (boolean), or state == "MERGED", or has `mergedAt` timestamp, where PR author ≠ merger if author info is available.
  - Timestamp: event time of that PR.

- SP12W
  - Slide a 12-week window from first pre-core week; flag achieved if any 12-week window has ≥ 9 active weeks.
  - Timestamp: the end week of the first window that satisfies the condition.

- CCCB
  - Skipped: timelines do not include file paths; we avoid weak proxies here. We may revisit if data are enriched.

- FRR
  - Identify rejected PRs: PR is closed and not merged (state == CLOSED or has `closedAt`, and merged == False and no `mergedAt`).
  - Flag achieved if there exists any later pre-core contribution (commit, PR, or issue) after the first rejection.
  - Timestamp: first post-rejection event time.

- RAEA
  - Compute gaps between consecutive pre-core events. If there is a gap in [90, 180] days, flag achieved when the contributor returns (the next event after the gap).
  - Timestamp: the first event after a qualifying gap.

- HART
  - Consider the contributor’s first 10 pre-core PRs (or all if <10). Acceptance rate = merged_count / total_considered.
  - Achieved if acceptance rate ≥ 0.67; timestamp = time of the PR that first makes the running acceptance ≥ 0.67 (or the 10th PR if only then).

Sequence construction (per contributor)
- Build a monotonic progression sequence of states: START → [each milestone at the time it is first achieved, ordered by timestamp] → END.
- Each milestone can occur at most once; milestones that never occur are omitted from that contributor’s sequence.
- Keep the contributor’s `project_type` for stratified graphs.

Markov graph creation
- Nodes: the states above. Edges: observed transitions between successive states in sequences.
- Transition probabilities: p(s→t) = count(s→t) / sum_over_u(count(s→u)).
- Pruning: drop edges with p ≤ 0.05.

Top-k flows
- Convert probabilities to costs: cost = -log(p).
- Use Yen’s algorithm (k-shortest paths) from START to END to extract top 5 highest-probability flows.

Stratified analyses and plots
- Compute three graphs and top flows: Overall, OSS only, OSS4SG only (based on `project_type`).
- Plot conventions:
  - Node = state, edge thickness ∝ p, edge labels show p (rounded), START in red, END in red.
  - Save figures under `step4.1_milestones_redo/figures/{overall,oss,oss4sg}_markov.png` and a CSV with the top-5 paths and their probabilities.

Execution plan (once scripts are added)
- Milestone extraction from timelines (pre-core): FMPR, SP12W, FRR, RAEA, HART (Classic & Adjusted & Simple), DCA (pre-core). CCCB skipped. Record timestamps and triggering variant where applicable.
- Build sequences (START → milestones in chronological order → END) per contributor.
- Construct transition matrices and probabilities per cohort (All, Min15) and stratum (Overall, OSS, OSS4SG).
- Prune edges with p ≤ 0.05; compute top-5 flows via Yen’s algorithm; output CSV + figures.

Quality and logging
- Log counts of contributors included/excluded, and how many milestones were detected per contributor.
- Explicitly report how many timelines lacked PR file paths and used CCCB fallbacks.

Repro checklist (once scripts are added)
- Run extractor to compute milestone timestamps per contributor from timelines.
- Build sequences and transition matrices for Overall/OSS/OSS4SG.
- Prune p ≤ 0.05, compute top-5 flows with Yen’s algorithm, and render plots + CSV summaries.
