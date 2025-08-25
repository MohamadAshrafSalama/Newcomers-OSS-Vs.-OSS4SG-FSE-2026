#!/usr/bin/env python3
"""
Extract all pre-core milestones per contributor using Option B timelines.

Milestones implemented (timestamps and/or event_week when applicable):
- START (first pre-core event timestamp)
- FMPR (First Merged Pull Request)
- SP12W (≥9 active weeks in any 12-week window; milestone week = end of first window)
- CCCB (≥5 distinct top-level directories across PR files; first time reaching 5)
- FRR (any later pre-core event after the first rejected PR)
- RAEA (return after 90–180 days gap; time of first event after the first qualifying gap)
- HART (Classic→Adjusted→Simple evaluation order; milestone when >= 0.67)
- DCA (Direct Commit Access proxy: first commit of the longest run of commits since last PR; M=3)

Outputs:
- milestones_per_contributor.csv: one row per contributor-timeline with milestone timestamps/weeks and metadata
- milestone_stats.json: counts per milestone achieved and data quality diagnostics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

BASE = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
TRANSITIONS = BASE / "RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv"
TIMELINES = BASE / "RQ2_newcomer_treatment_patterns_test2/step2_timelines/from_cache_timelines"
OUT_DIR = BASE / "RQ2_newcomer_treatment_patterns_test2/step4.1_milestones_redo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "milestones_per_contributor.csv"
OUT_JSON = OUT_DIR / "milestone_stats.json"


def build_timeline_name(project_name: str, email: str) -> str:
    return f"timeline_{project_name.replace('/', '_')}_{email.replace('@', '_at_')}.csv"


def safe_json(s: str | None) -> Dict:
    if not isinstance(s, str) or s.strip() == "":
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def is_pr_merged(pr: Dict) -> bool:
    merged = pr.get('merged')
    if isinstance(merged, bool) and merged:
        return True
    state = pr.get('state')
    if isinstance(state, str) and state.upper() == 'MERGED':
        return True
    merged_at = pr.get('mergedAt') or pr.get('merged_at')
    return isinstance(merged_at, str) and len(merged_at) > 0


def is_pr_closed(pr: Dict) -> bool:
    state = pr.get('state')
    if isinstance(state, str) and state.upper() == 'CLOSED':
        return True
    closed_at = pr.get('closedAt') or pr.get('closed_at')
    return isinstance(closed_at, str) and len(closed_at) > 0


def is_pr_rejected(pr: Dict) -> bool:
    return is_pr_closed(pr) and not is_pr_merged(pr)


def extract_top_level_dirs_from_pr(pr: Dict) -> List[str]:
    names: List[str] = []
    files = pr.get('files')
    if isinstance(files, dict):
        nodes = files.get('nodes') or []
        if isinstance(nodes, list):
            for n in nodes:
                if isinstance(n, dict):
                    p = n.get('path') or n.get('filename')
                    if isinstance(p, str) and '/' in p:
                        names.append(p.split('/')[0])
    return names


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z * sqrt((p*(1-p)/n) + (z*z/(4*n*n)))
    return (centre - margin) / denom


@dataclass
class Stats:
    total_with_timeline: int = 0
    fmpr: int = 0
    sp12w: int = 0
    cccb: int = 0
    cccb_with_file_paths: int = 0
    frr: int = 0
    raea: int = 0
    hart: int = 0
    hart_variant_classic: int = 0
    hart_variant_adjusted: int = 0
    hart_variant_simple: int = 0
    dca: int = 0


def process_contributor(pre: pd.DataFrame) -> Dict[str, Optional[object]]:
    # Guard required columns
    required = {'event_type','event_timestamp','event_week','event_data'}
    if not required.issubset(set(pre.columns)):
        return {
            'start_timestamp': None,
            'start_event_week': None,
            'fmpr_timestamp': None,
            'fmpr_event_week': None,
            'sp12w_event_week': None,
            'cccb_event_week': None,
            'cccb_supported': False,
            'frr_timestamp': None,
            'raea_timestamp': None,
            'hart_timestamp': None,
            'hart_variant': None,
            'dca_timestamp': None,
        }
    pre = pre.sort_values('event_timestamp').reset_index(drop=True)
    first_ts = pd.to_datetime(pre.iloc[0]['event_timestamp'])
    first_week = int(pre.iloc[0]['event_week'])

    # FMPR
    prs = pre[pre['event_type'] == 'pull_request'].copy()
    prs['data'] = prs['event_data'].apply(safe_json)
    merged_mask = prs['data'].apply(is_pr_merged)
    # pre is already sorted by event_timestamp; preserve order
    merged_prs = prs[merged_mask]
    fmpr_row = merged_prs.iloc[0] if len(merged_prs) > 0 else None

    fmpr_ts = None
    fmpr_week = None
    if fmpr_row is not None:
        fmpr_ts = str(fmpr_row['event_timestamp'])
        fmpr_week = int(fmpr_row['event_week'])

    # SP12W
    sp12w_week = None
    weeks = pre.groupby('event_week').size().sort_index()
    if len(weeks) > 0:
        week_indices = sorted(weeks.index.tolist())
        # Sliding window over integer weeks; fill missing weeks as zero active
        min_w, max_w = week_indices[0], week_indices[-1]
        active = pd.Series(0, index=range(min_w, max_w + 1))
        active.loc[weeks.index] = 1
        for start in range(min_w, max_w - 11):
            window_sum = int(active.loc[start:start+11].sum())
            if window_sum >= 9:
                sp12w_week = start + 11
                break

    # CCCB
    cccb_week = None
    cccb_supported = False
    seen_dirs: set[str] = set()
    if not prs.empty:
        for _, row in prs.iterrows():
            prd = row['data'] if 'data' in row else safe_json(row.get('event_data'))
            dirs = extract_top_level_dirs_from_pr(prd)
            if dirs:
                cccb_supported = True
            for d in dirs:
                if d:
                    seen_dirs.add(d)
            if len(seen_dirs) >= 5 and cccb_week is None:
                cccb_week = int(row['event_week'])
                # do not break; we still want cccb_supported flag set correctly

    # FRR
    frr_ts = None
    first_rejected_ts = None
    for _, row in prs.iterrows():
        prd = row['data'] if 'data' in row else safe_json(row.get('event_data'))
        if is_pr_rejected(prd):
            first_rejected_ts = pd.to_datetime(row['event_timestamp'])
            break
    if first_rejected_ts is not None:
        later = pre[pd.to_datetime(pre['event_timestamp']) > first_rejected_ts]
        if not later.empty:
            frr_ts = str(later.iloc[0]['event_timestamp'])

    # RAEA
    raea_ts = None
    ts_col = pd.to_datetime(pre['event_timestamp']).sort_values().reset_index(drop=True)
    gaps = ts_col.diff().dt.days
    for i in range(1, len(ts_col)):
        g = gaps.iloc[i]
        if g is not None and 90 <= int(g) <= 180:
            raea_ts = str(ts_col.iloc[i])
            break

    # HART (Classic -> Adjusted -> Simple)
    hart_ts = None
    hart_variant = None
    pr_list = prs.to_dict('records')
    merged_flags = []
    for pr in pr_list:
        data = pr['data'] if 'data' in pr else safe_json(pr.get('event_data'))
        merged_flags.append(1 if is_pr_merged(data) else 0)
    # Classic: within first 10 PRs
    if len(pr_list) > 0:
        upto = min(10, len(pr_list))
        for i in range(upto):
            acc = sum(merged_flags[:i+1]) / (i+1)
            if acc >= 0.67:
                hart_ts = str(pr_list[i]['event_timestamp'])
                hart_variant = 'classic'
                break
    # Adjusted: Wilson LB >= 0.67 with at least 3 PRs
    if hart_ts is None and len(pr_list) >= 3:
        successes = 0
        for i in range(len(pr_list)):
            successes += merged_flags[i]
            n = i + 1
            if n >= 3:
                lb = wilson_lower_bound(successes, n, 1.96)
                if lb >= 0.67:
                    hart_ts = str(pr_list[i]['event_timestamp'])
                    hart_variant = 'adjusted'
                    break
    # Simple-count fallback: at least 3 PRs and plain acceptance >= 0.67
    if hart_ts is None and len(pr_list) >= 3:
        successes = sum(merged_flags)
        acc = successes / len(pr_list)
        if acc >= 0.67:
            hart_ts = str(pr_list[min(len(pr_list)-1, 9)]['event_timestamp'])
            hart_variant = 'simple'

    # DCA (updated): at least K commits after the last pre-core PR, with no later PRs before core
    dca_ts = None
    K = 5
    events = pre[['event_type','event_timestamp']].copy()
    events['event_timestamp'] = pd.to_datetime(events['event_timestamp'])
    events = events.sort_values('event_timestamp').reset_index(drop=True)

    last_pr_idx = None
    for i in range(len(events)-1, -1, -1):
        if events.iloc[i]['event_type'] == 'pull_request':
            last_pr_idx = i
            break
    if last_pr_idx is not None:
        after_last_pr = events.iloc[last_pr_idx+1:]
        # If there is any PR after last_pr_idx, then last_pr_idx was not actually the last
        if (after_last_pr['event_type'] == 'pull_request').any():
            pass  # no DCA by this definition
        else:
            commits_after = after_last_pr[after_last_pr['event_type'] == 'commit']
            if len(commits_after) >= K and not commits_after.empty:
                dca_ts = str(commits_after.iloc[0]['event_timestamp'])

    return {
        'start_timestamp': str(first_ts),
        'start_event_week': first_week,
        'fmpr_timestamp': fmpr_ts,
        'fmpr_event_week': fmpr_week,
        'sp12w_event_week': sp12w_week,
        'cccb_event_week': cccb_week,
        'cccb_supported': cccb_supported,
        'frr_timestamp': frr_ts,
        'raea_timestamp': raea_ts,
        'hart_timestamp': hart_ts,
        'hart_variant': hart_variant,
        'dca_timestamp': dca_ts,
    }


def main() -> None:
    trans = pd.read_csv(TRANSITIONS, usecols=["project_name","project_type","contributor_email","became_core"], low_memory=False)
    core_df = trans[trans['became_core'] == True].copy()

    stats = Stats()
    rows: List[Dict] = []

    for _, r in tqdm(core_df.iterrows(), total=len(core_df), desc="All milestones"):
        proj = r['project_name']
        ptype = r['project_type']
        email = r['contributor_email']
        path = TIMELINES / build_timeline_name(proj, email)
        if not path.exists():
            continue
        try:
            tl = pd.read_csv(
                path,
                usecols=[
                    'event_type','event_timestamp','event_week','event_data','project_type','contributor_email','is_pre_core'
                ],
                low_memory=False
            )
        except Exception:
            continue
        if tl.empty:
            continue
        pre = tl[tl['is_pre_core'] == True].copy()
        if pre.empty:
            continue
        stats.total_with_timeline += 1
        res = process_contributor(pre)
        res.update({
            'project_name': proj,
            'project_type': ptype,
            'contributor_email': email,
            'precore_events': int(len(pre)),
        })
        rows.append(res)

        # increment stats
        if res['fmpr_timestamp'] is not None:
            stats.fmpr += 1
        if res['sp12w_event_week'] is not None:
            stats.sp12w += 1
        if res['cccb_event_week'] is not None:
            stats.cccb += 1
        if res['cccb_supported']:
            stats.cccb_with_file_paths += 1
        if res['frr_timestamp'] is not None:
            stats.frr += 1
        if res['raea_timestamp'] is not None:
            stats.raea += 1
        if res['hart_timestamp'] is not None:
            stats.hart += 1
            if res['hart_variant'] == 'classic':
                stats.hart_variant_classic += 1
            elif res['hart_variant'] == 'adjusted':
                stats.hart_variant_adjusted += 1
            elif res['hart_variant'] == 'simple':
                stats.hart_variant_simple += 1
        if res['dca_timestamp'] is not None:
            stats.dca += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    with OUT_JSON.open('w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, indent=2)

    print(json.dumps(asdict(stats), indent=2))


if __name__ == '__main__':
    main()


