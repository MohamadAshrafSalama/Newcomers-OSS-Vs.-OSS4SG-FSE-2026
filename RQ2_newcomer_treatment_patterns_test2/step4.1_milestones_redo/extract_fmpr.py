#!/usr/bin/env python3
"""
Part 1: Extract First Merged Pull Request (FMPR) per core contributor from pre-core timelines.

Outputs:
- CSV rows: project_name, project_type, contributor_email, fmpr_timestamp, fmpr_event_week
- Summary printout: counts with/without FMPR
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

BASE = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
TRANSITIONS = BASE / "RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv"
TIMELINES = BASE / "RQ2_newcomer_treatment_patterns_test2/step2_timelines/from_cache_timelines"
OUT_DIR = BASE / "RQ2_newcomer_treatment_patterns_test2/step4.1_milestones_redo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "fmpr_per_contributor.csv"


def build_timeline_name(project_name: str, email: str) -> str:
    return f"timeline_{project_name.replace('/', '_')}_{email.replace('@', '_at_')}.csv"


def safe_json(s: str | None) -> Dict:
    if not isinstance(s, str) or s.strip() == "":
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def is_merged(pr: Dict) -> bool:
    merged = pr.get('merged')
    if isinstance(merged, bool) and merged:
        return True
    state = pr.get('state')
    if isinstance(state, str) and state.upper() == 'MERGED':
        return True
    merged_at = pr.get('mergedAt') or pr.get('merged_at')
    return isinstance(merged_at, str) and len(merged_at) > 0


def find_fmpr(pre_df: pd.DataFrame) -> Optional[pd.Series]:
    prs = pre_df[pre_df['event_type'] == 'pull_request'].copy()
    if prs.empty:
        return None
    prs['data'] = prs['event_data'].apply(safe_json)
    merged_mask = prs['data'].apply(is_merged)
    merged_prs = prs[merged_mask]
    if merged_prs.empty:
        return None
    merged_prs = merged_prs.sort_values('event_timestamp')
    return merged_prs.iloc[0]


def main() -> None:
    df = pd.read_csv(TRANSITIONS, usecols=["project_name","project_type","contributor_email","became_core"], low_memory=False)
    core_df = df[df['became_core'] == True].copy()

    records = []
    found = 0
    missing_tl = 0
    processed = 0

    for _, r in tqdm(core_df.iterrows(), total=len(core_df), desc="Extracting FMPR"):
        proj = r['project_name']
        ptype = r['project_type']
        email = r['contributor_email']
        fname = build_timeline_name(proj, email)
        fpath = TIMELINES / fname
        if not fpath.exists():
            missing_tl += 1
            continue
        try:
            tl = pd.read_csv(fpath, usecols=["event_type","event_timestamp","event_week","event_data","is_pre_core"], low_memory=False)
        except Exception:
            continue

        pre = tl[tl['is_pre_core'] == True].copy()
        if pre.empty:
            continue

        fmpr = find_fmpr(pre)
        processed += 1
        if fmpr is not None:
            found += 1
            records.append({
                'project_name': proj,
                'project_type': ptype,
                'contributor_email': email,
                'fmpr_timestamp': fmpr['event_timestamp'],
                'fmpr_event_week': int(fmpr['event_week'])
            })

    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)

    print(f"Core contributors total: {len(core_df):,}")
    print(f"With timeline processed: {processed:,}")
    print(f"Missing timelines: {missing_tl:,}")
    print(f"FMPR found: {found:,}")
    print(f"FMPR not found: {processed - found:,}")


if __name__ == '__main__':
    main()


