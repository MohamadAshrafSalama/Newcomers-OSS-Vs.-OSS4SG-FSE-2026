#!/usr/bin/env python3
"""
Build milestone sequences (START → milestones in chronological order → END)
from milestones_per_contributor.csv produced by extract_all_milestones.py.

Outputs:
- sequences_all.csv (all contributors with timelines)
- sequences_min15.csv (contributors with ≥15 pre-core events)
Each row: project_name, project_type, contributor_email, sequence (comma-separated states)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

BASE = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
DIR = BASE / "RQ2_newcomer_treatment_patterns_test2/step4.1_milestones_redo"
IN_CSV = DIR / "milestones_per_contributor.csv"
OUT_ALL = DIR / "sequences_all.csv"
OUT_MIN15 = DIR / "sequences_min15.csv"


def to_ts(x: Optional[str]) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return pd.to_datetime(x, utc=False)
    except Exception:
        return None


def build_sequence(row: pd.Series) -> List[str]:
    # Collect milestones present with timestamps
    events: List[tuple[pd.Timestamp, str]] = []
    # Names and labels
    start_ts = to_ts(row.get('start_timestamp'))
    start_week = int(row.get('start_event_week')) if not pd.isna(row.get('start_event_week')) else 0

    # FMPR
    fmpr_ts = to_ts(row.get('fmpr_timestamp'))
    if isinstance(fmpr_ts, pd.Timestamp):
        events.append((fmpr_ts, 'FirstMergedPullRequest'))

    # SP12W: convert week to timestamp relative to start
    spw = row.get('sp12w_event_week')
    if not pd.isna(spw) and isinstance(start_ts, pd.Timestamp):
        delta_w = int(spw) - start_week
        sp_ts = start_ts + pd.Timedelta(weeks=delta_w)
        events.append((sp_ts, 'SustainedParticipation12w'))

    # FRR
    frr_ts = to_ts(row.get('frr_timestamp'))
    if isinstance(frr_ts, pd.Timestamp):
        events.append((frr_ts, 'FailureRecovery'))

    # RAEA
    raea_ts = to_ts(row.get('raea_timestamp'))
    if isinstance(raea_ts, pd.Timestamp):
        events.append((raea_ts, 'ReturnAfterExtendedAbsence'))

    # HART: classic-only
    if str(row.get('hart_variant')) == 'classic':
        hart_ts = to_ts(row.get('hart_timestamp'))
        if isinstance(hart_ts, pd.Timestamp):
            events.append((hart_ts, 'HighAcceptanceTrajectory'))

    # DCA
    dca_ts = to_ts(row.get('dca_timestamp'))
    if isinstance(dca_ts, pd.Timestamp):
        events.append((dca_ts, 'DirectCommitAccessProxy'))

    events.sort(key=lambda x: x[0])
    seq = ['START'] + [name for _, name in events] + ['END']
    # Deduplicate consecutive duplicates (safety)
    dedup: List[str] = []
    for s in seq:
        if not dedup or dedup[-1] != s:
            dedup.append(s)
    return dedup


def write_sequences(df: pd.DataFrame, out_file: Path) -> None:
    rows: List[Dict] = []
    for _, r in df.iterrows():
        seq = build_sequence(r)
        rows.append({
            'project_name': r['project_name'],
            'project_type': r['project_type'],
            'contributor_email': r['contributor_email'],
            'sequence': ','.join(seq)
        })
    pd.DataFrame(rows).to_csv(out_file, index=False)


def main() -> None:
    df = pd.read_csv(IN_CSV, low_memory=False)
    # All
    all_df = df.copy()
    write_sequences(all_df, OUT_ALL)
    # Min15
    if 'precore_events' in df.columns:
        min15_df = df[df['precore_events'] >= 15].copy()
    else:
        min15_df = df.copy()
    write_sequences(min15_df, OUT_MIN15)
    print(f"Wrote: {OUT_ALL}")
    print(f"Wrote: {OUT_MIN15}")


if __name__ == '__main__':
    main()


