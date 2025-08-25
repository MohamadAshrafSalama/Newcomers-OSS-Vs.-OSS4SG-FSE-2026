#!/usr/bin/env python3
"""
Summarize pre-core activity counts per core contributor (Option B timelines) and
report how many contributors would be excluded at various activity thresholds.

Outputs:
- CSV with per-contributor pre-core activity counts and project_type
- JSON summary with counts kept/dropped for thresholds
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

BASE = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
TRANSITIONS = BASE / "RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv"
TIMELINES = BASE / "RQ2_newcomer_treatment_patterns_test2/step2_timelines/from_cache_timelines"
OUT_DIR = BASE / "RQ2_newcomer_treatment_patterns_test2/step4.1_milestones_redo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "precore_activity_counts.csv"
OUT_JSON = OUT_DIR / "precore_activity_threshold_summary.json"


def build_timeline_name(project_name: str, email: str) -> str:
    return f"timeline_{project_name.replace('/', '_')}_{email.replace('@', '_at_')}.csv"


def main() -> None:
    df = pd.read_csv(TRANSITIONS, usecols=["project_name","project_type","contributor_email","became_core"], low_memory=False)
    core_df = df[df["became_core"] == True].copy()

    rows: List[Dict] = []
    for _, r in tqdm(core_df.iterrows(), total=len(core_df), desc="Counting pre-core activities"):
        proj = r["project_name"]
        ptype = r["project_type"]
        email = r["contributor_email"]
        fname = build_timeline_name(proj, email)
        fpath = TIMELINES / fname
        if not fpath.exists():
            continue
        try:
            tl = pd.read_csv(fpath, usecols=["event_type","is_pre_core"], low_memory=False)
        except Exception:
            continue
        pre = tl[tl["is_pre_core"] == True]
        rows.append({
            "project_name": proj,
            "project_type": ptype,
            "contributor_email": email,
            "precore_events": int(len(pre))
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    thresholds = [0, 1, 5, 10, 15, 20, 25, 50]
    summary = {"total_with_timeline": int(len(out_df))}

    for t in thresholds:
        kept = int((out_df["precore_events"] >= t).sum())
        dropped = int(len(out_df) - kept)
        summary[str(t)] = {"kept": kept, "dropped": dropped}

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Total with timelines:", summary["total_with_timeline"]) 
    print("Threshold summary (events >= T):")
    for t in thresholds:
        print(f"T={t}: kept={summary[str(t)]['kept']}, dropped={summary[str(t)]['dropped']}")


if __name__ == "__main__":
    main()
