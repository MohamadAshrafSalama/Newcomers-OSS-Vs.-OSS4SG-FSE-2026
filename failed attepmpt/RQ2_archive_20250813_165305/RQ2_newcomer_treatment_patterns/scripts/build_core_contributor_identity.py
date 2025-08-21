#!/usr/bin/env python3
"""
Build filtered core-contributor list (positives only), exclude instant core, and
limit to contributors who reached core within a maximum window (default 4 years).

Optionally, construct a commit-identity mapping by:
- Sampling one commit hash per (project_name, contributor_email) from Step 5
  weekly activity (either consolidated CSV or per-project activity files)
- Looking up that hash in the master commits dataset to retrieve the original
  author_name and author_email

Inputs (defaults point to RQ1 outputs and are read-only):
- Step 6 transitions: RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv
- Step 5 weekly activity (consolidated): RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv
- Step 5 per-project activity directory (fallback): RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/project_results
- Master commits dataset: RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv

Outputs (written under RQ2; RQ1 content is never modified):
- RQ2_newcomer_treatment_patterns/data/core_contributors_filtered.csv
- RQ2_newcomer_treatment_patterns/results/core_commit_identity.csv (optional if hashes found)
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd


DEFAULT_STEP6 = (
    "RQ1_transition_rates_and_speeds/step6_contributor_transitions/results/contributor_transitions.csv"
)
DEFAULT_STEP5 = (
    "RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/contributor_activity_weekly.csv"
)
DEFAULT_STEP5_PROJ_DIR = (
    "RQ1_transition_rates_and_speeds/step5_weekly_datasets/dataset2_contributor_activity/project_results"
)
DEFAULT_MASTER = (
    "RQ1_transition_rates_and_speeds/data_mining/step2_commit_analysis/consolidating_master_dataset/master_commits_dataset.csv"
)

OUT_FILTERED = (
    "RQ2_newcomer_treatment_patterns/data/core_contributors_filtered.csv"
)
OUT_IDENTITY = (
    "RQ2_newcomer_treatment_patterns/results/core_commit_identity.csv"
)


def load_filtered_core_list(
    step6_path: str,
    max_weeks_to_core: int,
    exclude_instant_core: bool = True,
) -> pd.DataFrame:
    """Load Step 6 transitions and filter to core contributors under constraints.

    Filters applied:
      - became_core == True
      - (weeks_to_core > 0) if exclude_instant_core
      - weeks_to_core <= max_weeks_to_core
    """
    usecols = [
        "project_name",
        "contributor_email",
        "became_core",
        "weeks_to_core",
        "first_core_week",
        "total_weeks_observed",
        "first_commit_date",
        "first_core_date",
        "last_observed_date",
    ]

    df = pd.read_csv(step6_path, usecols=[c for c in usecols if Path(step6_path).exists()])
    # Defensive: ensure required columns exist
    required = {"project_name", "contributor_email", "became_core", "weeks_to_core"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Step 6 file missing required columns: {missing}")

    # Positives only
    pos = df[df["became_core"] == True].copy()
    if exclude_instant_core and "weeks_to_core" in pos.columns:
        pos = pos[pos["weeks_to_core"] > 0]

    if "weeks_to_core" in pos.columns:
        pos = pos[pos["weeks_to_core"] <= max_weeks_to_core]

    # Deduplicate pairs; keep minimal fields for RQ2 downstream
    keep_cols = [
        c
        for c in [
            "project_name",
            "project_type" if "project_type" in pos.columns else None,
            "contributor_email",
            "weeks_to_core",
            "first_core_week" if "first_core_week" in pos.columns else None,
            "first_commit_date" if "first_commit_date" in pos.columns else None,
            "first_core_date" if "first_core_date" in pos.columns else None,
            "total_weeks_observed" if "total_weeks_observed" in pos.columns else None,
            "last_observed_date" if "last_observed_date" in pos.columns else None,
        ]
        if c is not None
    ]
    pos = pos[keep_cols]
    pos = pos.drop_duplicates(subset=["project_name", "contributor_email"]).reset_index(drop=True)
    return pos


def _iter_step5_chunks(
    step5_path: Path, project_results_dir: Path, usecols: List[str]
) -> Iterable[pd.DataFrame]:
    # Prefer consolidated file if available
    if step5_path.exists():
        try:
            for chunk in pd.read_csv(step5_path, usecols=usecols, chunksize=200_000):
                yield chunk
            return
        except Exception:
            pass

    # Fallback to per-project files
    files = []
    if project_results_dir.exists() and project_results_dir.is_dir():
        files = sorted([p for p in project_results_dir.glob("*_activity.csv") if p.is_file()])
    for fp in files:
        try:
            yield pd.read_csv(fp, usecols=usecols)
        except Exception:
            continue


def extract_sample_hashes(
    step5_path: str,
    project_results_dir: str,
    core_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """For each (project_name, contributor_email), find one commit hash from Step 5.
    Returns columns: project_name, contributor_email, sample_commit_hash.
    """
    usecols = ["project_name", "contributor_email", "commit_hashes"]
    chunks = _iter_step5_chunks(Path(step5_path), Path(project_results_dir), usecols)

    target_pairs: Set[Tuple[str, str]] = set(
        (r["project_name"], r["contributor_email"]) for _, r in core_pairs.iterrows()
    )
    rows: dict[Tuple[str, str], str] = {}

    for chunk in chunks:
        # Filter to only needed pairs
        mask_pairs = list(zip(chunk["project_name"], chunk["contributor_email"]))
        selected = chunk[[p in target_pairs for p in mask_pairs]]
        if selected.empty:
            continue
        # Parse commit_hashes (JSON-like list stored as string)
        for _, row in selected.iterrows():
            key = (row["project_name"], row["contributor_email"])
            if key in rows:
                continue
            hashes = row.get("commit_hashes")
            if pd.isna(hashes):
                continue
            try:
                cand_list = literal_eval(str(hashes))
                if isinstance(cand_list, list) and len(cand_list) > 0:
                    rows[key] = str(cand_list[0])
                    continue
            except Exception:
                pass
            s = str(hashes).strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            cand = s.split(",")[0].strip().strip('"').strip("'")
            if cand:
                rows[key] = cand

    out = [
        {"project_name": k[0], "contributor_email": k[1], "sample_commit_hash": v}
        for k, v in rows.items()
    ]
    return pd.DataFrame(out)


def lookup_in_master(master_path: str, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Join sample hashes to master commits to get original author identity."""
    if sample_df.empty:
        return pd.DataFrame(columns=["project_name", "commit_hash", "author_name", "author_email"])

    cols = ["project_name", "commit_hash", "author_name", "author_email"]
    found: List[pd.DataFrame] = []
    wanted: Set[str] = set(sample_df["sample_commit_hash"].dropna().astype(str))
    for chunk in pd.read_csv(master_path, usecols=cols, chunksize=500_000):
        sub = chunk[chunk["commit_hash"].astype(str).isin(wanted)]
        if not sub.empty:
            found.append(sub.copy())
    if not found:
        return pd.DataFrame(columns=cols)
    return pd.concat(found, ignore_index=True)


def fallback_match_by_author_email(
    master_path: str,
    core_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fallback when Step 5 commit hashes are missing: join by (project_name, contributor_email)
    to master commits (project_name, author_email). Returns rows with a representative
    commit per pair when available.
    """
    if core_pairs.empty:
        return pd.DataFrame(columns=[
            "project_name", "contributor_email", "sample_commit_hash",
            "original_author_name", "original_author_email"
        ])

    # Build lowercase keys for robust matching
    pairs = core_pairs.copy()
    pairs["project_name_lc"] = pairs["project_name"].astype(str)
    pairs["contributor_email_lc"] = pairs["contributor_email"].astype(str).str.lower()

    # We'll collect first seen commit per (project_name_lc, author_email_lc)
    cols = ["project_name", "commit_hash", "author_name", "author_email"]
    mapping: dict[tuple[str, str], tuple[str, str, str]] = {}
    for chunk in pd.read_csv(master_path, usecols=cols, chunksize=500_000):
        chunk["project_name_lc"] = chunk["project_name"].astype(str)
        chunk["author_email_lc"] = chunk["author_email"].astype(str).str.lower()
        # Only keep keys we care about to reduce memory footprint
        keys_wanted = set(zip(pairs["project_name_lc"], pairs["contributor_email_lc"]))
        mask = [ (pn, ae) not in mapping and (pn, ae) in keys_wanted for pn, ae in zip(chunk["project_name_lc"], chunk["author_email_lc"]) ]
        sub = chunk[mask]
        for _, r in sub.iterrows():
            key = (r["project_name_lc"], r["author_email_lc"])
            if key not in mapping:
                mapping[key] = (
                    str(r["commit_hash"]),
                    str(r["author_name"]) if pd.notna(r["author_name"]) else "",
                    str(r["author_email"]) if pd.notna(r["author_email"]) else "",
                )
        # Early exit if we've filled all
        if len(mapping) >= len(keys_wanted):
            break

    if not mapping:
        return pd.DataFrame(columns=[
            "project_name", "contributor_email", "sample_commit_hash",
            "original_author_name", "original_author_email"
        ])

    rows = []
    for _, r in pairs.iterrows():
        key = (r["project_name_lc"], r["contributor_email_lc"])
        if key in mapping:
            commit_hash, author_name, author_email = mapping[key]
            rows.append({
                "project_name": r["project_name"],
                "contributor_email": r["contributor_email"],
                "sample_commit_hash": commit_hash,
                "original_author_name": author_name,
                "original_author_email": author_email,
            })

    return pd.DataFrame(rows)


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filtered core list and optional identity mapping")
    parser.add_argument("--step6", default=DEFAULT_STEP6)
    parser.add_argument("--step5", default=DEFAULT_STEP5)
    parser.add_argument("--step5_project_results", default=DEFAULT_STEP5_PROJ_DIR)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--out_filtered", default=OUT_FILTERED)
    parser.add_argument("--out_identity", default=OUT_IDENTITY)
    parser.add_argument("--max_weeks_to_core", type=int, default=208, help="Max weeks to core (default 4 years ~ 208 weeks)")
    parser.add_argument("--include_identity", action="store_true", help="Also produce identity mapping via commit hash lookup")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of core pairs (for testing)")
    args = parser.parse_args()

    # Step 1: Filter core contributors from Step 6
    core_df = load_filtered_core_list(
        step6_path=args.step6,
        max_weeks_to_core=args.max_weeks_to_core,
        exclude_instant_core=True,
    )
    if args.limit:
        core_df = core_df.head(args.limit).copy()

    ensure_parent(args.out_filtered)
    core_df.to_csv(args.out_filtered, index=False)
    print(f"Saved filtered core list: {args.out_filtered} ({len(core_df)} rows)")

    if not args.include_identity or core_df.empty:
        return

    # Step 2: Extract one sample commit hash per (project, contributor) from Step 5
    pairs = core_df[["project_name", "contributor_email"]].drop_duplicates()
    samples = extract_sample_hashes(args.step5, args.step5_project_results, pairs)
    print(f"Sample hashes found: {len(samples)} / {len(pairs)}")

    if samples.empty:
        print("No sample hashes found in Step 5 data; skipping identity mapping.")
        return

    # Step 3: Lookup in master commits to get original identity via commit hash
    merged = pd.DataFrame(columns=[
        "project_name", "contributor_email", "sample_commit_hash",
        "original_author_name", "original_author_email"
    ])
    if not samples.empty:
        master_hits = lookup_in_master(args.master, samples)
        if not master_hits.empty:
            samples["sample_commit_hash"] = samples["sample_commit_hash"].astype(str)
            master_hits["commit_hash"] = master_hits["commit_hash"].astype(str)
            # Avoid duplicate project_name columns to keep project_name from samples
            if "project_name" in master_hits.columns:
                master_hits = master_hits.drop(columns=["project_name"], errors="ignore")
            merged = samples.merge(
                master_hits,
                left_on=["sample_commit_hash"],
                right_on=["commit_hash"],
                how="left",
            )
            merged = merged.rename(columns={
                "author_name": "original_author_name",
                "author_email": "original_author_email",
            })
            merged = merged[[
                "project_name",
                "contributor_email",
                "sample_commit_hash",
                "original_author_name",
                "original_author_email",
            ]]

    # Step 3b: Fallback matching by author_email when commit hashes are not available
    missing_pairs = pairs
    if not merged.empty:
        matched_keys = set(zip(merged["project_name"], merged["contributor_email"]))
        missing_pairs = pairs[~pairs.apply(lambda r: (r["project_name"], r["contributor_email"]) in matched_keys, axis=1)]

    if not missing_pairs.empty:
        fb = fallback_match_by_author_email(args.master, missing_pairs)
        if not fb.empty:
            if merged.empty:
                merged = fb
            else:
                merged = pd.concat([merged, fb], ignore_index=True)

    ensure_parent(args.out_identity)
    merged.to_csv(args.out_identity, index=False)
    print(f"Saved identity mapping: {args.out_identity} ({len(merged)} rows)")

    # Also enrich the filtered core list with canonical identity columns
    id_cols = [
        "project_name",
        "contributor_email",
        "sample_commit_hash",
        "original_author_name",
        "original_author_email",
    ]
    addable = merged[id_cols].copy() if not merged.empty else pd.DataFrame(columns=id_cols)

    def infer_username_from_email(email: Optional[str]) -> Optional[str]:
        if email is None or pd.isna(email):
            return None
        s = str(email).strip().strip("\"").strip("'")
        if "users.noreply.github.com" in s:
            local = s.split("@", 1)[0]
            if "+" in local:
                cand = local.split("+", 1)[1]
            else:
                cand = local
            cand = cand.strip()
            return cand or None
        return None

    core_enriched = core_df.merge(addable, on=["project_name", "contributor_email"], how="left")
    # Add resolved_username inferred from canonical email, fallback to normalized contributor email if noreply
    core_enriched["resolved_username"] = core_enriched["original_author_email"].apply(infer_username_from_email)
    core_enriched.loc[core_enriched["resolved_username"].isna(), "resolved_username"] = (
        core_enriched.loc[core_enriched["resolved_username"].isna(), "contributor_email"].apply(infer_username_from_email)
    )

    core_enriched.to_csv(args.out_filtered, index=False)
    print(f"Updated filtered core list with identity columns: {args.out_filtered} ({len(core_enriched)} rows)")


if __name__ == "__main__":
    main()


