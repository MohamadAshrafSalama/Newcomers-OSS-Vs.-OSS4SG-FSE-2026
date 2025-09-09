#!/usr/bin/env python3

import argparse
import csv
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Optional


def normalize_sequence(raw_sequence: str) -> str:
    """Return a canonical comma-separated sequence without internal spaces.

    Example: "START, FirstMergedPullRequest , END" -> "START,FirstMergedPullRequest,END"
    """
    tokens = [token.strip() for token in raw_sequence.split(",") if token.strip() != ""]
    return ",".join(tokens)


def parse_prefix(prefix: Optional[str]) -> Optional[List[str]]:
    if prefix is None or prefix.strip() == "":
        return None
    return [token.strip() for token in prefix.split(",") if token.strip() != ""]


def read_sequences(csv_path: str) -> List[Tuple[str, str]]:
    """Read (project_type, normalized_sequence) tuples from a sequences CSV.

    The CSV is expected to have the columns: project_name, project_type, contributor_email, sequence
    Extra columns are ignored.
    """
    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            project_type = str(record.get("project_type", "")).strip()
            sequence_raw = str(record.get("sequence", "")).strip()
            if sequence_raw == "":
                # Skip empty sequence rows just in case
                continue
            rows.append((project_type, normalize_sequence(sequence_raw)))
    return rows


def counter_to_ranked_rows(counter: Counter, total: int, top_k: int) -> List[Dict[str, str]]:
    """Convert a Counter of sequences to ranked rows with percent.

    Returns a list of dicts with keys: rank, sequence, count, percentage
    """
    ranked: List[Dict[str, str]] = []
    for idx, (sequence, count) in enumerate(counter.most_common(top_k), start=1):
        percentage = (count / total) * 100 if total > 0 else 0.0
        ranked.append(
            {
                "rank": str(idx),
                "sequence": sequence,
                "count": str(count),
                "percentage": f"{percentage:.2f}",
            }
        )
    return ranked


def write_csv(rows: Iterable[Dict[str, str]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows_list = list(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "sequence", "count", "percentage"])
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def compute_top_common_paths(
    rows: List[Tuple[str, str]],
    project_type_filter: Optional[str],
    top_k: int,
    exclude_empty: bool,
    prefix_tokens: Optional[List[str]] = None,
) -> Tuple[Counter, int, int, int, int]:
    """Compute top-k common full sequences for an optional project_type filter.

    - project_type_filter=None for Overall
    - Otherwise, filter to rows where project_type == filter
    Returns (counter, total_after, excluded_count, total_before).
    """
    if project_type_filter is None:
        filtered_sequences = [sequence for (_ptype, sequence) in rows]
    else:
        filtered_sequences = [sequence for (ptype, sequence) in rows if ptype == project_type_filter]

    total_before = len(filtered_sequences)
    if exclude_empty:
        filtered_sequences = [seq for seq in filtered_sequences if seq != "START,END"]
    after_empty = len(filtered_sequences)
    excluded_empty = total_before - after_empty

    excluded_prefix = 0
    if prefix_tokens:
        prefix_len = len(prefix_tokens)
        kept = []
        for seq in filtered_sequences:
            tokens = seq.split(",") if seq else []
            if len(tokens) >= prefix_len and tokens[:prefix_len] == prefix_tokens:
                kept.append(seq)
        excluded_prefix = after_empty - len(kept)
        filtered_sequences = kept

    total_after = len(filtered_sequences)

    return Counter(filtered_sequences), total_after, excluded_empty, excluded_prefix, total_before


def process_one_cohort(
    input_csv: str,
    out_dir: str,
    suffix: str,
    top_k: int,
    exclude_empty: bool,
    prefix_tokens: Optional[List[str]] = None,
    prefix_tag: Optional[str] = None,
) -> List[str]:
    """Process one cohort CSV and write three outputs (overall, oss, oss4sg).

    Returns a list of generated file paths.
    """
    rows = read_sequences(input_csv)

    outputs: List[str] = []

    name_suffix = f"_{prefix_tag}" if prefix_tokens and prefix_tag else ("" if not prefix_tokens else "_prefix")

    # Overall
    counter_all, total_all, excluded_empty_all, excluded_prefix_all, before_all = compute_top_common_paths(
        rows, None, top_k, exclude_empty, prefix_tokens
    )
    out_all = os.path.join(out_dir, f"common_paths_overall{name_suffix}{suffix}.csv")
    write_csv(counter_to_ranked_rows(counter_all, total_all, top_k), out_all)
    outputs.append(out_all)

    # OSS only
    counter_oss, total_oss, excluded_empty_oss, excluded_prefix_oss, before_oss = compute_top_common_paths(
        rows, "OSS", top_k, exclude_empty, prefix_tokens
    )
    out_oss = os.path.join(out_dir, f"common_paths_oss{name_suffix}{suffix}.csv")
    write_csv(counter_to_ranked_rows(counter_oss, total_oss, top_k), out_oss)
    outputs.append(out_oss)

    # OSS4SG only
    counter_oss4sg, total_oss4sg, excluded_empty_oss4sg, excluded_prefix_oss4sg, before_oss4sg = compute_top_common_paths(
        rows, "OSS4SG", top_k, exclude_empty, prefix_tokens
    )
    out_oss4sg = os.path.join(out_dir, f"common_paths_oss4sg{name_suffix}{suffix}.csv")
    write_csv(counter_to_ranked_rows(counter_oss4sg, total_oss4sg, top_k), out_oss4sg)
    outputs.append(out_oss4sg)

    # Write filter counts
    excl_path = os.path.join(out_dir, f"filtered_counts{name_suffix}{suffix}.csv")
    with open(excl_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stratum",
                "excluded_empty_sequences",
                "excluded_prefix_sequences",
                "total_before",
                "total_after",
                "excluded_percentage",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stratum": "Overall",
                "excluded_empty_sequences": str(excluded_empty_all),
                "excluded_prefix_sequences": str(excluded_prefix_all),
                "total_before": str(before_all),
                "total_after": str(total_all),
                "excluded_percentage": f"{(((excluded_empty_all + excluded_prefix_all) / before_all) * 100) if before_all else 0.0:.2f}",
            }
        )
        writer.writerow(
            {
                "stratum": "OSS",
                "excluded_empty_sequences": str(excluded_empty_oss),
                "excluded_prefix_sequences": str(excluded_prefix_oss),
                "total_before": str(before_oss),
                "total_after": str(total_oss),
                "excluded_percentage": f"{(((excluded_empty_oss + excluded_prefix_oss) / before_oss) * 100) if before_oss else 0.0:.2f}",
            }
        )
        writer.writerow(
            {
                "stratum": "OSS4SG",
                "excluded_empty_sequences": str(excluded_empty_oss4sg),
                "excluded_prefix_sequences": str(excluded_prefix_oss4sg),
                "total_before": str(before_oss4sg),
                "total_after": str(total_oss4sg),
                "excluded_percentage": f"{(((excluded_empty_oss4sg + excluded_prefix_oss4sg) / before_oss4sg) * 100) if before_oss4sg else 0.0:.2f}",
            }
        )
    outputs.append(excl_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute top-K most common full milestone paths.")
    parser.add_argument(
        "--sequences-all",
        dest="sequences_all",
        default=os.path.join(os.path.dirname(__file__), "..", "sequences_all.csv"),
        help="Path to sequences_all.csv",
    )
    parser.add_argument(
        "--sequences-min15",
        dest="sequences_min15",
        default=os.path.join(os.path.dirname(__file__), "..", "sequences_min15.csv"),
        help="Path to sequences_min15.csv",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        default=os.path.dirname(__file__),
        help="Output directory for common paths CSVs (defaults to this script's directory)",
    )
    parser.add_argument("--top", dest="top", type=int, default=5, help="Top-K sequences to output (default 5)")
    parser.add_argument(
        "--exclude-empty",
        dest="exclude_empty",
        action="store_true",
        help="Exclude sequences with no milestones (exactly 'START,END') and report how many were excluded",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        default=None,
        help="Optional comma-separated prefix tokens to filter sequences (e.g., 'START,FirstMergedPullRequest')",
    )
    parser.add_argument(
        "--prefix-tag",
        dest="prefix_tag",
        default=None,
        help="Tag to append to output filenames when --prefix is used (e.g., 'start_fmpr')",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    generated: List[str] = []

    prefix_tokens = parse_prefix(args.prefix)

    # Full cohort
    generated += process_one_cohort(
        input_csv=os.path.abspath(args.sequences_all),
        out_dir=out_dir,
        suffix="",
        top_k=args.top,
        exclude_empty=args.exclude_empty,
        prefix_tokens=prefix_tokens,
        prefix_tag=args.prefix_tag,
    )

    # Min15 cohort (if available)
    if os.path.exists(args.sequences_min15):
        generated += process_one_cohort(
            input_csv=os.path.abspath(args.sequences_min15),
            out_dir=out_dir,
            suffix="_min15",
            top_k=args.top,
            exclude_empty=args.exclude_empty,
            prefix_tokens=prefix_tokens,
            prefix_tag=args.prefix_tag,
        )

    # Print generated paths for convenience
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()


