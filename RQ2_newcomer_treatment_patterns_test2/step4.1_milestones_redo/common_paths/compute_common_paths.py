#!/usr/bin/env python3

import argparse
import csv
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple


def normalize_sequence(raw_sequence: str) -> str:
    """Return a canonical comma-separated sequence without internal spaces.

    Example: "START, FirstMergedPullRequest , END" -> "START,FirstMergedPullRequest,END"
    """
    tokens = [token.strip() for token in raw_sequence.split(",") if token.strip() != ""]
    return ",".join(tokens)


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
    rows: List[Tuple[str, str]], project_type_filter: str | None, top_k: int
) -> Tuple[Counter, int]:
    """Compute top-k common full sequences for an optional project_type filter.

    - project_type_filter=None for Overall
    - Otherwise, filter to rows where project_type == filter
    Returns counter and the total number of sequences considered.
    """
    if project_type_filter is None:
        filtered_sequences = [sequence for (_ptype, sequence) in rows]
    else:
        filtered_sequences = [sequence for (ptype, sequence) in rows if ptype == project_type_filter]

    total = len(filtered_sequences)
    return Counter(filtered_sequences), total


def process_one_cohort(
    input_csv: str,
    out_dir: str,
    suffix: str,
    top_k: int,
) -> List[str]:
    """Process one cohort CSV and write three outputs (overall, oss, oss4sg).

    Returns a list of generated file paths.
    """
    rows = read_sequences(input_csv)

    outputs: List[str] = []

    # Overall
    counter_all, total_all = compute_top_common_paths(rows, None, top_k)
    out_all = os.path.join(out_dir, f"common_paths_overall{suffix}.csv")
    write_csv(counter_to_ranked_rows(counter_all, total_all, top_k), out_all)
    outputs.append(out_all)

    # OSS only
    counter_oss, total_oss = compute_top_common_paths(rows, "OSS", top_k)
    out_oss = os.path.join(out_dir, f"common_paths_oss{suffix}.csv")
    write_csv(counter_to_ranked_rows(counter_oss, total_oss, top_k), out_oss)
    outputs.append(out_oss)

    # OSS4SG only
    counter_oss4sg, total_oss4sg = compute_top_common_paths(rows, "OSS4SG", top_k)
    out_oss4sg = os.path.join(out_dir, f"common_paths_oss4sg{suffix}.csv")
    write_csv(counter_to_ranked_rows(counter_oss4sg, total_oss4sg, top_k), out_oss4sg)
    outputs.append(out_oss4sg)

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
    args = parser.parse_args()

    out_dir = os.path.abspath(args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    generated: List[str] = []

    # Full cohort
    generated += process_one_cohort(
        input_csv=os.path.abspath(args.sequences_all), out_dir=out_dir, suffix="", top_k=args.top
    )

    # Min15 cohort (if available)
    if os.path.exists(args.sequences_min15):
        generated += process_one_cohort(
            input_csv=os.path.abspath(args.sequences_min15), out_dir=out_dir, suffix="_min15", top_k=args.top
        )

    # Print generated paths for convenience
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()


