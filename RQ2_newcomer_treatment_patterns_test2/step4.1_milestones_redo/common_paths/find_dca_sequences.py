#!/usr/bin/env python3

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def normalize_sequence(raw_sequence: str) -> str:
    tokens = [token.strip() for token in raw_sequence.split(",") if token.strip() != ""]
    return ",".join(tokens)


def read_sequences(csv_path: str) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            project_type = str(record.get("project_type", "")).strip()
            contributor = str(record.get("contributor_email", "")).strip()
            sequence_raw = str(record.get("sequence", "")).strip()
            if not sequence_raw:
                continue
            rows.append((project_type, contributor, normalize_sequence(sequence_raw)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Find sequences containing DCA (DirectCommitAccessProxy)")
    parser.add_argument(
        "--sequences",
        dest="sequences",
        default=os.path.join(os.path.dirname(__file__), "..", "sequences_all.csv"),
        help="Path to sequences CSV (default: sequences_all.csv)",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        default=os.path.dirname(__file__),
        help="Output directory for results",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        choices=["OSS", "OSS4SG", "ALL"],
        default="OSS4SG",
        help="Filter by project_type (default: OSS4SG)",
    )
    args = parser.parse_args()

    rows = read_sequences(os.path.abspath(args.sequences))

    # Filter to cohort
    if args.cohort == "ALL":
        cohort_rows = rows
    else:
        cohort_rows = [(ptype, contributor, seq) for (ptype, contributor, seq) in rows if ptype == args.cohort]

    # Keep only sequences containing DCA
    contains_dca = [(ptype, contributor, seq) for (ptype, contributor, seq) in cohort_rows if "DirectCommitAccessProxy" in seq]

    # Group by distinct sequence
    by_sequence: Dict[str, List[str]] = defaultdict(list)
    for _ptype, contributor, seq in contains_dca:
        by_sequence[seq].append(contributor)

    total = len(contains_dca)

    os.makedirs(args.outdir, exist_ok=True)

    # Summary with counts and percentages
    summary_path = os.path.join(args.outdir, f"dca_sequences_{args.cohort.lower()}_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "sequence", "count", "percentage"],
        )
        writer.writeheader()
        for idx, (sequence, contributors) in enumerate(sorted(by_sequence.items(), key=lambda kv: len(kv[1]), reverse=True), start=1):
            pct = (len(contributors) / total * 100) if total else 0.0
            writer.writerow({"rank": str(idx), "sequence": sequence, "count": str(len(contributors)), "percentage": f"{pct:.2f}"})

    # Detailed list of contributors per sequence
    details_path = os.path.join(args.outdir, f"dca_sequences_{args.cohort.lower()}_contributors.csv")
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sequence", "contributor_email"],
        )
        writer.writeheader()
        for sequence, contributors in sorted(by_sequence.items(), key=lambda kv: len(kv[1]), reverse=True):
            for contributor in contributors:
                writer.writerow({"sequence": sequence, "contributor_email": contributor})

    print(summary_path)
    print(details_path)


if __name__ == "__main__":
    main()





