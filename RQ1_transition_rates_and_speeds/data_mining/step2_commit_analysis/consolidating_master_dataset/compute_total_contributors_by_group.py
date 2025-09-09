#!/usr/bin/env python3
import sys
import csv


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_total_contributors_by_group.py <master_commits_dataset.csv>")
        sys.exit(1)
    path = sys.argv[1]

    authors_oss = set()
    authors_oss4sg = set()
    authors_all = set()

    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        required = ['project_type', 'author_email']
        for col in required:
            if col not in reader.fieldnames:
                print(f"ERROR: missing required column {col}. Columns: {reader.fieldnames}")
                sys.exit(2)
        for row in reader:
            pt = (row['project_type'] or '').strip()
            email = (row['author_email'] or '').strip().lower()
            if not email:
                continue
            authors_all.add(email)
            if pt == 'OSS':
                authors_oss.add(email)
            elif pt == 'OSS4SG':
                authors_oss4sg.add(email)

    overlap = authors_oss & authors_oss4sg
    only_oss = authors_oss - authors_oss4sg
    only_oss4sg = authors_oss4sg - authors_oss

    print("Unique contributors by group (based on author_email):")
    print(f"  OSS contributors:        {len(authors_oss)}")
    print(f"  OSS4SG contributors:     {len(authors_oss4sg)}")
    print(f"  Total unique contributors (union): {len(authors_all)}")
    print("")
    print("Overlap diagnostics:")
    print(f"  Contributed to both:     {len(overlap)}")
    print(f"  Only OSS:                {len(only_oss)}")
    print(f"  Only OSS4SG:             {len(only_oss4sg)}")


if __name__ == '__main__':
    main()

