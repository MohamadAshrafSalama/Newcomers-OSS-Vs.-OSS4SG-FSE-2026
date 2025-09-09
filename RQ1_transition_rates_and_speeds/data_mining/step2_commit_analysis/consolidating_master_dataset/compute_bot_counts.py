#!/usr/bin/env python3
import sys
import csv
from collections import Counter


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_bot_counts.py <master_commits_dataset.csv>")
        sys.exit(1)
    path = sys.argv[1]

    total = 0
    bot_author_rows = 0
    bot_message_rows = 0
    bot_emails = set()
    bot_names = set()
    by_type_total = Counter()
    by_type_bot = Counter()

    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        required = ['project_type', 'author_name', 'author_email', 'commit_message']
        for col in required:
            if col not in reader.fieldnames:
                print(f"ERROR: missing required column {col}. Columns: {reader.fieldnames}")
                sys.exit(2)

        for row in reader:
            total += 1
            pt = (row['project_type'] or '').strip()
            by_type_total[pt] += 1

            name = (row['author_name'] or '').lower()
            email = (row['author_email'] or '').lower()
            msg = (row['commit_message'] or '').lower()

            is_bot_author = ('bot' in name) or ('bot' in email)
            is_bot_message = ('bot' in msg)

            if is_bot_author:
                bot_author_rows += 1
                by_type_bot[pt] += 1
                if email:
                    bot_emails.add(email)
                if name:
                    bot_names.add(name)
            if is_bot_message:
                bot_message_rows += 1

    print("Bot indicators (case-insensitive) in author fields:")
    pct = 100.0 * bot_author_rows / total if total else 0.0
    print(f"  Commits with bot in author name/email: {bot_author_rows:,} / {total:,} ({pct:.2f}%)")
    print(f"  Unique bot author emails: {len(bot_emails):,}")
    print(f"  Unique bot author names:  {len(bot_names):,}")

    print("\nBreakdown by project_type (author contains 'bot'):")
    for pt in sorted(by_type_total.keys()):
        n = by_type_total[pt]
        b = by_type_bot[pt]
        bpct = 100.0 * b / n if n else 0.0
        print(f"  {pt}: {b:,} / {n:,} ({bpct:.2f}%)")

    print("\nReference (commit_message contains 'bot'):")
    mpct = 100.0 * bot_message_rows / total if total else 0.0
    print(f"  Commits with bot in message: {bot_message_rows:,} / {total:,} ({mpct:.2f}%)")


if __name__ == '__main__':
    main()

