#!/usr/bin/env python3
"""
Enrich core_commit_identity.csv with resolved GitHub usernames using commit hashes.

Strategy per row:
- If original_author_email is a GitHub noreply, parse username from it
- Else call: GET /repos/{project}/commits/{sample_commit_hash} and use response.author.login
- If author is null, fallback to commit.author.email noreply parsing

Outputs a CSV with an added column: resolved_username
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


@dataclass
class TokenState:
    token: str
    remaining: int = 5000
    reset_time: Optional[datetime] = None


class IdentityEnricher:
    def __init__(self, tokens: List[str]):
        self.tokens = [TokenState(t) for t in tokens]
        self.current_idx = 0
        self.rest_base = "https://api.github.com"

    def _pick_token(self) -> str:
        attempts = 0
        while attempts < len(self.tokens) * 2:
            tok = self.tokens[self.current_idx]
            # If reset time passed, refresh remaining
            if tok.reset_time and datetime.now() > tok.reset_time:
                tok.remaining = 5000
                tok.reset_time = None
            if tok.remaining > 0:
                return tok.token
            self.current_idx = (self.current_idx + 1) % len(self.tokens)
            attempts += 1
        # All exhausted: wait minimal 60s
        time.sleep(60)
        return self.tokens[0].token

    def _update_limits(self, idx: int, resp: requests.Response):
        try:
            if 'x-ratelimit-remaining' in resp.headers:
                self.tokens[idx].remaining = int(resp.headers['x-ratelimit-remaining'])
            if 'x-ratelimit-reset' in resp.headers:
                self.tokens[idx].reset_time = datetime.fromtimestamp(int(resp.headers['x-ratelimit-reset']))
        except Exception:
            pass

    @staticmethod
    def _username_from_noreply(email: Optional[str]) -> Optional[str]:
        if not email or not isinstance(email, str):
            return None
        s = email.strip().strip('"').strip("'")
        if 'users.noreply.github.com' in s:
            local = s.split('@', 1)[0]
            return local.split('+', 1)[1] if '+' in local else local
        return None

    def _login_from_commit(self, project: str, sha: str) -> Optional[str]:
        token = self._pick_token()
        idx = self.current_idx
        headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github+json'}
        url = f"{self.rest_base}/repos/{project}/commits/{sha}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except Exception:
            return None
        self._update_limits(idx, resp)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Prefer linked GitHub account
        author = data.get('author')
        if isinstance(author, dict) and author.get('login'):
            return str(author['login']).strip()
        # Fallback to noreply from commit header
        commit = data.get('commit', {}) if isinstance(data, dict) else {}
        ca = commit.get('author', {}) if isinstance(commit, dict) else {}
        email = ca.get('email')
        return self._username_from_noreply(email)

    def enrich(self, input_csv: str, output_csv: str, sample: Optional[int] = None) -> dict:
        df = pd.read_csv(input_csv)
        if sample:
            df = df.head(sample).copy()

        resolved: List[Optional[str]] = []
        processed = 0
        noreply_hits = 0
        api_hits = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Enriching'):
            # 1) From canonical email if noreply
            rn = self._username_from_noreply(row.get('original_author_email'))
            if rn:
                resolved.append(rn)
                noreply_hits += 1
                processed += 1
                continue
            # 2) From commit endpoint
            project = str(row['project_name']).strip()
            sha = str(row['sample_commit_hash']).strip()
            login = self._login_from_commit(project, sha)
            if login:
                resolved.append(login)
                api_hits += 1
            else:
                resolved.append(None)
            processed += 1

        out = df.copy()
        out['resolved_username'] = resolved
        out.to_csv(output_csv, index=False)
        return {
            'total': len(df),
            'resolved_from_noreply': noreply_hits,
            'resolved_from_api': api_hits,
            'unresolved': len(df) - noreply_hits - api_hits,
            'output': output_csv
        }


def main():
    p = argparse.ArgumentParser(description='Enrich identity CSV with GitHub usernames via commit hashes')
    p.add_argument('--tokens', required=True, help='Path to token file or comma-separated tokens')
    p.add_argument('--input', required=True, help='Path to core_commit_identity.csv')
    p.add_argument('--output', required=True, help='Path to write enriched CSV with resolved_username')
    p.add_argument('--sample', type=int, default=None, help='Optional sample size for testing')
    args = p.parse_args()

    if Path(args.tokens).exists():
        tokens = [t.strip() for t in Path(args.tokens).read_text().splitlines() if t.strip() and not t.strip().startswith('#')]
    else:
        tokens = [t.strip() for t in args.tokens.split(',') if t.strip()]
    if not tokens:
        print('No tokens provided', file=sys.stderr)
        sys.exit(1)

    enricher = IdentityEnricher(tokens)
    stats = enricher.enrich(args.input, args.output, args.sample)
    print('\nSummary:')
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()


