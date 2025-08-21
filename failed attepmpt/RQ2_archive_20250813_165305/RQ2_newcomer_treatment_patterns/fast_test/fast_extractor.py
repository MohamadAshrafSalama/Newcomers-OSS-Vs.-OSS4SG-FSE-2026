#!/usr/bin/env python3
"""
FAST GitHub Treatment Data Extractor for RQ2 (Standalone Test)
- Uses GraphQL to avoid Search API caps
- Parallel workers (one per token, up to 8)
- Resumable via state file

Inputs:
- Contributors CSV: use enriched core list with columns:
  project_name, contributor_email, first_commit_date, first_core_date,
  resolved_username (optional), original_author_email (optional), sample_commit_hash (optional)

Outputs:
- JSON files under output_dir/cache (one per contributor)
- extraction_state.pkl, username_cache.json under output_dir
"""

from __future__ import annotations

import json
import time
import pickle
import hashlib
import logging
import sys
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from queue import Queue

import pandas as pd
from tqdm import tqdm
import requests
from dataclasses import dataclass, field


@dataclass
class TokenStatus:
    token: str
    worker_id: int
    graphql_remaining: int = 5000
    reset_time: Optional[datetime] = None
    total_requests: int = 0


@dataclass
class WorkItem:
    row: pd.Series
    contributor_id: str


class FastGitHubExtractor:
    def __init__(self, tokens: List[str], output_dir: str, max_workers: Optional[int] = None) -> None:
        self.max_workers = min(max_workers or len(tokens), len(tokens), 8)
        self.tokens: List[TokenStatus] = [TokenStatus(token=tokens[i], worker_id=i) for i in range(self.max_workers)]

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.state_file = self.output_dir / "extraction_state.pkl"
        self.failed_dir = self.output_dir / "failed"
        self.failed_dir.mkdir(exist_ok=True)

        self.state_lock = threading.Lock()
        self.username_cache_lock = threading.Lock()
        self.state = self._load_state()
        self.username_cache: Dict[str, str] = {}
        self._load_username_cache()

        self.graphql_url = "https://api.github.com/graphql"
        self.rest_url = "https://api.github.com"

        self._setup_logging()
        self.logger.info(f"Initialized fast extractor with {self.max_workers} workers")

    def _setup_logging(self) -> None:
        fmt = '%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
        fh = logging.FileHandler(self.output_dir / "extraction.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt))
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(fmt))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {
            'processed_contributors': set(),
            'failed_contributors': {},
            'skipped_contributors': {},
            'processed_count': 0,
            'failed_count': 0,
            'skipped_count': 0,
            'start_time': datetime.now().isoformat(),
        }

    def _save_state(self) -> None:
        with self.state_lock:
            self.state['last_save'] = datetime.now().isoformat()
            with open(self.state_file, 'wb') as f:
                pickle.dump(self.state, f)

    def _load_username_cache(self) -> None:
        cache_file = self.output_dir / "username_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.username_cache = json.load(f)
            except Exception:
                self.username_cache = {}

    def _save_username_cache(self) -> None:
        with self.username_cache_lock:
            with open(self.output_dir / "username_cache.json", 'w') as f:
                json.dump(self.username_cache, f)

    def _graphql(self, query: str, variables: Dict, token: TokenStatus) -> Optional[Dict]:
        headers = {'Authorization': f'Bearer {token.token}', 'Content-Type': 'application/json'}
        try:
            resp = requests.post(self.graphql_url, headers=headers, json={'query': query, 'variables': variables}, timeout=30)
            token.total_requests += 1
            if resp.status_code == 200:
                data = resp.json()
                return data.get('data')
            elif resp.status_code == 403:
                # soft backoff
                time.sleep(60)
                return None
            else:
                return None
        except Exception:
            return None

    def _verify_user(self, username: str, token: TokenStatus) -> bool:
        q = """
        query($login: String!) { user(login: $login) { login } }
        """
        data = self._graphql(q, {'login': username}, token)
        return bool(data and data.get('user'))

    @staticmethod
    def _username_from_email(email: Optional[str]) -> Optional[str]:
        if not email or not isinstance(email, str):
            return None
        s = email.strip().strip('"').strip("'")
        if 'users.noreply.github.com' in s:
            local = s.split('@', 1)[0]
            cand = local.split('+', 1)[1] if '+' in local else local
            cand = cand.strip()
            return cand or None
        return None

    def _resolve_username(self, row: pd.Series, token: TokenStatus) -> Optional[str]:
        # 1) resolved_username
        if 'resolved_username' in row and pd.notna(row['resolved_username']) and str(row['resolved_username']).strip() != '':
            return str(row['resolved_username']).strip()
        # 2) derive from canonical email
        if 'original_author_email' in row and pd.notna(row['original_author_email']):
            cand = self._username_from_email(str(row['original_author_email']))
            if cand and self._verify_user(cand, token):
                return cand
        # 3) derive from normalized email
        cand = self._username_from_email(str(row.get('contributor_email', '')))
        if cand and self._verify_user(cand, token):
            return cand
        # 4) fallback via sample commit hash (REST commits endpoint)
        if 'sample_commit_hash' in row and pd.notna(row['sample_commit_hash']) and str(row['sample_commit_hash']).strip() != '':
            sha = str(row['sample_commit_hash']).strip()
            project = str(row['project_name']).strip()
            try:
                resp = requests.get(f"{self.rest_url}/repos/{project}/commits/{sha}", headers={'Authorization': f'Bearer {token.token}'}, timeout=20)
                token.total_requests += 1
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        author = data.get('author')
                        if isinstance(author, dict) and author.get('login'):
                            login = str(author['login']).strip()
                            if login and self._verify_user(login, token):
                                return login
                        # fallback to noreply pattern from commit.author.email
                        commit = data.get('commit', {})
                        ca = commit.get('author', {}) if isinstance(commit, dict) else {}
                        email = ca.get('email')
                        cand2 = self._username_from_email(email)
                        if cand2 and self._verify_user(cand2, token):
                            return cand2
            except Exception:
                pass
        return None

    def _fetch_contributions_by_user(self, owner: str, repo: str, username: str, start_iso: str, end_iso: str, token: TokenStatus) -> tuple[List[Dict], List[Dict]]:
        # Use user.contributionsCollection filtered by time, then pick entries for our repo
        q = """
        query($login: String!, $from: DateTime!, $to: DateTime!) {
          user(login: $login) {
            contributionsCollection(from: $from, to: $to) {
              pullRequestContributionsByRepository(maxRepositories: 25) {
                repository { nameWithOwner }
                contributions(first: 100) {
                  nodes {
                    pullRequest {
                      number title state createdAt closedAt mergedAt additions deletions changedFiles
                      comments(first: 10) { nodes { author { login } createdAt } }
                      reviews(first: 10) { nodes { author { login } state submittedAt } }
                    }
                  }
                }
              }
              issueContributionsByRepository(maxRepositories: 25) {
                repository { nameWithOwner }
                contributions(first: 100) {
                  nodes {
                    issue {
                      number title state createdAt closedAt
                      comments(first: 10) { nodes { author { login } createdAt } }
                    }
                  }
                }
              }
            }
          }
        }
        """
        vars = {'login': username, 'from': start_iso, 'to': end_iso}
        data = self._graphql(q, vars, token)
        prs: List[Dict] = []
        issues: List[Dict] = []
        if not data or not data.get('user'):
            return prs, issues
        coll = data['user'].get('contributionsCollection', {})
        repo_key = f"{owner}/{repo}"
        for block in (coll.get('pullRequestContributionsByRepository') or []):
            if block and block.get('repository', {}).get('nameWithOwner') == repo_key:
                for node in (block.get('contributions', {}).get('nodes') or []):
                    pr = node.get('pullRequest')
                    if pr:
                        prs.append(pr)
        for block in (coll.get('issueContributionsByRepository') or []):
            if block and block.get('repository', {}).get('nameWithOwner') == repo_key:
                for node in (block.get('contributions', {}).get('nodes') or []):
                    issue = node.get('issue')
                    if issue:
                        issues.append(issue)
        return prs, issues

    @staticmethod
    def _compute_metrics(username: str, prs: List[Dict], issues: List[Dict]) -> Dict:
        metrics = {
            'total_prs': len(prs),
            'total_issues': len(issues),
            'total_reviews': 0,
            'total_comments': 0,
            'unique_reviewers': set(),
            'unique_commenters': set(),
            'approval_rate': 0.0,
            'response_times': [],
            'avg_response_hours': None,
        }
        approvals = 0
        total_reviews = 0
        # PRs
        for pr in prs:
            created = pd.to_datetime(pr['createdAt'])
            first_response = None
            for review in (pr.get('reviews', {}).get('nodes') or []):
                if review and review.get('author', {}).get('login') != username:
                    metrics['unique_reviewers'].add(review['author']['login'])
                    total_reviews += 1
                    if review.get('state') == 'APPROVED':
                        approvals += 1
                    if review.get('submittedAt'):
                        rt = pd.to_datetime(review['submittedAt'])
                        if not first_response or rt < first_response:
                            first_response = rt
            for c in (pr.get('comments', {}).get('nodes') or []):
                if c and c.get('author', {}).get('login') != username:
                    metrics['unique_commenters'].add(c['author']['login'])
                    metrics['total_comments'] += 1
                    if c.get('createdAt'):
                        ct = pd.to_datetime(c['createdAt'])
                        if not first_response or ct < first_response:
                            first_response = ct
            if first_response is not None:
                metrics['response_times'].append((first_response - created).total_seconds() / 3600.0)
        # Issues
        for issue in issues:
            created = pd.to_datetime(issue['createdAt'])
            first_response = None
            for c in (issue.get('comments', {}).get('nodes') or []):
                if c and c.get('author', {}).get('login') != username:
                    metrics['unique_commenters'].add(c['author']['login'])
                    metrics['total_comments'] += 1
                    if c.get('createdAt'):
                        ct = pd.to_datetime(c['createdAt'])
                        if not first_response or ct < first_response:
                            first_response = ct
            if first_response is not None:
                metrics['response_times'].append((first_response - created).total_seconds() / 3600.0)
        metrics['total_reviews'] = total_reviews
        if total_reviews > 0:
            metrics['approval_rate'] = approvals / total_reviews
        if metrics['response_times']:
            metrics['avg_response_hours'] = sum(metrics['response_times']) / len(metrics['response_times'])
        metrics['unique_reviewers'] = len(metrics['unique_reviewers'])
        metrics['unique_commenters'] = len(metrics['unique_commenters'])
        return metrics

    def _process_one(self, row: pd.Series, token: TokenStatus) -> bool:
        contributor_id = f"{row['project_name']}_{row['contributor_email']}"
        # Resolve username
        username = self._resolve_username(row, token)
        if not username:
            with self.state_lock:
                self.state['skipped_contributors'][contributor_id] = 'username_resolution_failed'
                self.state['skipped_count'] += 1
            return False
        # Dates
        start_iso = pd.to_datetime(row['first_commit_date']).isoformat()
        end_iso = pd.to_datetime(row['first_core_date']).isoformat()
        owner, repo = str(row['project_name']).split('/')
        prs, issues = self._fetch_contributions_by_user(owner, repo, username, start_iso, end_iso, token)
        data = {
            'username': username,
            'project': row['project_name'],
            'project_type': row.get('project_type', ''),
            'weeks_to_core': row.get('weeks_to_core', None),
            'transition_start': start_iso,
            'transition_end': end_iso,
            'pull_requests': prs,
            'issues': issues,
        }
        data['treatment_metrics'] = self._compute_metrics(username, prs, issues)
        # Save
        safe_name = f"{row['project_name'].replace('/', '_')}_{username}"
        hash_name = hashlib.md5(safe_name.encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"{hash_name}_{safe_name}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        with self.state_lock:
            self.state['processed_contributors'].add(contributor_id)
            self.state['processed_count'] += 1
            if self.state['processed_count'] % 50 == 0:
                self._save_state()
                self._save_username_cache()
        return True

    def _worker(self, q: Queue, token: TokenStatus, pbar: tqdm) -> None:
        while True:
            try:
                item: Optional[WorkItem] = q.get(timeout=1)
            except Exception:
                # empty for a bit
                continue
            if item is None:
                break
            try:
                self._process_one(item.row, token)
            except Exception as e:
                contributor_id = item.contributor_id
                with self.state_lock:
                    self.state['failed_contributors'][contributor_id] = str(e)
                    self.state['failed_count'] += 1
                with open(self.failed_dir / f"{contributor_id.replace('/', '_')}.txt", 'w') as f:
                    f.write(str(e))
            finally:
                pbar.update(1)
                q.task_done()

    def run(self, contributors_csv: str, sample_size: Optional[int] = None) -> None:
        df = pd.read_csv(contributors_csv)
        df['contributor_id'] = df['project_name'] + '_' + df['contributor_email']
        df = df[~df['contributor_id'].isin(self.state['processed_contributors'])]
        if sample_size:
            df = df.head(sample_size)
        total = len(df)
        q: Queue = Queue(maxsize=200)
        workers: List[threading.Thread] = []
        with tqdm(total=total, desc='Overall Progress') as pbar:
            for token in self.tokens:
                t = threading.Thread(target=self._worker, args=(q, token, pbar), name=f"Worker-{token.worker_id}")
                t.start()
                workers.append(t)
            for _, row in df.iterrows():
                q.put(WorkItem(row=row, contributor_id=row['contributor_id']))
            for _ in self.tokens:
                q.put(None)
            for t in workers:
                t.join()
        self._save_state()
        self._save_username_cache()
        print(f"Processed: {self.state['processed_count']}, Skipped: {self.state['skipped_count']}, Failed: {self.state['failed_count']}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description='FAST GraphQL extractor (test)')
    p.add_argument('--tokens', required=True, help='Path to token file or comma-separated tokens')
    p.add_argument('--contributors', required=True, help='Contributors CSV (enriched core list)')
    p.add_argument('--output-dir', default='RQ2_newcomer_treatment_patterns/fast_test/results', help='Output directory')
    p.add_argument('--workers', type=int, default=None, help='Max parallel workers (default: num tokens, max 8)')
    p.add_argument('--sample', type=int, default=None, help='Optional sample size')
    args = p.parse_args()

    if Path(args.tokens).exists():
        with open(args.tokens, 'r') as f:
            tokens = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        tokens = [t.strip() for t in args.tokens.split(',')]
    if not tokens:
        print('No tokens provided!')
        sys.exit(1)

    print(f"FAST extractor: tokens={len(tokens)}, workers={args.workers or len(tokens)}, out={args.output_dir}")
    ex = FastGitHubExtractor(tokens=tokens, output_dir=args.output_dir, max_workers=args.workers)
    ex.run(contributors_csv=args.contributors, sample_size=args.sample)


if __name__ == '__main__':
    main()


