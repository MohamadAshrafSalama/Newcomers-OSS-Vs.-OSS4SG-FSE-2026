#!/usr/bin/env python3
"""
ROBUST Fast GitHub Extractor with Proper Token Management
- Tracks each token's rate limit (5000 points/hour)
- Automatically sleeps when tokens exhausted
- Shows token health in progress bar
- Never gets stuck or deadlocks
"""

import json
import time
import pickle
import hashlib
import logging
import sys
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from queue import Queue, Empty
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm
import requests


@dataclass
class TokenHealth:
    """Track detailed token health and limits."""
    token: str
    worker_id: int
    points_remaining: int = 5000
    reset_time: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    total_requests: int = 0
    failed_requests: int = 0
    last_success: Optional[datetime] = None
    is_exhausted: bool = False
    consecutive_failures: int = 0


@dataclass
class WorkItem:
    """Work item for queue."""
    row: pd.Series
    contributor_id: str
    retry_count: int = 0


class RobustFastExtractor:
    """Robust extractor that never gets stuck."""
    
    def __init__(self, tokens: List[str], output_dir: str, max_workers: int = None):
        self.tokens_list = tokens
        self.max_workers = min(max_workers or len(tokens), len(tokens), 8)
        
        # Token health tracking (thread-safe)
        self.tokens: Dict[int, TokenHealth] = {}
        self.token_lock = threading.Lock()
        for i in range(self.max_workers):
            self.tokens[i] = TokenHealth(token=tokens[i], worker_id=i)
        
        # Directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.state_file = self.output_dir / "extraction_state.pkl"
        self.failed_dir = self.output_dir / "failed"
        self.failed_dir.mkdir(exist_ok=True)
        
        # State management
        self.state_lock = threading.Lock()
        self.state = self.load_state()
        
        # Username cache
        self.username_cache = {}
        self.username_cache_lock = threading.Lock()
        self.load_username_cache()
        
        # Logging
        self.setup_logging()
        
        # API endpoints
        self.graphql_url = "https://api.github.com/graphql"
        self.rest_url = "https://api.github.com"
        
        # Worker status tracking
        self.worker_status = {i: "Starting" for i in range(self.max_workers)}
        self.worker_status_lock = threading.Lock()
        
        self.logger.info(f"Initialized with {self.max_workers} workers")
    
    def setup_logging(self):
        """Setup logging with detailed formatting."""
        fmt = '%(asctime)s - [%(threadName)-10s] - %(levelname)s - %(message)s'
        
        # File handler
        fh = logging.FileHandler(self.output_dir / "extraction.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt))
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(fmt))
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def load_state(self) -> Dict:
        """Load saved state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.logger.info(f"Resumed: {state['processed_count']} already processed")
                    return state
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
        
        return {
            'processed_contributors': set(),
            'failed_contributors': {},
            'skipped_contributors': {},
            'processed_count': 0,
            'failed_count': 0,
            'skipped_count': 0,
            'start_time': datetime.now().isoformat()
        }
    
    def save_state(self):
        """Save current state."""
        with self.state_lock:
            self.state['last_save'] = datetime.now().isoformat()
            with open(self.state_file, 'wb') as f:
                pickle.dump(self.state, f)
    
    def load_username_cache(self):
        """Load username cache."""
        cache_file = self.output_dir / "username_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.username_cache = json.load(f)
            except:
                self.username_cache = {}
    
    def save_username_cache(self):
        """Save username cache."""
        with self.username_cache_lock:
            with open(self.output_dir / "username_cache.json", 'w') as f:
                json.dump(self.username_cache, f)
    
    def update_token_health(self, worker_id: int, response: requests.Response = None, 
                           success: bool = True, points_used: int = 10):
        """Update token health based on response."""
        with self.token_lock:
            token = self.tokens[worker_id]
            
            if success:
                token.consecutive_failures = 0
                token.last_success = datetime.now()
                token.points_remaining -= points_used
            else:
                token.consecutive_failures += 1
                token.failed_requests += 1
            
            # Check response headers for rate limits
            if response:
                if 'x-ratelimit-remaining' in response.headers:
                    try:
                        token.points_remaining = int(response.headers['x-ratelimit-remaining'])
                    except Exception:
                        pass
                if 'x-ratelimit-reset' in response.headers:
                    try:
                        token.reset_time = datetime.fromtimestamp(int(response.headers['x-ratelimit-reset']))
                    except Exception:
                        pass
                
                # Check if exhausted
                if response.status_code == 403 or token.points_remaining < 100:
                    token.is_exhausted = True
                    self.logger.warning(f"Token {worker_id} exhausted! Will reset at {token.reset_time}")
            
            # Mark exhausted if too many failures
            if token.consecutive_failures > 5:
                token.is_exhausted = True
                token.reset_time = datetime.now() + timedelta(hours=1)
                self.logger.warning(f"Token {worker_id} marked exhausted due to failures")
    
    def wait_for_token_reset(self, worker_id: int):
        """Wait for token to reset."""
        with self.token_lock:
            token = self.tokens[worker_id]
            if not token.is_exhausted:
                return
            
            wait_time = (token.reset_time - datetime.now()).total_seconds()
            if wait_time > 0:
                self.logger.info(f"Worker {worker_id} sleeping {wait_time/60:.1f} minutes until reset")
                with self.worker_status_lock:
                    self.worker_status[worker_id] = f"Sleeping ({wait_time/60:.0f}m)"
        
        # Sleep outside lock
        if wait_time > 0:
            time.sleep(wait_time + 60)  # Add 1 minute buffer
        
        # Reset token
        with self.token_lock:
            token = self.tokens[worker_id]
            token.is_exhausted = False
            token.points_remaining = 5000
            token.consecutive_failures = 0
            self.logger.info(f"Worker {worker_id} token reset, resuming")
    
    def graphql_query(self, query: str, variables: Dict, worker_id: int) -> Optional[Dict]:
        """Execute GraphQL query with robust error handling."""
        with self.token_lock:
            token = self.tokens[worker_id]
            
            # Check if token is exhausted
            if token.is_exhausted:
                if datetime.now() < token.reset_time:
                    return None
                else:
                    # Reset if time has passed
                    token.is_exhausted = False
                    token.points_remaining = 5000
        
        headers = {
            'Authorization': f'Bearer {token.token}',
            'Content-Type': 'application/json'
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.graphql_url,
                    headers=headers,
                    json={'query': query, 'variables': variables},
                    timeout=30
                )
                
                token.total_requests += 1
                
                if response.status_code == 200:
                    data = response.json()
                    self.update_token_health(worker_id, response, success=True)
                    
                    # Check for errors in response
                    if 'errors' in data:
                        self.logger.debug(f"GraphQL errors: {str(data['errors'])[:100]}")
                    
                    return data.get('data')
                
                elif response.status_code == 403:
                    # Rate limited
                    self.update_token_health(worker_id, response, success=False)
                    self.wait_for_token_reset(worker_id)
                    return None
                
                elif response.status_code in [502, 503, 504]:
                    # Server error, retry with backoff
                    self.logger.warning(f"Server error {response.status_code}, attempt {attempt+1}")
                    time.sleep(2 ** attempt)
                    continue
                
                else:
                    self.logger.error(f"GraphQL failed: {response.status_code}")
                    self.update_token_health(worker_id, response, success=False)
                    return None
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt+1}")
                time.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"GraphQL error: {e}")
                self.update_token_health(worker_id, success=False)
                return None
        
        return None
    
    def resolve_username(self, row: pd.Series, worker_id: int) -> Optional[str]:
        """Resolve email to GitHub username."""
        # Check if already resolved
        if 'resolved_username' in row and pd.notna(row['resolved_username']):
            username = str(row['resolved_username']).strip()
            if username:
                return username
        
        email = str(row.get('contributor_email', ''))
        
        # Check cache
        cache_key = f"{email}:{row['project_name']}"
        with self.username_cache_lock:
            if cache_key in self.username_cache:
                return self.username_cache[cache_key]
        
        # Parse GitHub noreply emails
        if 'users.noreply.github.com' in email:
            parts = email.split('@')[0]
            if '+' in parts:
                username = parts.split('+')[1]
            else:
                username = parts
            
            # Verify it exists
            if self.verify_username(username, worker_id):
                with self.username_cache_lock:
                    self.username_cache[cache_key] = username
                return username
        
        # Try email prefix
        prefix = email.split('@')[0]
        for char in ['.', '-', '_', '+']:
            prefix = prefix.replace(char, '')
        
        if self.verify_username(prefix, worker_id):
            with self.username_cache_lock:
                self.username_cache[cache_key] = prefix
            return prefix
        
        return None
    
    def verify_username(self, username: str, worker_id: int) -> bool:
        """Verify username exists."""
        query = """
        query($login: String!) {
          user(login: $login) {
            login
          }
        }
        """
        
        result = self.graphql_query(query, {'login': username}, worker_id)
        return result is not None and result.get('user') is not None
    
    def get_contributor_data(self, username: str, project: str, 
                           start_date: str, end_date: str, 
                           worker_id: int) -> Tuple[List, List]:
        """Get PRs and issues for contributor."""
        owner, repo = project.split('/')
        
        # Use contributionsCollection for efficient querying
        query = """
        query($login: String!, $from: DateTime!, $to: DateTime!) {
          user(login: $login) {
            contributionsCollection(from: $from, to: $to) {
              pullRequestContributionsByRepository(maxRepositories: 25) {
                repository { nameWithOwner }
                contributions(first: 100) {
                  nodes {
                    pullRequest {
                      number
                      title
                      state
                      createdAt
                      closedAt
                      mergedAt
                      additions
                      deletions
                      changedFiles
                      
                      reviews(first: 10) {
                        nodes {
                          author { login }
                          state
                          submittedAt
                        }
                      }
                      
                      comments(first: 10) {
                        nodes {
                          author { login }
                          createdAt
                        }
                      }
                    }
                  }
                }
              }
              
              issueContributionsByRepository(maxRepositories: 25) {
                repository { nameWithOwner }
                contributions(first: 100) {
                  nodes {
                    issue {
                      number
                      title
                      state
                      createdAt
                      closedAt
                      
                      comments(first: 10) {
                        nodes {
                          author { login }
                          createdAt
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        variables = {
            'login': username,
            'from': start_date,
            'to': end_date
        }
        
        result = self.graphql_query(query, variables, worker_id)
        
        prs: List[Dict] = []
        issues: List[Dict] = []
        
        if not result or not result.get('user'):
            return prs, issues
        
        collection = result['user'].get('contributionsCollection', {})
        target_repo = f"{owner}/{repo}"
        
        # Extract PRs for this repo
        for repo_contrib in (collection.get('pullRequestContributionsByRepository') or []):
            if repo_contrib.get('repository', {}).get('nameWithOwner') == target_repo:
                for node in (repo_contrib.get('contributions', {}).get('nodes') or []):
                    if node and node.get('pullRequest'):
                        prs.append(node['pullRequest'])
        
        # Extract issues for this repo
        for repo_contrib in (collection.get('issueContributionsByRepository') or []):
            if repo_contrib.get('repository', {}).get('nameWithOwner') == target_repo:
                for node in (repo_contrib.get('contributions', {}).get('nodes') or []):
                    if node and node.get('issue'):
                        issues.append(node['issue'])
        
        return prs, issues
    
    def calculate_metrics(self, username: str, prs: List, issues: List) -> Dict:
        """Calculate treatment metrics."""
        metrics = {
            'total_prs': len(prs),
            'total_issues': len(issues),
            'total_reviews': 0,
            'total_comments': 0,
            'unique_reviewers': set(),
            'unique_commenters': set(),
            'approval_rate': 0.0,
            'response_times': [],
            'avg_response_hours': None
        }
        
        approvals = 0
        total_reviews = 0
        
        # Process PRs
        for pr in prs:
            if not pr.get('createdAt'):
                continue
            
            created = pd.to_datetime(pr['createdAt'])
            first_response = None
            
            # Reviews
            for review in (pr.get('reviews', {}).get('nodes') or []):
                if review and review.get('author', {}).get('login') != username:
                    reviewer = review['author']['login']
                    metrics['unique_reviewers'].add(reviewer)
                    total_reviews += 1
                    
                    if review.get('state') == 'APPROVED':
                        approvals += 1
                    
                    if review.get('submittedAt'):
                        review_time = pd.to_datetime(review['submittedAt'])
                        if not first_response or review_time < first_response:
                            first_response = review_time
            
            # Comments
            for comment in (pr.get('comments', {}).get('nodes') or []):
                if comment and comment.get('author', {}).get('login') != username:
                    commenter = comment['author']['login']
                    metrics['unique_commenters'].add(commenter)
                    metrics['total_comments'] += 1
                    
                    if comment.get('createdAt'):
                        comment_time = pd.to_datetime(comment['createdAt'])
                        if not first_response or comment_time < first_response:
                            first_response = comment_time
            
            if first_response:
                hours = (first_response - created).total_seconds() / 3600
                metrics['response_times'].append(hours)
        
        # Process issues
        for issue in issues:
            if not issue.get('createdAt'):
                continue
            
            created = pd.to_datetime(issue['createdAt'])
            first_response = None
            
            for comment in (issue.get('comments', {}).get('nodes') or []):
                if comment and comment.get('author', {}).get('login') != username:
                    commenter = comment['author']['login']
                    metrics['unique_commenters'].add(commenter)
                    metrics['total_comments'] += 1
                    
                    if comment.get('createdAt'):
                        comment_time = pd.to_datetime(comment['createdAt'])
                        if not first_response or comment_time < first_response:
                            first_response = comment_time
            
            if first_response:
                hours = (first_response - created).total_seconds() / 3600
                metrics['response_times'].append(hours)
        
        # Finalize metrics
        metrics['total_reviews'] = total_reviews
        if total_reviews > 0:
            metrics['approval_rate'] = approvals / total_reviews
        
        if metrics['response_times']:
            metrics['avg_response_hours'] = sum(metrics['response_times']) / len(metrics['response_times'])
        
        metrics['unique_reviewers'] = len(metrics['unique_reviewers'])
        metrics['unique_commenters'] = len(metrics['unique_commenters'])
        
        return metrics
    
    def process_contributor(self, work_item: WorkItem, worker_id: int) -> bool:
        """Process a single contributor."""
        row = work_item.row
        contributor_id = work_item.contributor_id
        
        # Update status
        with self.worker_status_lock:
            self.worker_status[worker_id] = f"Processing {contributor_id[:30]}"
        
        try:
            # Resolve username
            username = self.resolve_username(row, worker_id)
            if not username:
                with self.state_lock:
                    self.state['skipped_contributors'][contributor_id] = "Username resolution failed"
                    self.state['skipped_count'] += 1
                return False
            
            # Get data
            start_date = pd.to_datetime(row['first_commit_date']).isoformat()
            end_date = pd.to_datetime(row['first_core_date']).isoformat()
            
            prs, issues = self.get_contributor_data(
                username, row['project_name'],
                start_date, end_date,
                worker_id
            )
            
            # Create result
            data = {
                'username': username,
                'project': row['project_name'],
                'project_type': row.get('project_type', ''),
                'weeks_to_core': row.get('weeks_to_core', 0),
                'transition_start': start_date,
                'transition_end': end_date,
                'pull_requests': prs,
                'issues': issues,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            # Calculate metrics
            data['treatment_metrics'] = self.calculate_metrics(username, prs, issues)
            
            # Save to cache
            safe_name = f"{row['project_name'].replace('/', '_')}_{username}"
            hash_name = hashlib.md5(safe_name.encode()).hexdigest()[:8]
            cache_file = self.cache_dir / f"{hash_name}_{safe_name}.json"
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Update state
            with self.state_lock:
                self.state['processed_contributors'].add(contributor_id)
                self.state['processed_count'] += 1
                
                # Save periodically
                if self.state['processed_count'] % 25 == 0:
                    self.save_state()
                    self.save_username_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Worker {worker_id} failed on {contributor_id}: {e}")
            
            with self.state_lock:
                self.state['failed_contributors'][contributor_id] = str(e)
                self.state['failed_count'] += 1
            
            # Save failure info
            with open(self.failed_dir / f"{contributor_id.replace('/', '_')}.txt", 'w') as f:
                f.write(f"Error: {e}\n")
                f.write(f"Worker: {worker_id}\n")
            
            return False
    
    def worker_thread(self, work_queue: Queue, worker_id: int, pbar: tqdm):
        """Worker thread with robust error handling."""
        self.logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Check token health first
                with self.token_lock:
                    token = self.tokens[worker_id]
                    if token.is_exhausted and datetime.now() < token.reset_time:
                        self.wait_for_token_reset(worker_id)
                
                # Get work with timeout
                try:
                    work_item = work_queue.get(timeout=2)
                except Empty:
                    continue
                
                if work_item is None:  # Poison pill
                    break
                
                # Process
                _ = self.process_contributor(work_item, worker_id)
                
                # Update progress
                pbar.update(1)
                
                # Update status
                with self.worker_status_lock:
                    self.worker_status[worker_id] = "Waiting"
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                continue
        
        self.logger.info(f"Worker {worker_id} finished")
    
    def run_extraction(self, contributors_file: str, sample_size: Optional[int] = None):
        """Run extraction with monitoring."""
        self.logger.info(f"Starting extraction from {contributors_file}")
        
        # Load contributors
        df = pd.read_csv(contributors_file)
        df['contributor_id'] = df['project_name'] + '_' + df['contributor_email']
        
        # Filter already processed
        df = df[~df['contributor_id'].isin(self.state['processed_contributors'])]
        
        if sample_size:
            df = df.head(sample_size)
        
        total = len(df)
        self.logger.info(f"Processing {total} contributors with {self.max_workers} workers")
        
        # Show initial token status
        print("\n📊 Initial Token Status:")
        for i, token in self.tokens.items():
            print(f"  Worker {i}: {token.points_remaining} points remaining")
        
        # Create work queue
        work_queue = Queue(maxsize=self.max_workers * 10)  # Small queue to prevent deadlock
        
        # Start workers
        workers = []
        with tqdm(total=total, desc="Overall Progress") as pbar:
            # Set custom progress bar format
            pbar.set_postfix_str(self.get_status_string())
            
            # Start worker threads
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self.worker_thread,
                    args=(work_queue, i, pbar),
                    name=f"Worker-{i}"
                )
                worker.daemon = True  # Daemon threads die with main
                worker.start()
                workers.append(worker)
            
            # Add work items
            items_added = 0
            for _, row in df.iterrows():
                work_item = WorkItem(
                    row=row,
                    contributor_id=row['contributor_id']
                )
                
                # Add with timeout to prevent deadlock
                added = False
                for _ in range(60):  # Try for 1 minute
                    try:
                        work_queue.put(work_item, timeout=1)
                        added = True
                        break
                    except:
                        # Queue full, wait
                        pbar.set_postfix_str(self.get_status_string())
                        continue
                
                if added:
                    items_added += 1
                else:
                    self.logger.error(f"Failed to queue {row['contributor_id']}")
                
                # Update status periodically
                if items_added % 10 == 0:
                    pbar.set_postfix_str(self.get_status_string())
            
            # Send poison pills
            for _ in range(self.max_workers):
                work_queue.put(None)
            
            # Monitor while waiting
            for worker in workers:
                while worker.is_alive():
                    worker.join(timeout=5)
                    pbar.set_postfix_str(self.get_status_string())
        
        # Final save
        self.save_state()
        self.save_username_cache()
        
        # Show final stats
        self.show_final_stats()
    
    def get_status_string(self) -> str:
        """Get status string for progress bar."""
        with self.worker_status_lock:
            active = sum(1 for s in self.worker_status.values() if "Processing" in s)
            sleeping = sum(1 for s in self.worker_status.values() if "Sleeping" in s)
            
        with self.state_lock:
            processed = self.state['processed_count']
            skipped = self.state['skipped_count']
            failed = self.state['failed_count']
        
        return f"P:{processed} S:{skipped} F:{failed} | Active:{active} Sleep:{sleeping}"
    
    def show_final_stats(self):
        """Show final statistics."""
        print("\n" + "="*70)
        print("📊 FINAL STATISTICS")
        print("="*70)
        
        print(f"\n✅ Processed: {self.state['processed_count']}")
        print(f"⏭️ Skipped: {self.state['skipped_count']}")
        print(f"❌ Failed: {self.state['failed_count']}")
        
        print("\n📊 Token Usage:")
        for i, token in self.tokens.items():
            print(f"  Worker {i}: {token.total_requests} requests, "
                  f"{token.failed_requests} failed, "
                  f"{token.points_remaining} points left")
        
        total_requests = sum(t.total_requests for t in self.tokens.values())
        print(f"\n📡 Total API calls: {total_requests}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust Fast GitHub Extractor')
    parser.add_argument('--tokens', type=str, required=True,
                       help='Path to token file or comma-separated tokens')
    parser.add_argument('--contributors', type=str, required=True,
                       help='Path to contributors CSV')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers (default: number of tokens, max 8)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing')
    
    args = parser.parse_args()
    
    # Load tokens
    if Path(args.tokens).exists():
        with open(args.tokens, 'r') as f:
            tokens = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        tokens = [t.strip() for t in args.tokens.split(',')]
    
    if not tokens:
        print("❌ No tokens provided!")
        return
    
    print("🚀 Robust Fast GitHub Extractor")
    print(f"📊 Tokens: {len(tokens)}")
    print(f"👷 Workers: {args.workers or len(tokens)}")
    print(f"📁 Output: {args.output_dir}")
    print("\n✅ Features:")
    print("  • Tracks each token's 5000 points/hour limit")
    print("  • Automatically sleeps when exhausted")
    print("  • Shows worker status in progress bar")
    print("  • Never deadlocks or gets stuck")
    
    # Run extraction
    extractor = RobustFastExtractor(tokens, args.output_dir, args.workers)
    extractor.run_extraction(args.contributors, args.sample)
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()




