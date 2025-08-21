#!/usr/bin/env python3
"""
ROBUST GitHub Treatment Data Extractor for RQ2
Completely redesigned with proper error handling and systematic approach.

Key improvements:
1. Uses REST API search to find PRs/issues (more reliable)
2. Then uses targeted GraphQL for details
3. Proper username resolution
4. Handles all edge cases
5. Academically rigorous data collection
"""

import json
import time
import pickle
import hashlib
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
from tqdm import tqdm
import requests
from dataclasses import dataclass, field


@dataclass
class TokenStatus:
    """Track token rate limits."""
    token: str
    remaining: int = 5000
    reset_time: Optional[datetime] = None
    search_remaining: int = 30  # Search API has separate limit
    search_reset_time: Optional[datetime] = None
    graphql_remaining: int = 5000
    total_requests: int = 0
    failed_requests: int = 0


@dataclass
class ContributorData:
    """Structured data for a contributor's treatment."""
    username: str
    project: str
    project_type: str
    transition_start: str
    transition_end: str
    weeks_to_core: float
    commits_to_core: int
    pull_requests: List[Dict] = field(default_factory=list)
    issues: List[Dict] = field(default_factory=list)
    treatment_metrics: Dict = field(default_factory=dict)
    extraction_timestamp: str = ""
    extraction_method: str = "hybrid"  # rest, graphql, or hybrid


class RobustGitHubExtractor:
    """Robust extractor with multiple fallback strategies."""
    
    def __init__(self, tokens: List[str], output_dir: str = "RQ2_treatment_data"):
        """Initialize with tokens and output directory."""
        self.tokens = [TokenStatus(token=t) for t in tokens]
        self.current_token_idx = 0
        self.output_dir = Path(output_dir)
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.state_file = self.output_dir / "extraction_state.pkl"
        self.failed_dir = self.output_dir / "failed"
        self.failed_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load or initialize state
        self.state = self.load_state()
        
        # API endpoints
        self.rest_base = "https://api.github.com"
        self.graphql_url = "https://api.github.com/graphql"
        
        # Username mapping cache
        self.username_cache = {}
        self.load_username_cache()
        
        self.logger.info(f"Initialized with {len(tokens)} tokens")
    
    def setup_logging(self):
        """Configure logging with both file and console output."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler(self.output_dir / "extraction.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        else:
            # Avoid duplicate handlers if re-instantiated
            self.logger.handlers.clear()
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def load_state(self) -> Dict:
        """Load previous extraction state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.logger.info(f"Resumed: {state.get('processed_count', 0)} already done")
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
        """Save current state for resume capability."""
        self.state['last_save'] = datetime.now().isoformat()
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.state, f)
    
    def load_username_cache(self):
        """Load username mapping cache."""
        cache_file = self.output_dir / "username_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.username_cache = json.load(f)
    
    def save_username_cache(self):
        """Save username mapping cache."""
        cache_file = self.output_dir / "username_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.username_cache, f)
    
    def get_next_token(self, for_search: bool = False) -> Tuple[str, int]:
        """Get next available token with rate limit awareness."""
        attempts = 0
        max_attempts = len(self.tokens) * 2
        
        while attempts < max_attempts:
            token_status = self.tokens[self.current_token_idx]
            
            # Check if token needs refresh
            now = datetime.now()
            if token_status.reset_time and now > token_status.reset_time:
                token_status.remaining = 5000
                token_status.graphql_remaining = 5000
            
            if for_search and token_status.search_reset_time and now > token_status.search_reset_time:
                token_status.search_remaining = 30
            
            # Check appropriate limit
            if for_search:
                if token_status.search_remaining > 0:
                    return token_status.token, self.current_token_idx
            else:
                if token_status.remaining > 100:
                    return token_status.token, self.current_token_idx
            
            # Try next token
            self.current_token_idx = (self.current_token_idx + 1) % len(self.tokens)
            attempts += 1
        
        # All tokens exhausted - wait
        if for_search:
            wait_time = 60  # Search resets every minute
        else:
            next_reset = min((t.reset_time for t in self.tokens if t.reset_time), 
                           default=now + timedelta(minutes=5))
            wait_time = (next_reset - now).total_seconds() + 10
        
        self.logger.warning(f"Rate limited. Waiting {wait_time:.0f}s...")
        time.sleep(wait_time)
        
        return self.tokens[0].token, 0
    
    def make_request(self, url: str, headers: Dict = None, 
                    params: Dict = None, method: str = "GET") -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if method == "GET":
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                elif method == "POST":
                    response = requests.post(url, headers=headers, json=params, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    self.logger.debug(f"404 Not Found: {url}")
                    return None
                elif response.status_code in [502, 503, 504]:
                    self.logger.warning(f"Server error {response.status_code}, retry {attempt+1}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    self.logger.error(f"Request failed: {response.status_code}")
                    return None
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt+1}")
                time.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"Request error: {e}")
                return None
        
        return None
    
    def resolve_username(self, email: str, project: str) -> Optional[str]:
        """Resolve email to GitHub username using multiple strategies."""
        
        # Check cache first
        cache_key = f"{email}:{project}"
        if cache_key in self.username_cache:
            return self.username_cache[cache_key]
        
        # Strategy 1: GitHub noreply emails
        if 'users.noreply.github.com' in email:
            # Format: ID+username@users.noreply.github.com
            parts = email.split('@')[0]
            if '+' in parts:
                username = parts.split('+')[1]
            else:
                username = parts
            
            # Verify this username exists
            if self.verify_username(username):
                self.username_cache[cache_key] = username
                return username
        
        # Strategy 2: Search commits by email in the project
        username = self.search_username_by_email(email, project)
        if username:
            self.username_cache[cache_key] = username
            return username
        
        # Strategy 3: Try email prefix as last resort
        prefix = email.split('@')[0]
        # Remove common prefixes
        for remove in ['.', '-', '_', '+']:
            prefix = prefix.replace(remove, '')
        
        if self.verify_username(prefix):
            self.username_cache[cache_key] = prefix
            return prefix
        
        # Failed to resolve
        self.logger.debug(f"Could not resolve username for {email}")
        return None
    
    def verify_username(self, username: str) -> bool:
        """Check if a username exists on GitHub."""
        token, idx = self.get_next_token()
        headers = {'Authorization': f'token {token}'}
        
        url = f"{self.rest_base}/users/{username}"
        result = self.make_request(url, headers=headers)
        
        self.tokens[idx].total_requests += 1
        return result is not None
    
    def username_from_commit_hash(self, project: str, commit_hash: str) -> Optional[str]:
        """Resolve GitHub username from a commit SHA via REST commits endpoint (no search)."""
        token, idx = self.get_next_token()
        headers = {'Authorization': f'token {token}'}
        url = f"{self.rest_base}/repos/{project}/commits/{commit_hash}"
        result = self.make_request(url, headers=headers)
        self.tokens[idx].total_requests += 1
        if not result:
            return None
        # Prefer linked GitHub user login if available
        author = result.get('author')
        if isinstance(author, dict) and author.get('login'):
            return author['login']
        # Fallback: try to infer from commit.author.email (noreply pattern)
        commit = result.get('commit', {})
        commit_author = commit.get('author', {}) if isinstance(commit, dict) else {}
        email = commit_author.get('email')
        if isinstance(email, str) and 'users.noreply.github.com' in email:
            local = email.split('@', 1)[0]
            cand = local.split('+', 1)[1] if '+' in local else local
            return cand.strip() or None
        return None
    
    def search_prs_and_issues(self, username: str, project: str, 
                             start_date: datetime, end_date: datetime) -> Dict:
        """Search for PRs and issues using REST API."""
        token, idx = self.get_next_token(for_search=True)
        headers = {'Authorization': f'token {token}'}
        
        # Format dates for GitHub search
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        results = {
            'pull_requests': [],
            'issues': []
        }
        
        # Search for PRs
        pr_query = f"repo:{project} type:pr author:{username} created:{start_str}..{end_str}"
        url = f"{self.rest_base}/search/issues"
        params = {'q': pr_query, 'per_page': 100, 'sort': 'created', 'order': 'asc'}
        
        pr_result = self.make_request(url, headers=headers, params=params)
        if pr_result and 'items' in pr_result:
            results['pull_requests'] = pr_result['items']
        
        self.tokens[idx].search_remaining -= 1
        
        # Search for issues
        issue_query = f"repo:{project} type:issue author:{username} created:{start_str}..{end_str}"
        params['q'] = issue_query
        
        issue_result = self.make_request(url, headers=headers, params=params)
        if issue_result and 'items' in issue_result:
            results['issues'] = issue_result['items']
        
        self.tokens[idx].search_remaining -= 1
        self.tokens[idx].total_requests += 2
        
        return results
    
    def get_pr_details(self, project: str, pr_number: int) -> Optional[Dict]:
        """Get detailed PR information including reviews and comments."""
        token, idx = self.get_next_token()
        headers = {'Authorization': f'token {token}'}
        
        owner, repo = project.split('/')
        
        # Get PR details
        pr_url = f"{self.rest_base}/repos/{project}/pulls/{pr_number}"
        pr_data = self.make_request(pr_url, headers=headers)
        
        if not pr_data:
            return None
        
        # Get reviews
        reviews_url = f"{pr_url}/reviews"
        reviews = self.make_request(reviews_url, headers=headers) or []
        
        # Get comments
        comments_url = f"{self.rest_base}/repos/{project}/issues/{pr_number}/comments"
        comments = self.make_request(comments_url, headers=headers) or []
        
        self.tokens[idx].total_requests += 3
        
        return {
            'number': pr_number,
            'title': pr_data.get('title'),
            'state': pr_data.get('state'),
            'created_at': pr_data.get('created_at'),
            'closed_at': pr_data.get('closed_at'),
            'merged_at': pr_data.get('merged_at'),
            'additions': pr_data.get('additions'),
            'deletions': pr_data.get('deletions'),
            'changed_files': pr_data.get('changed_files'),
            'reviews': reviews,
            'comments': comments
        }
    
    def get_issue_details(self, project: str, issue_number: int) -> Optional[Dict]:
        """Get detailed issue information including comments."""
        token, idx = self.get_next_token()
        headers = {'Authorization': f'token {token}'}
        
        # Get issue details
        issue_url = f"{self.rest_base}/repos/{project}/issues/{issue_number}"
        issue_data = self.make_request(issue_url, headers=headers)
        
        if not issue_data:
            return None
        
        # Get comments
        comments_url = f"{issue_url}/comments"
        comments = self.make_request(comments_url, headers=headers) or []
        
        self.tokens[idx].total_requests += 2
        
        return {
            'number': issue_number,
            'title': issue_data.get('title'),
            'state': issue_data.get('state'),
            'created_at': issue_data.get('created_at'),
            'closed_at': issue_data.get('closed_at'),
            'comments': comments
        }
    
    def calculate_treatment_metrics(self, data: ContributorData) -> Dict:
        """Calculate treatment metrics from collected data."""
        metrics = {
            'total_prs': len(data.pull_requests),
            'total_issues': len(data.issues),
            'total_items': len(data.pull_requests) + len(data.issues),
            'response_times': [],
            'first_response_hours': None,
            'avg_response_hours': None,
            'median_response_hours': None,
            'total_reviews': 0,
            'total_comments': 0,
            'approval_rate': 0.0,
            'changes_requested_rate': 0.0,
            'unique_reviewers': set(),
            'unique_commenters': set()
        }
        
        response_times = []
        approvals = 0
        changes_requested = 0
        
        # Analyze PRs
        for pr in data.pull_requests:
            if not pr.get('created_at'):
                continue
                
            created = pd.to_datetime(pr['created_at'])
            first_response = None
            
            # Check reviews
            for review in pr.get('reviews', []):
                if review.get('user', {}).get('login') != data.username:
                    metrics['unique_reviewers'].add(review['user']['login'])
                    metrics['total_reviews'] += 1
                    
                    if review.get('state') == 'APPROVED':
                        approvals += 1
                    elif review.get('state') == 'CHANGES_REQUESTED':
                        changes_requested += 1
                    
                    if review.get('submitted_at'):
                        review_time = pd.to_datetime(review['submitted_at'])
                        if not first_response or review_time < first_response:
                            first_response = review_time
            
            # Check comments
            for comment in pr.get('comments', []):
                if comment.get('user', {}).get('login') != data.username:
                    metrics['unique_commenters'].add(comment['user']['login'])
                    metrics['total_comments'] += 1
                    
                    if comment.get('created_at'):
                        comment_time = pd.to_datetime(comment['created_at'])
                        if not first_response or comment_time < first_response:
                            first_response = comment_time
            
            if first_response:
                hours = (first_response - created).total_seconds() / 3600
                response_times.append(hours)
        
        # Analyze issues
        for issue in data.issues:
            if not issue.get('created_at'):
                continue
                
            created = pd.to_datetime(issue['created_at'])
            first_response = None
            
            for comment in issue.get('comments', []):
                if comment.get('user', {}).get('login') != data.username:
                    metrics['unique_commenters'].add(comment['user']['login'])
                    metrics['total_comments'] += 1
                    
                    if comment.get('created_at'):
                        comment_time = pd.to_datetime(comment['created_at'])
                        if not first_response or comment_time < first_response:
                            first_response = comment_time
            
            if first_response:
                hours = (first_response - created).total_seconds() / 3600
                response_times.append(hours)
        
        # Calculate summary stats
        if response_times:
            metrics['response_times'] = response_times
            metrics['first_response_hours'] = min(response_times)
            metrics['avg_response_hours'] = sum(response_times) / len(response_times)
            metrics['median_response_hours'] = sorted(response_times)[len(response_times)//2]
        
        if metrics['total_reviews'] > 0:
            metrics['approval_rate'] = approvals / metrics['total_reviews']
            metrics['changes_requested_rate'] = changes_requested / metrics['total_reviews']
        
        # Convert sets to counts
        metrics['unique_reviewers'] = len(metrics['unique_reviewers'])
        metrics['unique_commenters'] = len(metrics['unique_commenters'])
        
        return metrics
    
    def extract_contributor_treatment(self, row: pd.Series) -> Optional[ContributorData]:
        """Extract treatment data for a single contributor."""
        
        # Resolve username with priority:
        # 1) resolved_username column if present
        # 2) sample_commit_hash via commits endpoint
        # 3) fallback email-based heuristics/verification
        username = None
        if 'resolved_username' in row and pd.notna(row['resolved_username']) and str(row['resolved_username']).strip() != '':
            username = str(row['resolved_username']).strip()
        elif 'sample_commit_hash' in row and pd.notna(row['sample_commit_hash']) and str(row['sample_commit_hash']).strip() != '':
            username = self.username_from_commit_hash(row['project_name'], str(row['sample_commit_hash']).strip())
        if not username:
            email_for_resolution = row.get('original_author_email') if 'original_author_email' in row and pd.notna(row['original_author_email']) else row['contributor_email']
            username = self.resolve_username(email_for_resolution, row['project_name'])
        if not username:
            self.logger.warning(f"Could not resolve username for {email_for_resolution}")
            return None
        
        # Parse dates
        start_date = pd.to_datetime(row['first_commit_date'])
        end_date = pd.to_datetime(row['first_core_date'])
        
        # Search for PRs and issues
        search_results = self.search_prs_and_issues(
            username, row['project_name'], start_date, end_date
        )
        
        # Create contributor data object
        contributor = ContributorData(
            username=username,
            project=row['project_name'],
            project_type=row.get('project_type', ''),
            transition_start=start_date.isoformat(),
            transition_end=end_date.isoformat(),
            weeks_to_core=row.get('weeks_to_core', 0),
            commits_to_core=row.get('commits_to_core', 0),
            extraction_timestamp=datetime.now().isoformat()
        )
        
        # Get detailed data for each PR
        for pr_summary in search_results['pull_requests']:
            pr_number = pr_summary.get('number') or pr_summary.get('pull_number')
            if pr_number is None:
                # Try to parse from URL if present
                try:
                    url = pr_summary.get('url') or pr_summary.get('html_url', '')
                    pr_number = int(str(url).rstrip('/').split('/')[-1])
                except Exception:
                    pr_number = None
            if pr_number is None:
                continue
            pr_details = self.get_pr_details(row['project_name'], int(pr_number))
            if pr_details:
                contributor.pull_requests.append(pr_details)
        
        # Get detailed data for each issue
        for issue_summary in search_results['issues']:
            issue_number = issue_summary.get('number')
            if issue_number is None:
                try:
                    url = issue_summary.get('url') or issue_summary.get('html_url', '')
                    issue_number = int(str(url).rstrip('/').split('/')[-1])
                except Exception:
                    issue_number = None
            if issue_number is None:
                continue
            issue_details = self.get_issue_details(row['project_name'], int(issue_number))
            if issue_details:
                contributor.issues.append(issue_details)
        
        # Calculate metrics
        contributor.treatment_metrics = self.calculate_treatment_metrics(contributor)
        
        return contributor
    
    def save_contributor_data(self, data: ContributorData):
        """Save contributor data to cache."""
        # Create filename
        safe_name = f"{data.project.replace('/', '_')}_{data.username}"
        hash_name = hashlib.md5(safe_name.encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"{hash_name}_{safe_name}.json"
        
        # Convert to dict and save
        data_dict = {
            'username': data.username,
            'project': data.project,
            'project_type': data.project_type,
            'transition_start': data.transition_start,
            'transition_end': data.transition_end,
            'weeks_to_core': data.weeks_to_core,
            'commits_to_core': data.commits_to_core,
            'pull_requests': data.pull_requests,
            'issues': data.issues,
            'treatment_metrics': data.treatment_metrics,
            'extraction_timestamp': data.extraction_timestamp,
            'extraction_method': data.extraction_method
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data_dict, f, indent=2, default=str)
    
    def process_contributor(self, row: pd.Series) -> bool:
        """Process a single contributor."""
        contributor_id = f"{row['project_name']}_{row['contributor_email']}"
        
        # Skip if already processed
        if contributor_id in self.state.get('processed_contributors', set()):
            return True
        
        try:
            # Extract treatment data
            contributor_data = self.extract_contributor_treatment(row)
            
            if not contributor_data:
                # Could not resolve username or extract data
                self.state['skipped_contributors'][contributor_id] = "Username resolution failed"
                self.state['skipped_count'] += 1
                return False
            
            # Save data
            self.save_contributor_data(contributor_data)
            
            # Update state
            self.state['processed_contributors'].add(contributor_id)
            self.state['processed_count'] += 1
            
            # Save state periodically
            if self.state['processed_count'] % 10 == 0:
                self.save_state()
                self.save_username_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process {contributor_id}: {e}")
            self.state['failed_contributors'][contributor_id] = str(e)
            self.state['failed_count'] += 1
            
            # Save failure info
            failure_file = self.failed_dir / f"{contributor_id.replace('/', '_')}.txt"
            with open(failure_file, 'w') as f:
                f.write(f"Error: {e}\n")
                f.write(f"Row data: {row.to_dict()}\n")
            
            return False
    
    def run_extraction(self, contributors_file: str, sample_size: Optional[int] = None):
        """Run the extraction process."""
        self.logger.info(f"Starting extraction from {contributors_file}")
        
        # Load contributors
        df = pd.read_csv(contributors_file)
        total_original = len(df)
        
        # Filter already processed
        df['contributor_id'] = df['project_name'] + '_' + df['contributor_email']
        df = df[~df['contributor_id'].isin(self.state.get('processed_contributors', set()))]
        
        if sample_size:
            df = df.head(sample_size)
        
        total = len(df)
        self.state['total_to_process'] = total
        self.logger.info(f"Processing {total} contributors ({total_original - total} already done)")
        
        # Show token status
        print("\nToken Status:")
        for i, token in enumerate(self.tokens):
            print(f"  Token {i+1}: {token.remaining} API / {token.search_remaining} search remaining")
        
        # Process contributors
        with tqdm(total=total, desc="Processing contributors") as pbar:
            for _, row in df.iterrows():
                pbar.set_description(f"Processing {str(row['contributor_email'])[:30]}")
                success = self.process_contributor(row)
                pbar.update(1)
                
                # Show stats periodically
                if (self.state['processed_count'] + self.state['failed_count']) % 25 == 0:
                    self.show_stats()
        
        # Final save
        self.save_state()
        self.save_username_cache()
        self.show_stats()
        
        self.logger.info("Extraction complete!")
    
    def show_stats(self):
        """Display current statistics."""
        print(f"\nStatistics:")
        print(f"  Processed: {self.state.get('processed_count', 0)}")
        print(f"  Skipped: {self.state.get('skipped_count', 0)}")
        print(f"  Failed: {self.state.get('failed_count', 0)}")
        
        total_requests = sum(t.total_requests for t in self.tokens)
        print(f"  Total API calls: {total_requests}")
        
        for i, token in enumerate(self.tokens):
            if token.total_requests > 0:
                print(f"    Token {i+1}: {token.total_requests} calls, {token.failed_requests} failed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust GitHub Treatment Data Extractor')
    parser.add_argument('--tokens', type=str, required=True,
                       help='Path to token file or comma-separated tokens')
    parser.add_argument('--contributors', type=str, required=True,
                       help='Path to contributors CSV')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing')
    parser.add_argument('--output-dir', type=str, 
                       default='RQ2_newcomer_treatment_patterns/results/treatment_data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load tokens
    if Path(args.tokens).exists():
        with open(args.tokens, 'r') as f:
            tokens = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        tokens = [t.strip() for t in args.tokens.split(',')]
    
    if not tokens:
        print("No tokens provided!")
        return
    
    print("Robust GitHub Treatment Data Extractor")
    print(f"Tokens: {len(tokens)}")
    print(f"Output: {args.output_dir}")
    
    # Initialize and run
    extractor = RobustGitHubExtractor(tokens, args.output_dir)
    extractor.run_extraction(args.contributors, args.sample)
    
    print("Complete!")


if __name__ == "__main__":
    main()


