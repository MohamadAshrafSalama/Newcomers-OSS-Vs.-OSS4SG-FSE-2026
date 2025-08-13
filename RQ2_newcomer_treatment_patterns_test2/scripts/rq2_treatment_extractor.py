#!/usr/bin/env python3
"""
RQ2 Treatment Patterns Data Extractor
Sequential GitHub data extraction with smart token rotation
Extracts PRs and issues for core contributors up to their first core date
"""

import json
import time
import pickle
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import requests


class GitHubToken:
    """GitHub token with rate limit tracking."""
    
    def __init__(self, token_str: str, token_id: int):
        self.token = token_str
        self.id = token_id
        self.points_used = 0
        self.points_limit = 5000
        self.reset_time = datetime.now() + timedelta(hours=1)
        self.total_calls = 0
        self.last_used = None
        self.consecutive_errors = 0
        
    def can_use(self) -> bool:
        """Check if token can be used."""
        # Reset if hour passed
        if datetime.now() >= self.reset_time:
            self.points_used = 0
            self.reset_time = datetime.now() + timedelta(hours=1)
            self.consecutive_errors = 0
            
        # Keep 100 points buffer and check for errors
        return self.points_used < (self.points_limit - 100) and self.consecutive_errors < 5
    
    def use(self, points: int = 10):
        """Mark points as used."""
        self.points_used += points
        self.total_calls += 1
        self.last_used = datetime.now()
    
    def mark_error(self):
        """Mark an error occurred."""
        self.consecutive_errors += 1
    
    def clear_errors(self):
        """Clear error count on successful request."""
        self.consecutive_errors = 0
    
    def time_until_reset(self) -> float:
        """Seconds until reset."""
        return max(0, (self.reset_time - datetime.now()).total_seconds())


class RQ2DataExtractor:
    """Extract PR and issue treatment data for RQ2 analysis."""
    
    def __init__(self, token_strings: List[str], output_dir: str):
        """Initialize extractor with tokens."""
        # Create token objects
        self.tokens = [GitHubToken(t, i) for i, t in enumerate(token_strings)]
        self.current_token_idx = 0
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.state_file = self.output_dir / "extraction_state.pkl"
        self.failed_dir = self.output_dir / "failed"
        self.failed_dir.mkdir(exist_ok=True)
        
        # Load state
        self.state = self.load_state()
        
        # API endpoint
        self.graphql_url = "https://api.github.com/graphql"
        
        print(f"✅ Initialized with {len(self.tokens)} tokens")
        print(f"📁 Output directory: {output_dir}")
    
    def load_state(self) -> Dict:
        """Load saved state for resumption."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    print(f"📂 Resumed: {state['processed_count']} already processed")
                    return state
            except Exception as e:
                print(f"⚠️ Could not load state: {e}")
        
        return {
            'processed_contributors': set(),
            'failed_contributors': {},
            'skipped_contributors': {},
            'processed_count': 0,
            'failed_count': 0,
            'skipped_count': 0,
            'start_time': datetime.now().isoformat(),
            'total_prs': 0,
            'total_issues': 0
        }
    
    def save_state(self):
        """Save current state to disk."""
        self.state['last_save'] = datetime.now().isoformat()
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.state, f)
    
    def get_next_token(self) -> Optional[GitHubToken]:
        """Get next available token or wait for reset."""
        attempts = 0
        while attempts < len(self.tokens) * 2:
            token = self.tokens[self.current_token_idx]
            
            if token.can_use():
                return token
            
            # Try next token
            self.current_token_idx = (self.current_token_idx + 1) % len(self.tokens)
            attempts += 1
        
        # All tokens exhausted - find which resets first
        next_token = min(self.tokens, key=lambda t: t.time_until_reset())
        wait_time = next_token.time_until_reset()
        
        if wait_time > 0:
            print(f"\n⏰ All tokens exhausted. Waiting {wait_time/60:.1f} minutes...")
            print(f"   Reset time: {next_token.reset_time.strftime('%H:%M:%S')}")
            
            # Show all token status
            for t in self.tokens:
                remaining = max(0, t.points_limit - t.points_used)
                reset_in = t.time_until_reset() / 60
                print(f"   Token {t.id}: {remaining} points left, resets in {reset_in:.1f}m")
            
            # Sleep with progress bar
            wait_seconds = int(wait_time) + 60  # Add 1 minute buffer
            for _ in tqdm(range(wait_seconds), desc="Waiting", unit="s"):
                time.sleep(1)
            
            # Reset the token that's ready
            next_token.points_used = 0
            next_token.reset_time = datetime.now() + timedelta(hours=1)
            next_token.consecutive_errors = 0
            print(f"✅ Token {next_token.id} reset! Resuming...")
        
        self.current_token_idx = next_token.id
        return next_token
    
    def graphql_query(self, query: str, variables: Dict) -> Optional[Dict]:
        """Execute GraphQL query with automatic token rotation."""
        max_retries = 3
        
        for attempt in range(max_retries):
            # Get available token
            token = self.get_next_token()
            if not token:
                print("❌ No tokens available")
                return None
            
            headers = {
                'Authorization': f'Bearer {token.token}',
                'Content-Type': 'application/json'
            }
            
            try:
                response = requests.post(
                    self.graphql_url,
                    headers=headers,
                    json={'query': query, 'variables': variables},
                    timeout=30
                )
                
                # Track usage (GraphQL queries typically use 1-10 points)
                token.use(10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update token limits from headers
                    if 'x-ratelimit-remaining' in response.headers:
                        remaining = int(response.headers['x-ratelimit-remaining'])
                        token.points_used = token.points_limit - remaining
                    
                    if 'x-ratelimit-reset' in response.headers:
                        token.reset_time = datetime.fromtimestamp(
                            int(response.headers['x-ratelimit-reset'])
                        )
                    
                    token.clear_errors()
                    
                    # Check for GraphQL errors
                    if 'errors' in data:
                        print(f"   GraphQL error: {data['errors']}")
                        return None
                    
                    return data.get('data')
                
                elif response.status_code == 403:
                    # Rate limited
                    print(f"   Token {token.id} rate limited")
                    token.points_used = token.points_limit
                    token.mark_error()
                    continue
                
                elif response.status_code in [502, 503, 504]:
                    # Server error - retry with backoff
                    time.sleep(2 ** attempt)
                    continue
                
                else:
                    print(f"   HTTP error {response.status_code}: {response.text[:200]}")
                    token.mark_error()
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"   Timeout on attempt {attempt + 1}")
                time.sleep(2 ** attempt)
                continue
            except Exception as e:
                print(f"   Request error: {e}")
                token.mark_error()
                return None
        
        return None
    
    def get_prs_and_issues(self, username: str, repo: str, 
                          end_date: str, lookback_days: int = 365) -> Dict:
        """
        Get PRs and issues for a contributor up to their core date.
        
        Args:
            username: GitHub username
            repo: Repository name (owner/name format)
            end_date: ISO format date when they became core
            lookback_days: How many days before core date to look (default 365)
        """
        # Calculate start date (1 year before becoming core by default)
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        start_dt = end_dt - timedelta(days=lookback_days)
        
        # GraphQL query for PRs and issues
        query = """
        query($owner: String!, $name: String!, $author: String!, $prCursor: String, $issueCursor: String) {
          repository(owner: $owner, name: $name) {
            pullRequests(first: 50, author: $author, after: $prCursor, orderBy: {field: CREATED_AT, direction: ASC}) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                number
                title
                state
                createdAt
                closedAt
                mergedAt
                additions
                deletions
                changedFiles
                mergeable
                merged
                
                reviews(first: 20) {
                  totalCount
                  nodes {
                    author { login }
                    state
                    submittedAt
                    body
                  }
                }
                
                comments(first: 20) {
                  totalCount
                  nodes {
                    author { login }
                    createdAt
                    body
                  }
                }
                
                labels(first: 10) {
                  nodes { name }
                }
                
                timelineItems(first: 30, itemTypes: [REVIEW_REQUESTED_EVENT, ASSIGNED_EVENT, MENTIONED_EVENT]) {
                  nodes {
                    __typename
                    ... on ReviewRequestedEvent {
                      createdAt
                      requestedReviewer {
                        ... on User { login }
                      }
                    }
                    ... on AssignedEvent {
                      createdAt
                      assignee { login }
                    }
                    ... on MentionedEvent {
                      createdAt
                    }
                  }
                }
              }
            }
            
            issues(first: 50, author: $author, after: $issueCursor, orderBy: {field: CREATED_AT, direction: ASC}) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                number
                title
                state
                createdAt
                closedAt
                
                comments(first: 20) {
                  totalCount
                  nodes {
                    author { login }
                    createdAt
                    body
                  }
                }
                
                labels(first: 10) {
                  nodes { name }
                }
                
                timelineItems(first: 20, itemTypes: [ASSIGNED_EVENT, MENTIONED_EVENT]) {
                  nodes {
                    __typename
                    ... on AssignedEvent {
                      createdAt
                      assignee { login }
                    }
                    ... on MentionedEvent {
                      createdAt
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        owner, name = repo.split('/')
        all_prs = []
        all_issues = []
        
        # Paginate through PRs
        pr_cursor = None
        while True:
            result = self.graphql_query(query, {
                'owner': owner,
                'name': name,
                'author': username,
                'prCursor': pr_cursor,
                'issueCursor': None
            })
            
            if not result or not result.get('repository'):
                break
            
            prs = result['repository'].get('pullRequests', {})
            pr_nodes = prs.get('nodes', [])
            
            # Filter by date
            for pr in pr_nodes:
                if pr and pr.get('createdAt'):
                    created_date = datetime.fromisoformat(pr['createdAt'].replace('Z', '+00:00'))
                    if created_date <= end_dt:
                        all_prs.append(pr)
            
            # Check for more pages
            page_info = prs.get('pageInfo', {})
            if page_info.get('hasNextPage'):
                pr_cursor = page_info.get('endCursor')
            else:
                break
        
        # Paginate through issues
        issue_cursor = None
        while True:
            result = self.graphql_query(query, {
                'owner': owner,
                'name': name,
                'author': username,
                'prCursor': None,
                'issueCursor': issue_cursor
            })
            
            if not result or not result.get('repository'):
                break
            
            issues = result['repository'].get('issues', {})
            issue_nodes = issues.get('nodes', [])
            
            # Filter by date
            for issue in issue_nodes:
                if issue and issue.get('createdAt'):
                    created_date = datetime.fromisoformat(issue['createdAt'].replace('Z', '+00:00'))
                    if created_date <= end_dt:
                        all_issues.append(issue)
            
            # Check for more pages
            page_info = issues.get('pageInfo', {})
            if page_info.get('hasNextPage'):
                issue_cursor = page_info.get('endCursor')
            else:
                break
        
        return {
            'pull_requests': all_prs,
            'issues': all_issues,
            'extraction_date': datetime.now().isoformat(),
            'lookback_days': lookback_days,
            'date_range': {
                'start': start_dt.isoformat(),
                'end': end_dt.isoformat()
            }
        }
    
    def calculate_treatment_metrics(self, data: Dict, username: str) -> Dict:
        """Calculate treatment metrics from PR and issue data."""
        metrics = {
            # PR metrics
            'total_prs': len(data['pull_requests']),
            'merged_prs': 0,
            'pr_merge_rate': 0.0,
            'total_pr_reviews': 0,
            'total_pr_comments': 0,
            'unique_pr_reviewers': set(),
            'unique_pr_commenters': set(),
            'pr_approval_count': 0,
            'pr_changes_requested_count': 0,
            'pr_response_times_hours': [],
            'avg_pr_response_hours': None,
            
            # Issue metrics
            'total_issues': len(data['issues']),
            'closed_issues': 0,
            'issue_close_rate': 0.0,
            'total_issue_comments': 0,
            'unique_issue_commenters': set(),
            'issue_response_times_hours': [],
            'avg_issue_response_hours': None,
            
            # Combined metrics
            'total_unique_responders': 0,
            'avg_overall_response_hours': None
        }
        
        # Process PRs
        for pr in data['pull_requests']:
            if not pr:
                continue
                
            # Merge status
            if pr.get('merged'):
                metrics['merged_prs'] += 1
            
            created_at = datetime.fromisoformat(pr['createdAt'].replace('Z', '+00:00'))
            first_response = None
            
            # Reviews
            for review in (pr.get('reviews', {}).get('nodes') or []):
                if review and review.get('author', {}).get('login') != username:
                    metrics['total_pr_reviews'] += 1
                    metrics['unique_pr_reviewers'].add(review['author']['login'])
                    
                    if review.get('state') == 'APPROVED':
                        metrics['pr_approval_count'] += 1
                    elif review.get('state') == 'CHANGES_REQUESTED':
                        metrics['pr_changes_requested_count'] += 1
                    
                    if review.get('submittedAt'):
                        review_time = datetime.fromisoformat(
                            review['submittedAt'].replace('Z', '+00:00')
                        )
                        if not first_response or review_time < first_response:
                            first_response = review_time
            
            # Comments
            for comment in (pr.get('comments', {}).get('nodes') or []):
                if comment and comment.get('author', {}).get('login') != username:
                    metrics['total_pr_comments'] += 1
                    metrics['unique_pr_commenters'].add(comment['author']['login'])
                    
                    if comment.get('createdAt'):
                        comment_time = datetime.fromisoformat(
                            comment['createdAt'].replace('Z', '+00:00')
                        )
                        if not first_response or comment_time < first_response:
                            first_response = comment_time
            
            # Calculate response time
            if first_response:
                hours = (first_response - created_at).total_seconds() / 3600
                if hours >= 0:  # Sanity check
                    metrics['pr_response_times_hours'].append(hours)
        
        # Process issues
        for issue in data['issues']:
            if not issue:
                continue
                
            # Close status
            if issue.get('state') == 'CLOSED':
                metrics['closed_issues'] += 1
            
            created_at = datetime.fromisoformat(issue['createdAt'].replace('Z', '+00:00'))
            first_response = None
            
            # Comments
            for comment in (issue.get('comments', {}).get('nodes') or []):
                if comment and comment.get('author', {}).get('login') != username:
                    metrics['total_issue_comments'] += 1
                    metrics['unique_issue_commenters'].add(comment['author']['login'])
                    
                    if comment.get('createdAt'):
                        comment_time = datetime.fromisoformat(
                            comment['createdAt'].replace('Z', '+00:00')
                        )
                        if not first_response or comment_time < first_response:
                            first_response = comment_time
            
            # Calculate response time
            if first_response:
                hours = (first_response - created_at).total_seconds() / 3600
                if hours >= 0:  # Sanity check
                    metrics['issue_response_times_hours'].append(hours)
        
        # Calculate final metrics
        if metrics['total_prs'] > 0:
            metrics['pr_merge_rate'] = metrics['merged_prs'] / metrics['total_prs']
        
        if metrics['total_issues'] > 0:
            metrics['issue_close_rate'] = metrics['closed_issues'] / metrics['total_issues']
        
        # Convert sets to counts
        metrics['unique_pr_reviewers'] = len(metrics['unique_pr_reviewers'])
        metrics['unique_pr_commenters'] = len(metrics['unique_pr_commenters'])
        metrics['unique_issue_commenters'] = len(metrics['unique_issue_commenters'])
        
        # Calculate average response times
        if metrics['pr_response_times_hours']:
            metrics['avg_pr_response_hours'] = sum(metrics['pr_response_times_hours']) / len(metrics['pr_response_times_hours'])
            metrics['median_pr_response_hours'] = sorted(metrics['pr_response_times_hours'])[len(metrics['pr_response_times_hours'])//2]
        
        if metrics['issue_response_times_hours']:
            metrics['avg_issue_response_hours'] = sum(metrics['issue_response_times_hours']) / len(metrics['issue_response_times_hours'])
            metrics['median_issue_response_hours'] = sorted(metrics['issue_response_times_hours'])[len(metrics['issue_response_times_hours'])//2]
        
        # Overall response time
        all_response_times = metrics['pr_response_times_hours'] + metrics['issue_response_times_hours']
        if all_response_times:
            metrics['avg_overall_response_hours'] = sum(all_response_times) / len(all_response_times)
            metrics['median_overall_response_hours'] = sorted(all_response_times)[len(all_response_times)//2]
        
        # Remove the raw lists for cleaner output
        del metrics['pr_response_times_hours']
        del metrics['issue_response_times_hours']
        
        return metrics
    
    def process_contributor(self, row: pd.Series) -> bool:
        """Process a single contributor."""
        contributor_id = f"{row['project_name']}_{row['contributor_email']}"
        
        # Skip if already processed
        if contributor_id in self.state['processed_contributors']:
            return True
        
        # Skip if no username resolved
        if pd.isna(row.get('resolved_username')) or not row.get('resolved_username'):
            self.state['skipped_contributors'][contributor_id] = "No resolved username"
            self.state['skipped_count'] += 1
            return False
        
        try:
            # Get PR and issue data
            data = self.get_prs_and_issues(
                username=row['resolved_username'],
                repo=row['project_name'],
                end_date=row['first_core_date'],
                lookback_days=365  # Look back 1 year before becoming core
            )
            
            # Calculate metrics
            metrics = self.calculate_treatment_metrics(data, row['resolved_username'])
            
            # Build complete result
            result = {
                'contributor_id': contributor_id,
                'project_name': row['project_name'],
                'project_type': row.get('project_type', ''),
                'contributor_email': row['contributor_email'],
                'username': row['resolved_username'],
                'weeks_to_core': row.get('weeks_to_core', 0),
                'first_commit_date': row.get('first_commit_date'),
                'first_core_date': row['first_core_date'],
                'treatment_metrics': metrics,
                'raw_data': data,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            # Save to cache
            safe_name = contributor_id.replace('/', '_').replace('@', '_at_')
            hash_name = hashlib.md5(safe_name.encode()).hexdigest()[:8]
            cache_file = self.cache_dir / f"{hash_name}_{safe_name[:50]}.json"
            
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Update state
            self.state['processed_contributors'].add(contributor_id)
            self.state['processed_count'] += 1
            self.state['total_prs'] += metrics['total_prs']
            self.state['total_issues'] += metrics['total_issues']
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error processing {contributor_id}: {e}")
            self.state['failed_contributors'][contributor_id] = str(e)
            self.state['failed_count'] += 1
            
            # Save error details
            error_file = self.failed_dir / f"{contributor_id.replace('/', '_')}.txt"
            with open(error_file, 'w') as f:
                f.write(f"Error: {e}\n")
                f.write(f"Row data: {row.to_dict()}\n")
            
            return False
    
    def run(self, contributors_file: str, sample_size: Optional[int] = None):
        """Run extraction process."""
        print(f"\n🚀 Starting RQ2 treatment data extraction")
        print(f"📄 Input file: {contributors_file}")
        
        # Load contributors
        df = pd.read_csv(contributors_file)
        
        # Create unique ID
        df['contributor_id'] = df['project_name'] + '_' + df['contributor_email']
        
        # Filter already processed
        df_to_process = df[~df['contributor_id'].isin(self.state['processed_contributors'])]
        
        # Apply sample size if specified
        if sample_size:
            df_to_process = df_to_process.head(sample_size)
        
        total = len(df_to_process)
        print(f"📊 Processing {total} contributors")
        print(f"   Already processed: {len(self.state['processed_contributors'])}")
        print(f"   Skipped: {self.state['skipped_count']}")
        print(f"   Failed: {self.state['failed_count']}")
        
        # Show token status
        print(f"\n📊 Token Status:")
        for t in self.tokens:
            available = max(0, t.points_limit - t.points_used)
            print(f"  Token {t.id}: {available}/{t.points_limit} points available")
        
        # Process each contributor
        with tqdm(total=total, desc="Extracting") as pbar:
            for idx, row in df_to_process.iterrows():
                # Update description
                username = row.get('resolved_username', 'unknown')[:20]
                pbar.set_description(f"Processing {username}")
                
                # Process contributor
                success = self.process_contributor(row)
                
                # Update progress
                pbar.update(1)
                
                # Update stats
                pbar.set_postfix({
                    'Done': self.state['processed_count'],
                    'Skip': self.state['skipped_count'],
                    'Fail': self.state['failed_count'],
                    'PRs': self.state['total_prs'],
                    'Issues': self.state['total_issues'],
                    'Token': self.current_token_idx
                })
                
                # Save state periodically
                if self.state['processed_count'] % 10 == 0:
                    self.save_state()
        
        # Final save
        self.save_state()
        
        # Generate summary
        self.generate_summary()
        
        print(f"\n✅ Extraction complete!")
        print(f"📊 Final Statistics:")
        print(f"  ✅ Processed: {self.state['processed_count']}")
        print(f"  ⏭️ Skipped: {self.state['skipped_count']}")
        print(f"  ❌ Failed: {self.state['failed_count']}")
        print(f"  📝 Total PRs: {self.state['total_prs']}")
        print(f"  📋 Total Issues: {self.state['total_issues']}")
        
        # Token usage
        print(f"\n📊 Token Usage:")
        total_calls = 0
        for t in self.tokens:
            print(f"  Token {t.id}: {t.total_calls} calls, {t.points_used}/{t.points_limit} points")
            total_calls += t.total_calls
        print(f"\n📡 Total API calls: {total_calls}")
    
    def generate_summary(self):
        """Generate summary CSV from cached results."""
        summary_data = []
        
        # Read all cache files
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract key metrics
                    metrics = data['treatment_metrics']
                    summary_data.append({
                        'project_name': data['project_name'],
                        'project_type': data['project_type'],
                        'contributor_email': data['contributor_email'],
                        'username': data['username'],
                        'weeks_to_core': data['weeks_to_core'],
                        'total_prs': metrics['total_prs'],
                        'merged_prs': metrics['merged_prs'],
                        'pr_merge_rate': metrics['pr_merge_rate'],
                        'total_pr_reviews': metrics['total_pr_reviews'],
                        'pr_approval_count': metrics['pr_approval_count'],
                        'avg_pr_response_hours': metrics['avg_pr_response_hours'],
                        'total_issues': metrics['total_issues'],
                        'closed_issues': metrics['closed_issues'],
                        'issue_close_rate': metrics['issue_close_rate'],
                        'total_issue_comments': metrics['total_issue_comments'],
                        'avg_issue_response_hours': metrics['avg_issue_response_hours'],
                        'avg_overall_response_hours': metrics['avg_overall_response_hours']
                    })
            except Exception as e:
                print(f"Error reading {cache_file}: {e}")
        
        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.output_dir / "treatment_metrics_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"\n📊 Summary saved to: {summary_file}")
            print(f"   Total records: {len(summary_df)}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ2 GitHub Treatment Data Extractor')
    parser.add_argument('--tokens', type=str, required=True,
                       help='Comma-separated GitHub tokens or path to token file')
    parser.add_argument('--contributors', type=str, required=True,
                       help='Path to core_contributors_filtered.csv')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing (optional)')
    
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
    
    print("=" * 70)
    print("🔬 RQ2 TREATMENT PATTERNS DATA EXTRACTOR")
    print("=" * 70)
    print("\nThis script extracts PR and issue treatment data for core contributors")
    print("to analyze how newcomers are treated during their journey to becoming core.")
    print("\n✅ Features:")
    print("  • Sequential processing (no deadlocks)")
    print("  • Smart token rotation with exact tracking")
    print("  • Automatic rate limit handling")
    print("  • State persistence for resumption")
    print("  • Comprehensive treatment metrics")
    
    # Run extractor
    extractor = RQ2DataExtractor(tokens, args.output_dir)
    extractor.run(args.contributors, args.sample)


if __name__ == "__main__":
    main()


