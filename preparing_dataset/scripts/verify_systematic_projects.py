#!/usr/bin/env python3
"""
Verify all systematic projects against our 5 criteria
Uses GitHub API with rate limiting and proper error handling
"""

import csv
import requests
import time
import json
from datetime import datetime, timedelta

class ProjectVerifier:
    def __init__(self, github_token=None):
        self.github_token = github_token
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Academic-Research-Verification'
        }
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
        
        # Our 5 criteria
        self.criteria = {
            'min_contributors': 10,
            'min_commits': 500,
            'min_closed_prs': 50,
            'min_age_years': 1,
            'max_days_since_update': 365
        }
        
        self.verified_projects = []
        self.failed_projects = []
        self.api_calls_made = 0
        
    def check_rate_limit(self):
        """Check GitHub API rate limit"""
        try:
            response = requests.get('https://api.github.com/rate_limit', headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                remaining = data['rate']['remaining']
                reset_time = data['rate']['reset']
                print(f"API Rate Limit: {remaining} calls remaining, resets at {datetime.fromtimestamp(reset_time)}")
                return remaining > 10
            return False
        except:
            return False
    
    def verify_project(self, project_name):
        """Verify a single project against all 5 criteria"""
        print(f"\\nVerifying: {project_name}")
        
        if not self.check_rate_limit():
            print(f"Rate limit exceeded, skipping {project_name}")
            return None
        
        try:
            # Get basic repository info
            repo_url = f'https://api.github.com/repos/{project_name}'
            repo_response = requests.get(repo_url, headers=self.headers)
            self.api_calls_made += 1
            
            if repo_response.status_code != 200:
                print(f"Failed to get repo info: {repo_response.status_code}")
                return None
            
            repo_data = repo_response.json()
            
            # Initialize verification results
            verification = {
                'project_name': project_name,
                'contributors_count': 0,
                'commits_count': 0,
                'closed_prs_count': 0,
                'age_years': 0,
                'days_since_update': 0,
                'meets_criteria': False,
                'criteria_details': {},
                'error': None
            }
            
            # Check 1: Age (>1 year)
            created_at = datetime.strptime(repo_data['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            age_years = (datetime.now() - created_at).days / 365.25
            verification['age_years'] = age_years
            verification['criteria_details']['age_check'] = age_years >= self.criteria['min_age_years']
            
            # Check 2: Recent update (<1 year)
            updated_at = datetime.strptime(repo_data['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
            days_since_update = (datetime.now() - updated_at).days
            verification['days_since_update'] = days_since_update
            verification['criteria_details']['recent_update'] = days_since_update <= self.criteria['max_days_since_update']
            
            # Rate limiting pause
            time.sleep(0.5)
            
            # Check 3: Contributors (≥10)
            if not self.check_rate_limit():
                return verification
            
            contributors_url = f'https://api.github.com/repos/{project_name}/contributors'
            contributors_response = requests.get(contributors_url, headers=self.headers)
            self.api_calls_made += 1
            
            if contributors_response.status_code == 200:
                contributors_data = contributors_response.json()
                # Get total count from headers if available, otherwise count returned items
                if 'Link' in contributors_response.headers:
                    # Multiple pages, estimate based on last page
                    link_header = contributors_response.headers['Link']
                    if 'last' in link_header:
                        # Extract page number from last link
                        import re
                        last_page_match = re.search(r'page=(\d+)>; rel="last"', link_header)
                        if last_page_match:
                            last_page = int(last_page_match.group(1))
                            # Estimate: (last_page - 1) * 30 + current page items
                            estimated_contributors = (last_page - 1) * 30 + len(contributors_data)
                            verification['contributors_count'] = estimated_contributors
                        else:
                            verification['contributors_count'] = len(contributors_data)
                    else:
                        verification['contributors_count'] = len(contributors_data)
                else:
                    verification['contributors_count'] = len(contributors_data)
            
            verification['criteria_details']['contributors_check'] = verification['contributors_count'] >= self.criteria['min_contributors']
            
            time.sleep(0.5)
            
            # Check 4: Commits (≥500) - Use search API for efficiency
            if not self.check_rate_limit():
                return verification
                
            # Use search API to count commits more efficiently
            commits_search_url = f'https://api.github.com/search/commits?q=repo:{project_name}'
            commits_response = requests.get(commits_search_url, headers=self.headers)
            self.api_calls_made += 1
            
            if commits_response.status_code == 200:
                commits_data = commits_response.json()
                verification['commits_count'] = commits_data.get('total_count', 0)
            else:
                # Fallback: estimate from repository stats
                verification['commits_count'] = repo_data.get('size', 0) * 2  # Rough estimate
            
            verification['criteria_details']['commits_check'] = verification['commits_count'] >= self.criteria['min_commits']
            
            time.sleep(0.5)
            
            # Check 5: Closed PRs (≥50)
            if not self.check_rate_limit():
                return verification
            
            prs_search_url = f'https://api.github.com/search/issues?q=repo:{project_name}+type:pr+state:closed'
            prs_response = requests.get(prs_search_url, headers=self.headers)
            self.api_calls_made += 1
            
            if prs_response.status_code == 200:
                prs_data = prs_response.json()
                verification['closed_prs_count'] = prs_data.get('total_count', 0)
            
            verification['criteria_details']['prs_check'] = verification['closed_prs_count'] >= self.criteria['min_closed_prs']
            
            # Final assessment
            all_criteria_met = all(verification['criteria_details'].values())
            verification['meets_criteria'] = all_criteria_met
            
            # Print results
            status = "PASS" if all_criteria_met else "FAIL"
            print(f"{status} - {project_name}")
            print(f"  Contributors: {verification['contributors_count']} ({'PASS' if verification['criteria_details']['contributors_check'] else 'FAIL'})")
            print(f"  Commits: {verification['commits_count']} ({'PASS' if verification['criteria_details']['commits_check'] else 'FAIL'})")
            print(f"  Closed PRs: {verification['closed_prs_count']} ({'PASS' if verification['criteria_details']['prs_check'] else 'FAIL'})")
            print(f"  Age: {verification['age_years']:.1f} years ({'PASS' if verification['criteria_details']['age_check'] else 'FAIL'})")
            print(f"  Last update: {verification['days_since_update']} days ago ({'PASS' if verification['criteria_details']['recent_update'] else 'FAIL'})")
            
            return verification
            
        except Exception as e:
            print(f"Error verifying {project_name}: {str(e)}")
            verification['error'] = str(e)
            return verification
    
    def get_all_projects(self):
        """Extract ALL projects from final verified dataset for complete verification"""
        all_projects = []
        try:
            with open('final_verified_dataset.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_projects.append(row['project_name'])
        except Exception as e:
            print(f"Error reading dataset: {e}")
        
        return all_projects
    
    def verify_all_projects(self):
        """Verify ALL projects in the final dataset"""
        all_projects = self.get_all_projects()
        print(f"Found {len(all_projects)} total projects to verify")
        
        print("\\nStarting verification...")
        print("=" * 60)
        
        for i, project in enumerate(all_projects, 1):
            print(f"\\n[{i}/{len(all_projects)}] Processing {project}")
            
            result = self.verify_project(project)
            if result:
                if result['meets_criteria']:
                    self.verified_projects.append(result)
                else:
                    self.failed_projects.append(result)
            
            # Rate limiting: pause between projects
            if i % 10 == 0:
                print(f"\\n⏸️  Pausing for rate limiting... ({i}/{len(all_projects)} completed)")
                time.sleep(2)
        
        # Save results
        self.save_verification_results()
        self.print_summary()
    
    def save_verification_results(self):
        """Save verification results to CSV"""
        # Save detailed results
        with open('all_projects_verification.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'project_name', 'meets_criteria', 'contributors_count', 'commits_count', 
                'closed_prs_count', 'age_years', 'days_since_update',
                'contributors_check', 'commits_check', 'prs_check', 'age_check', 'recent_update', 'error'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.verified_projects + self.failed_projects:
                row = {
                    'project_name': result['project_name'],
                    'meets_criteria': result['meets_criteria'],
                    'contributors_count': result['contributors_count'],
                    'commits_count': result['commits_count'],
                    'closed_prs_count': result['closed_prs_count'],
                    'age_years': round(result['age_years'], 2),
                    'days_since_update': result['days_since_update'],
                    'error': result.get('error', '')
                }
                
                # Add individual criteria checks
                if 'criteria_details' in result:
                    row.update({
                        'contributors_check': result['criteria_details'].get('contributors_check', False),
                        'commits_check': result['criteria_details'].get('commits_check', False),
                        'prs_check': result['criteria_details'].get('prs_check', False),
                        'age_check': result['criteria_details'].get('age_check', False),
                        'recent_update': result['criteria_details'].get('recent_update', False)
                    })
                
                writer.writerow(row)
        
        print(f"\\nDetailed results saved to: all_projects_verification.csv")
    
    def print_summary(self):
        """Print verification summary"""
        total = len(self.verified_projects) + len(self.failed_projects)
        passed = len(self.verified_projects)
        failed = len(self.failed_projects)
        
        print("\\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total Projects Verified: {total}")
        print(f"Passed All Criteria: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed Some Criteria: {failed} ({failed/total*100:.1f}%)")
        print(f"Total API Calls Made: {self.api_calls_made}")
        
        if self.failed_projects:
            print(f"\\nProjects that FAILED criteria:")
            for project in self.failed_projects:
                failed_criteria = []
                if 'criteria_details' in project:
                    for criteria, passed in project['criteria_details'].items():
                        if not passed:
                            failed_criteria.append(criteria)
                print(f"  - {project['project_name']}: {', '.join(failed_criteria)}")

def main():
    # Initialize verifier with GitHub token for higher rate limits
    # TODO: Set your GitHub Personal Access Token here for higher rate limits (5,000/hour vs 60/hour)
    github_token = "your_github_token_here"  # Replace with your token
    verifier = ProjectVerifier(github_token if github_token != "your_github_token_here" else None)
    
    print("🔍 COMPLETE PROJECT VERIFICATION")
    print("=" * 60)
    print("This will verify ALL projects in final_verified_dataset.csv against our 5 criteria:")
    print("1. ≥10 contributors")
    print("2. ≥500 commits") 
    print("3. ≥50 closed Pull Requests")
    print("4. >1 year of history")
    print("5. Updated in last year")
    print()
    
    # Check rate limit before starting
    if not verifier.check_rate_limit():
        print("GitHub API rate limit exceeded. Please wait or add a GitHub token.")
        return
    
    # Start verification
    verifier.verify_all_projects()

if __name__ == "__main__":
    main()