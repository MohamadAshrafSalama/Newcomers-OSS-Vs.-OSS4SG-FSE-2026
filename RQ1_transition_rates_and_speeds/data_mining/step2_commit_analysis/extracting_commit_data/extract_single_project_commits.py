#!/usr/bin/env python3
"""
Extract commit data from a single project and save as JSON
This script takes a project path and extracts all commit info, saving it in the same location
"""

import subprocess
import json
import os
import sys
from datetime import datetime

def extract_commits_from_single_project(project_path, project_name):
    """
    Extract all commit data from a single repository and save to JSON
    
    Args:
        project_path: Absolute path to the cloned repository
        project_name: Original project name (e.g., "5calls/ios")
    
    Returns:
        tuple: (success: bool, json_file_path: str, error_message: str)
    """
    
    if not os.path.exists(project_path):
        return False, None, f"Repository path does not exist: {project_path}"
    
    if not os.path.exists(os.path.join(project_path, '.git')):
        return False, None, f"Not a git repository: {project_path}"
    
    # Create JSON filename (save in the same directory as the cloned project)
    safe_name = project_name.replace('/', '_')
    json_filename = f"{safe_name}_commits.json"
    json_file_path = os.path.join(project_path, json_filename)
    
    print(f"Extracting commits from: {project_name}")
    print(f"Repository path: {project_path}")
    print(f"Output JSON: {json_file_path}")
    
    # Save current directory to return to it later
    original_dir = os.getcwd()
    
    try:
        # Change to repository directory
        os.chdir(project_path)
        
        # Get commit data with file statistics
        git_cmd = [
            'git', 'log',
            '--pretty=format:%H|%an|%ae|%ad|%s',
            '--date=iso',
            '--numstat'
        ]
        
        result = subprocess.run(git_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            return False, None, f"Git command failed: {result.stderr}"
        
        # Parse the output
        commits = []
        lines = result.stdout.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if this is a commit info line (contains |)
            if '|' in line:
                parts = line.split('|', 4)
                if len(parts) == 5:
                    commit = {
                        'hash': parts[0],
                        'author_name': parts[1],
                        'author_email': parts[2],
                        'date': parts[3],
                        'message': parts[4],
                        'files_changed': []
                    }
                    
                    # Look for file stats in following lines
                    i += 1
                    while i < len(lines) and lines[i].strip() and '|' not in lines[i]:
                        stat_line = lines[i].strip()
                        if stat_line:
                            # Parse numstat format: insertions deletions filename
                            parts = stat_line.split('\t')
                            if len(parts) >= 3:
                                insertions = parts[0] if parts[0] != '-' else '0'
                                deletions = parts[1] if parts[1] != '-' else '0'
                                filename = parts[2]
                                
                                commit['files_changed'].append({
                                    'filename': filename,
                                    'insertions': insertions,
                                    'deletions': deletions
                                })
                        i += 1
                    
                    commits.append(commit)
                    continue
            
            i += 1
        
        # Create output data
        output_data = {
            'project_name': project_name,
            'repository_path': project_path,
            'extraction_date': datetime.now().isoformat(),
            'total_commits': len(commits),
            'commits': commits
        }
        
        # Save to JSON file
        with open(json_file_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Successfully extracted {len(commits)} commits")
        return True, json_file_path, None
        
    except subprocess.TimeoutExpired:
        return False, None, "Git command timed out"
    except Exception as e:
        return False, None, f"Error extracting commits: {str(e)}"
    finally:
        # Always return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_single_project_commits.py <project_path> <project_name>")
        print("Example: python extract_single_project_commits.py /path/to/repo '5calls/ios'")
        sys.exit(1)
    
    project_path = sys.argv[1]
    project_name = sys.argv[2]
    
    print("=" * 70)
    print("EXTRACTING COMMITS FROM SINGLE PROJECT")
    print("=" * 70)
    
    success, json_file, error = extract_commits_from_single_project(project_path, project_name)
    
    if success:
        print(f"\nSuccess! JSON saved to: {json_file}")
    else:
        print(f"\nFailed: {error}")
        sys.exit(1)