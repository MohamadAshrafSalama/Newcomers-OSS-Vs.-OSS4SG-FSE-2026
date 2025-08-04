# Step 1: Repository Cloning

This step clones all 375 projects from the final_balanced_dataset.csv to prepare for commit analysis.

Part of RQ1 Data Mining process.

## Overview

- **Input**: `../../../preparing_dataset/data/final_balanced_dataset.csv` (375 projects)
- **Output**: `cloned_repositories/` directory with all GitHub repositories
- **Features**: Progress tracking, resume capability, error handling

## Files

- `clone_all_projects.py` - Main cloning script
- `cloned_repositories/` - Directory containing all cloned repos (created during execution)
- `clone_progress.json` - Progress tracking file (created during execution)
- `clone_log.txt` - Detailed execution log (created during execution)

## Dependencies

```bash
pip install tqdm
```

## Usage

### Basic Usage
```bash
python clone_all_projects.py
```

### With GitHub Token (Recommended)
```bash
python clone_all_projects.py --github-token YOUR_GITHUB_TOKEN
```

### Force Re-clone All Repositories
```bash
python clone_all_projects.py --force-reclone
```

## Features

### Progress Tracking
- Real-time progress bar showing current status
- Live statistics: successful, failed, and skipped repositories
- Estimated time remaining

### Resume Capability
- Automatically skips already cloned repositories
- Can resume after interruption (Ctrl+C)
- Progress saved every 10 repositories

### Error Handling
- 5-minute timeout per repository
- Detailed error logging
- Failed repositories tracked with error messages

### Directory Structure
Repositories are cloned with safe naming:
- `owner/repository` becomes `owner_repository/`
- Example: `facebook/react` → `cloned_repositories/facebook_react/`

## Expected Results

- **Total Projects**: 375
- **OSS Projects**: 185
- **OSS4SG Projects**: 190
- **Estimated Time**: 2-4 hours (depending on network and project sizes)
- **Estimated Storage**: 20-50 GB

## Troubleshooting

### Rate Limiting
If you encounter rate limiting errors, use a GitHub Personal Access Token:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a token with `public_repo` access
3. Use: `python clone_all_projects.py --github-token YOUR_TOKEN`

### Resume After Interruption
The script automatically resumes from where it left off. Just run the same command again.

### Disk Space
Ensure you have at least 50 GB of free disk space before starting.

## Output Files

### clone_progress.json
```json
{
  "total_projects": 375,
  "completed": ["owner/repo1", "owner/repo2"],
  "failed": [
    {
      "project": "owner/repo3",
      "error": "Repository not found",
      "timestamp": "2024-08-04T09:30:00"
    }
  ],
  "skipped": ["owner/repo4"],
  "start_time": "2024-08-04T09:00:00",
  "last_update": "2024-08-04T09:30:00"
}
```

### clone_log.txt
Detailed chronological log of all operations with timestamps and error messages.

## Next Step

After successful completion, proceed to Step 2: Commit Analysis.