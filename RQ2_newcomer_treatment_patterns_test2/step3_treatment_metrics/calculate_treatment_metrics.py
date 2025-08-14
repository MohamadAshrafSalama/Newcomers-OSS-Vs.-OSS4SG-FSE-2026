#!/usr/bin/env python3
"""
Script: calculate_treatment_metrics.py
Purpose: Calculate all treatment metrics from contributor timeline data
Input: Individual contributor timeline CSV files
Output: Dataset with treatment metrics per contributor
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import re
import emoji

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TreatmentMetricsCalculator:
    def __init__(self, base_path):
        """Initialize calculator with paths"""
        self.base_path = Path(base_path)
        # Use step2 timelines created in test2 workspace
        self.timeline_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step2_timelines" / "from_cache_timelines"
        # Write outputs into step3 results under test2 workspace
        self.output_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load contributor metadata for project types
        transitions_path = self.base_path / "RQ1_transition_rates_and_speeds" / "step6_contributor_transitions" / "results" / "contributor_transitions.csv"
        self.transitions_df = pd.read_csv(transitions_path)

    def parse_pr_data(self, event_data_json):
        """Parse PR data from JSON string"""
        try:
            return json.loads(event_data_json)
        except Exception:
            return {}

    def parse_issue_data(self, event_data_json):
        """Parse issue data from JSON string"""
        try:
            return json.loads(event_data_json)
        except Exception:
            return {}

    def calculate_response_timing_metrics(self, pr_events, issue_events):
        """Category 1: Response Timing Metrics"""
        metrics = {
            'time_to_first_response_mean': np.nan,
            'time_to_first_response_median': np.nan,
            'avg_response_time_mean': np.nan,
            'avg_response_time_median': np.nan,
            'response_rate': 0.0,
            'weekend_response_rate': 0.0,
            'off_hours_response_rate': 0.0
        }

        first_response_times = []
        all_response_times = []
        total_items_with_response = 0
        total_items = 0
        weekend_responses = 0
        off_hours_responses = 0
        total_responses = 0

        # Process PRs
        for pr_json in pr_events:
            pr_data = self.parse_pr_data(pr_json)
            if not pr_data:
                continue

            total_items += 1
            created_at = pd.to_datetime(pr_data.get('created_at') or pr_data.get('createdAt'))
            conversations = pr_data.get('conversations', [])
            # Fallback: build simple conversation list from comments nodes if present
            if not conversations and isinstance(pr_data.get('comments'), dict) and 'nodes' in pr_data['comments']:
                tmp = []
                for node in pr_data['comments']['nodes']:
                    tmp.append({
                        'author': (node.get('author') or {}).get('login'),
                        'timestamp': node.get('createdAt') or node.get('created_at'),
                        'body': node.get('body')
                    })
                conversations = tmp

            author = (pr_data.get('author') or {}).get('login') if isinstance(pr_data.get('author'), dict) else pr_data.get('author')

            if conversations:
                total_items_with_response += 1

                # First response time (exclude self-responses)
                for conv in conversations:
                    if conv.get('author') and conv.get('author') != author:
                        response_time = pd.to_datetime(conv.get('timestamp', conv.get('created_at')))
                        time_diff = (response_time - created_at).total_seconds() / 3600  # hours
                        first_response_times.append(time_diff)
                        break

                # All response times between messages
                prev_time = created_at
                for conv in conversations:
                    conv_time = pd.to_datetime(conv.get('timestamp', conv.get('created_at')))
                    if conv.get('author') and conv.get('author') != author:
                        time_diff = (conv_time - prev_time).total_seconds() / 3600
                        all_response_times.append(time_diff)
                        total_responses += 1

                        # Check weekend
                        if conv_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                            weekend_responses += 1

                        # Check off-hours (6pm-9am)
                        hour = conv_time.hour
                        if hour >= 18 or hour < 9:
                            off_hours_responses += 1

                    prev_time = conv_time

        # Process Issues (similar logic)
        for issue_json in issue_events:
            issue_data = self.parse_issue_data(issue_json)
            if not issue_data:
                continue

            total_items += 1
            created_at = pd.to_datetime(issue_data.get('created_at') or issue_data.get('createdAt'))
            conversations = issue_data.get('conversations', [])
            if not conversations and isinstance(issue_data.get('comments'), dict) and 'nodes' in issue_data['comments']:
                tmp = []
                for node in issue_data['comments']['nodes']:
                    tmp.append({
                        'author': (node.get('author') or {}).get('login'),
                        'timestamp': node.get('createdAt') or node.get('created_at'),
                        'body': node.get('body')
                    })
                conversations = tmp

            author = (issue_data.get('author') or {}).get('login') if isinstance(issue_data.get('author'), dict) else issue_data.get('author')

            if conversations:
                total_items_with_response += 1

                # First response time
                for conv in conversations:
                    if conv.get('author') and conv.get('author') != author:
                        response_time = pd.to_datetime(conv.get('timestamp', conv.get('created_at')))
                        time_diff = (response_time - created_at).total_seconds() / 3600
                        first_response_times.append(time_diff)
                        break

                # All response times
                prev_time = created_at
                for conv in conversations:
                    conv_time = pd.to_datetime(conv.get('timestamp', conv.get('created_at')))
                    if conv.get('author') and conv.get('author') != author:
                        time_diff = (conv_time - prev_time).total_seconds() / 3600
                        all_response_times.append(time_diff)
                        total_responses += 1

                        if conv_time.weekday() >= 5:
                            weekend_responses += 1

                        hour = conv_time.hour
                        if hour >= 18 or hour < 9:
                            off_hours_responses += 1

                    prev_time = conv_time

        # Calculate metrics
        if first_response_times:
            metrics['time_to_first_response_mean'] = np.mean(first_response_times)
            metrics['time_to_first_response_median'] = np.median(first_response_times)

        if all_response_times:
            metrics['avg_response_time_mean'] = np.mean(all_response_times)
            metrics['avg_response_time_median'] = np.median(all_response_times)

        if total_items > 0:
            metrics['response_rate'] = total_items_with_response / total_items

        if total_responses > 0:
            metrics['weekend_response_rate'] = weekend_responses / total_responses
            metrics['off_hours_response_rate'] = off_hours_responses / total_responses

        return metrics

    def calculate_engagement_breadth_metrics(self, pr_events, issue_events):
        """Category 2: Engagement Breadth Metrics"""
        metrics = {
            'unique_responders': 0,
            'maintainer_response_rate': 0.0,
            'peer_response_rate': 0.0,
            'repeat_engagers': 0,
            'response_diversity_index': 0.0
        }

        all_responders = []
        maintainer_responses = 0
        peer_responses = 0
        total_responses = 0
        pr_responders = {}

        # Process PRs
        for pr_json in pr_events:
            pr_data = self.parse_pr_data(pr_json)
            if not pr_data:
                continue

            pr_number = pr_data.get('number', 'unknown')
            pr_responders[pr_number] = set()

            conversations = pr_data.get('conversations', [])
            if not conversations and isinstance(pr_data.get('comments'), dict) and 'nodes' in pr_data['comments']:
                conversations = [{
                    'author': (n.get('author') or {}).get('login'),
                    'role': (n.get('authorAssociation') or 'NONE'),
                    'body': n.get('body')
                } for n in pr_data['comments']['nodes']]

            for conv in conversations:
                author = conv.get('author')
                role = conv.get('author_role', conv.get('role', 'NONE'))

                if author and author != ((pr_data.get('author') or {}).get('login') if isinstance(pr_data.get('author'), dict) else pr_data.get('author')):
                    all_responders.append(author)
                    pr_responders[pr_number].add(author)
                    total_responses += 1

                    if role in ['MEMBER', 'OWNER']:
                        maintainer_responses += 1
                    elif role == 'CONTRIBUTOR':
                        peer_responses += 1

            # Also check reviews
            reviews = pr_data.get('reviews', [])
            if isinstance(reviews, dict) and 'nodes' in reviews:
                review_nodes = reviews['nodes']
            else:
                review_nodes = reviews
            for review in (review_nodes or []):
                reviewer = (review.get('author') or {}).get('login') if isinstance(review.get('author'), dict) else (review.get('reviewer') or review.get('author'))
                if reviewer and reviewer != ((pr_data.get('author') or {}).get('login') if isinstance(pr_data.get('author'), dict) else pr_data.get('author')):
                    all_responders.append(reviewer)
                    pr_responders[pr_number].add(reviewer)
                    total_responses += 1
                    maintainer_responses += 1  # reviewers typically maintainers

        # Process Issues
        for issue_json in issue_events:
            issue_data = self.parse_issue_data(issue_json)
            if not issue_data:
                continue

            conversations = issue_data.get('conversations', [])
            if not conversations and isinstance(issue_data.get('comments'), dict) and 'nodes' in issue_data['comments']:
                conversations = [{
                    'author': (n.get('author') or {}).get('login'),
                    'role': (n.get('authorAssociation') or 'NONE'),
                    'body': n.get('body')
                } for n in issue_data['comments']['nodes']]

            for conv in conversations:
                author = conv.get('author')
                role = conv.get('author_role', conv.get('role', 'NONE'))

                if author and author != ((issue_data.get('author') or {}).get('login') if isinstance(issue_data.get('author'), dict) else issue_data.get('author')):
                    all_responders.append(author)
                    total_responses += 1

                    if role in ['MEMBER', 'OWNER']:
                        maintainer_responses += 1
                    elif role == 'CONTRIBUTOR':
                        peer_responses += 1

        # Calculate metrics
        unique_responders = list(set(all_responders))
        metrics['unique_responders'] = len(unique_responders)

        if total_responses > 0:
            metrics['maintainer_response_rate'] = maintainer_responses / total_responses
            metrics['peer_response_rate'] = peer_responses / total_responses

        # Count repeat engagers (people who responded to >1 PR)
        responder_pr_counts = {}
        for _, responders in pr_responders.items():
            for responder in responders:
                responder_pr_counts[responder] = responder_pr_counts.get(responder, 0) + 1

        metrics['repeat_engagers'] = sum(1 for count in responder_pr_counts.values() if count > 1)

        # Diversity index (Simpson)
        if all_responders:
            responder_counts = pd.Series(all_responders).value_counts()
            proportions = responder_counts / len(all_responders)
            metrics['response_diversity_index'] = 1 - float((proportions ** 2).sum())

        return metrics

    def calculate_interaction_patterns_metrics(self, pr_events, issue_events):
        """Category 3: Interaction Patterns Metrics"""
        metrics = {
            'conversation_length_mean': 0.0,
            'conversation_length_median': 0.0,
            'back_forth_turns_mean': 0.0,
            'back_forth_turns_median': 0.0,
            'response_word_count_mean': 0.0,
            'response_word_count_median': 0.0,
            'question_rate': 0.0,
            'link_sharing_rate': 0.0
        }

        conversation_lengths = []
        back_forth_counts = []
        word_counts = []
        messages_with_questions = 0
        messages_with_links = 0
        total_messages = 0

        # Process PRs
        for pr_json in pr_events:
            pr_data = self.parse_pr_data(pr_json)
            if not pr_data:
                continue

            conversations = pr_data.get('conversations', [])
            if not conversations and isinstance(pr_data.get('comments'), dict) and 'nodes' in pr_data['comments']:
                conversations = [{
                    'author': (n.get('author') or {}).get('login'),
                    'body': n.get('body')
                } for n in pr_data['comments']['nodes']]

            if conversations:
                conversation_lengths.append(len(conversations))

                # Count back-and-forth turns
                prev_author = (pr_data.get('author') or {}).get('login') if isinstance(pr_data.get('author'), dict) else pr_data.get('author')
                turns = 0
                for conv in conversations:
                    curr_author = conv.get('author')
                    if curr_author != prev_author:
                        turns += 1
                        prev_author = curr_author
                back_forth_counts.append(turns)

                # Analyze message content
                for conv in conversations:
                    text = conv.get('text', conv.get('body', ''))
                    if text:
                        total_messages += 1
                        word_count = len(text.split())
                        word_counts.append(word_count)

                        if '?' in text:
                            messages_with_questions += 1

                        if 'http://' in text or 'https://' in text:
                            messages_with_links += 1

        # Process Issues
        for issue_json in issue_events:
            issue_data = self.parse_issue_data(issue_json)
            if not issue_data:
                continue

            conversations = issue_data.get('conversations', [])
            if not conversations and isinstance(issue_data.get('comments'), dict) and 'nodes' in issue_data['comments']:
                conversations = [{
                    'author': (n.get('author') or {}).get('login'),
                    'body': n.get('body')
                } for n in issue_data['comments']['nodes']]

            if conversations:
                conversation_lengths.append(len(conversations))

                # Count turns
                prev_author = (issue_data.get('author') or {}).get('login') if isinstance(issue_data.get('author'), dict) else issue_data.get('author')
                turns = 0
                for conv in conversations:
                    curr_author = conv.get('author')
                    if curr_author != prev_author:
                        turns += 1
                        prev_author = curr_author
                back_forth_counts.append(turns)

                # Analyze content
                for conv in conversations:
                    text = conv.get('text', conv.get('body', ''))
                    if text:
                        total_messages += 1
                        word_count = len(text.split())
                        word_counts.append(word_count)

                        if '?' in text:
                            messages_with_questions += 1

                        if 'http://' in text or 'https://' in text:
                            messages_with_links += 1

        # Calculate metrics
        if conversation_lengths:
            metrics['conversation_length_mean'] = float(np.mean(conversation_lengths))
            metrics['conversation_length_median'] = float(np.median(conversation_lengths))

        if back_forth_counts:
            metrics['back_forth_turns_mean'] = float(np.mean(back_forth_counts))
            metrics['back_forth_turns_median'] = float(np.median(back_forth_counts))

        if word_counts:
            metrics['response_word_count_mean'] = float(np.mean(word_counts))
            metrics['response_word_count_median'] = float(np.median(word_counts))

        if total_messages > 0:
            metrics['question_rate'] = messages_with_questions / total_messages
            metrics['link_sharing_rate'] = messages_with_links / total_messages

        return metrics

    def calculate_recognition_signals_metrics(self, pr_events, issue_events):
        """Category 4: Recognition Signals Metrics"""
        metrics = {
            'thanks_rate': 0.0,
            'emoji_usage_rate': 0.0,
            'approval_speed_mean': np.nan,
            'approval_speed_median': np.nan,
            'merge_rate': 0.0,
            'author_attribution_rate': 0.0
        }

        messages_with_thanks = 0
        messages_with_emoji = 0
        total_messages = 0
        approval_speeds = []
        total_prs = 0
        merged_prs = 0
        merged_with_attribution = 0

        thanks_patterns = re.compile(r'\b(thank|thanks|appreciate|great|excellent|awesome|good job|well done)\b', re.IGNORECASE)

        # Process PRs
        for pr_json in pr_events:
            pr_data = self.parse_pr_data(pr_json)
            if not pr_data:
                continue

            total_prs += 1
            created_at = pd.to_datetime(pr_data.get('created_at') or pr_data.get('createdAt'))
            pr_author = (pr_data.get('author') or {}).get('login') if isinstance(pr_data.get('author'), dict) else pr_data.get('author')

            # Check if merged
            state = pr_data.get('state') or pr_data.get('stateStr')
            if (state == 'MERGED') or bool(pr_data.get('merged')):
                merged_prs += 1

                # Simplified attribution check
                merged_by = (pr_data.get('mergedBy') or {}).get('login') if isinstance(pr_data.get('mergedBy'), dict) else pr_data.get('merged_by')
                if merged_by and merged_by != pr_author:
                    merged_with_attribution += 1

            # Analyze conversations for thanks and emojis
            conversations = pr_data.get('conversations', [])
            if not conversations and isinstance(pr_data.get('comments'), dict) and 'nodes' in pr_data['comments']:
                conversations = [{
                    'author': (n.get('author') or {}).get('login'),
                    'body': n.get('body')
                } for n in pr_data['comments']['nodes']]
            for conv in conversations:
                text = conv.get('text', conv.get('body', ''))
                if text and conv.get('author') != pr_author:
                    total_messages += 1

                    if thanks_patterns.search(text):
                        messages_with_thanks += 1

                    if emoji.emoji_count(text) > 0:
                        messages_with_emoji += 1

            # Check review approval speed
            reviews = pr_data.get('reviews', [])
            if isinstance(reviews, dict) and 'nodes' in reviews:
                review_nodes = reviews['nodes']
            else:
                review_nodes = reviews
            for review in (review_nodes or []):
                if (review.get('state') or review.get('stateStr')) == 'APPROVED':
                    approved_at = pd.to_datetime(review.get('submitted_at') or review.get('submittedAt'))
                    if created_at is not None and approved_at is not None:
                        speed = (approved_at - created_at).total_seconds() / 3600  # hours
                        approval_speeds.append(speed)

        # Process Issues
        for issue_json in issue_events:
            issue_data = self.parse_issue_data(issue_json)
            if not issue_data:
                continue

            issue_author = (issue_data.get('author') or {}).get('login') if isinstance(issue_data.get('author'), dict) else issue_data.get('author')
            conversations = issue_data.get('conversations', [])
            if not conversations and isinstance(issue_data.get('comments'), dict) and 'nodes' in issue_data['comments']:
                conversations = [{
                    'author': (n.get('author') or {}).get('login'),
                    'body': n.get('body')
                } for n in issue_data['comments']['nodes']]

            for conv in conversations:
                text = conv.get('text', conv.get('body', ''))
                if text and conv.get('author') != issue_author:
                    total_messages += 1

                    if thanks_patterns.search(text):
                        messages_with_thanks += 1

                    if emoji.emoji_count(text) > 0:
                        messages_with_emoji += 1

        # Calculate metrics
        if total_messages > 0:
            metrics['thanks_rate'] = messages_with_thanks / total_messages
            metrics['emoji_usage_rate'] = messages_with_emoji / total_messages

        if approval_speeds:
            metrics['approval_speed_mean'] = float(np.mean(approval_speeds))
            metrics['approval_speed_median'] = float(np.median(approval_speeds))

        if total_prs > 0:
            metrics['merge_rate'] = merged_prs / total_prs

        if merged_prs > 0:
            metrics['author_attribution_rate'] = merged_with_attribution / merged_prs

        return metrics

    def calculate_trust_indicators_metrics(self, pr_events, issue_events):
        """Category 5: Trust Indicators Metrics"""
        metrics = {
            'review_requests_received': 0,
            'at_mentions': 0,
            'issue_assignments': 0,
            'label_additions': 0,
            're_engagement_rate': 0.0
        }

        total_prs = 0
        prs_with_multiple_cycles = 0

        # Process PRs
        for pr_json in pr_events:
            pr_data = self.parse_pr_data(pr_json)
            if not pr_data:
                continue

            total_prs += 1

            # Review requests
            requested = pr_data.get('requested_reviewers') or []
            if isinstance(requested, dict) and 'nodes' in requested:
                metrics['review_requests_received'] += len(requested['nodes'])
            elif isinstance(requested, list):
                metrics['review_requests_received'] += len(requested)

            # Labels
            labels = pr_data.get('labels')
            if isinstance(labels, dict) and 'nodes' in labels and labels['nodes']:
                metrics['label_additions'] += 1
            elif isinstance(labels, list) and labels:
                metrics['label_additions'] += 1

            # Multiple review cycles
            reviews = pr_data.get('reviews', [])
            if isinstance(reviews, dict) and 'nodes' in reviews:
                review_nodes = reviews['nodes']
            else:
                review_nodes = reviews
            if isinstance(review_nodes, list) and len(review_nodes) > 1:
                prs_with_multiple_cycles += 1

            # @mentions in conversations
            conversations = pr_data.get('conversations', [])
            if not conversations and isinstance(pr_data.get('comments'), dict) and 'nodes' in pr_data['comments']:
                conversations = [{
                    'body': n.get('body')
                } for n in pr_data['comments']['nodes']]
            for conv in conversations:
                text = conv.get('text', conv.get('body', ''))
                if text:
                    mentions = re.findall(r'@[\w-]+', text)
                    metrics['at_mentions'] += len(mentions)

        # Process Issues
        for issue_json in issue_events:
            issue_data = self.parse_issue_data(issue_json)
            if not issue_data:
                continue

            # Assignments
            assignees = issue_data.get('assignees')
            if isinstance(assignees, dict) and 'nodes' in assignees and assignees['nodes']:
                metrics['issue_assignments'] += 1
            elif isinstance(assignees, list) and assignees:
                metrics['issue_assignments'] += 1

            # @mentions
            conversations = issue_data.get('conversations', [])
            if not conversations and isinstance(issue_data.get('comments'), dict) and 'nodes' in issue_data['comments']:
                conversations = [{
                    'body': n.get('body')
                } for n in issue_data['comments']['nodes']]
            for conv in conversations:
                text = conv.get('text', conv.get('body', ''))
                if text:
                    mentions = re.findall(r'@[\w-]+', text)
                    metrics['at_mentions'] += len(mentions)

        # Re-engagement rate
        if total_prs > 0:
            metrics['re_engagement_rate'] = prs_with_multiple_cycles / total_prs

        return metrics

    def process_contributor_timeline(self, timeline_file):
        """Process one contributor's timeline and calculate all metrics"""

        # Load timeline
        timeline_df = pd.read_csv(timeline_file)

        # Filter to pre-core events only
        pre_core_df = timeline_df[timeline_df['is_pre_core'] == True].copy()

        # Separate PR and Issue events
        pr_events = pre_core_df[pre_core_df['event_type'] == 'pull_request']['event_data'].tolist()
        issue_events = pre_core_df[pre_core_df['event_type'] == 'issue']['event_data'].tolist()

        # Calculate all metric categories
        metrics = {}

        # Category 1: Response Timing
        timing_metrics = self.calculate_response_timing_metrics(pr_events, issue_events)
        metrics.update(timing_metrics)

        # Category 2: Engagement Breadth
        engagement_metrics = self.calculate_engagement_breadth_metrics(pr_events, issue_events)
        metrics.update(engagement_metrics)

        # Category 3: Interaction Patterns
        interaction_metrics = self.calculate_interaction_patterns_metrics(pr_events, issue_events)
        metrics.update(interaction_metrics)

        # Category 4: Recognition Signals
        recognition_metrics = self.calculate_recognition_signals_metrics(pr_events, issue_events)
        metrics.update(recognition_metrics)

        # Category 5: Trust Indicators
        trust_metrics = self.calculate_trust_indicators_metrics(pr_events, issue_events)
        metrics.update(trust_metrics)

        # Add metadata
        project_name = timeline_df['project_name'].iloc[0] if not timeline_df.empty else 'unknown'

        # Extract contributor email from filename
        filename = timeline_file.stem  # timeline_<id>.csv
        # We already have the contributor_email column; prefer it if present
        contributor_email = timeline_df['contributor_email'].iloc[0] if 'contributor_email' in timeline_df.columns and not timeline_df.empty else 'unknown'

        metrics['contributor_email'] = contributor_email
        metrics['project_name'] = project_name

        # Get project type from transitions data
        contributor_info = self.transitions_df[
            (self.transitions_df['contributor_email'].str.lower() == str(contributor_email).lower()) &
            (self.transitions_df['project_name'] == project_name)
        ]

        if not contributor_info.empty:
            metrics['project_type'] = contributor_info['project_type'].iloc[0]
        else:
            metrics['project_type'] = 'unknown'

        # Add counts for context
        metrics['total_prs'] = len(pr_events)
        metrics['total_issues'] = len(issue_events)
        metrics['total_commits'] = len(pre_core_df[pre_core_df['event_type'] == 'commit'])

        return metrics

    def process_all_contributors(self):
        """Process all contributor timelines and create treatment metrics dataset"""

        # Get all timeline files
        timeline_files = list(self.timeline_dir.glob("timeline_*.csv"))
        logger.info(f"Found {len(timeline_files)} timeline files to process")

        all_metrics = []

        # Process each timeline
        for timeline_file in tqdm(timeline_files, desc="Processing timelines"):
            try:
                metrics = self.process_contributor_timeline(timeline_file)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error processing {timeline_file}: {e}")
                continue

        # Create DataFrame
        metrics_df = pd.DataFrame(all_metrics)

        # Save to CSV
        output_file = self.output_dir / "treatment_metrics_per_contributor.csv"
        metrics_df.to_csv(output_file, index=False)

        logger.info(f"Saved treatment metrics for {len(metrics_df)} contributors to {output_file}")

        # Calculate summary statistics by project type
        self.generate_summary_statistics(metrics_df)

        return metrics_df

    def generate_summary_statistics(self, metrics_df):
        """Generate summary statistics for OSS vs OSS4SG comparison"""

        # Group by project type
        summary_stats = {}

        for project_type in ['OSS', 'OSS4SG']:
            type_df = metrics_df[metrics_df['project_type'] == project_type]

            stats = {}
            # For each numeric column, calculate mean and median
            numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col not in ['total_prs', 'total_issues', 'total_commits']:
                    stats[f"{col}_mean"] = type_df[col].mean()
                    stats[f"{col}_median"] = type_df[col].median()
                    stats[f"{col}_std"] = type_df[col].std()

            stats['n_contributors'] = len(type_df)
            summary_stats[project_type] = stats

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(summary_stats).T

        # Save summary
        summary_file = self.output_dir / "treatment_metrics_summary_by_type.csv"
        comparison_df.to_csv(summary_file)

        logger.info(f"Saved summary statistics to {summary_file}")

        # Print key findings
        print("\n" + "="*60)
        print("KEY TREATMENT DIFFERENCES (OSS vs OSS4SG)")
        print("="*60)

        key_metrics = [
            'time_to_first_response_median',
            'response_rate',
            'unique_responders',
            'thanks_rate',
            'merge_rate'
        ]

        for metric in key_metrics:
            if metric in comparison_df.columns:
                oss_val = comparison_df.loc['OSS', metric] if 'OSS' in comparison_df.index else np.nan
                oss4sg_val = comparison_df.loc['OSS4SG', metric] if 'OSS4SG' in comparison_df.index else np.nan
                if pd.notnull(oss_val) and oss_val != 0:
                    diff_pct = (oss4sg_val - oss_val) / oss_val * 100
                else:
                    diff_pct = 0

                print(f"\n{metric}:")
                print(f"  OSS: {oss_val:.2f}" if pd.notnull(oss_val) else "  OSS: NA")
                print(f"  OSS4SG: {oss4sg_val:.2f}" if pd.notnull(oss4sg_val) else "  OSS4SG: NA")
                print(f"  Difference: {diff_pct:+.1f}%")


def main():
    """Main execution function"""

    # Set base path
    base_path = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"

    # Create calculator
    calculator = TreatmentMetricsCalculator(base_path)

    # Process all contributors
    metrics_df = calculator.process_all_contributors()

    print("\n" + "="*50)
    print("TREATMENT METRICS CALCULATION COMPLETE")
    print("="*50)
    print(f"Processed {len(metrics_df)} contributors")
    print(f"Output saved to: {calculator.output_dir}")


if __name__ == "__main__":
    main()


