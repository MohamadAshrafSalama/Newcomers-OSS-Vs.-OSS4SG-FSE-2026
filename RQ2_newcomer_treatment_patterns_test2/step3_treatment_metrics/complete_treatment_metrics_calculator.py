#!/usr/bin/env python3
"""
COMPLETE Treatment Metrics Dataset Creator for RQ2
- Calculates ALL possible treatment metrics (60+ metrics)
- Handles every edge case and zero-value scenario
- No shortcuts - every metric category fully implemented
- Bulletproof error handling and validation
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import re
import sys
import warnings
import subprocess

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteTreatmentMetricsCalculator:
    def __init__(self, base_path):
        """Initialize calculator with full validation"""
        self.base_path = Path(base_path)
        
        # Paths
        self.timeline_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step2_timelines" / "from_cache_timelines"
        self.output_dir = self.base_path / "RQ2_newcomer_treatment_patterns_test2" / "step3_treatment_metrics" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Install and validate dependencies
        self.validate_and_install_dependencies()
        
        # Load transitions data
        transitions_path = self.base_path / "RQ1_transition_rates_and_speeds" / "step6_contributor_transitions" / "results" / "contributor_transitions.csv"
        self.transitions_df = pd.read_csv(transitions_path)
        
        # Debug stats
        self.debug_stats = {
            'total_contributors': 0,
            'contributors_with_prs': 0,
            'contributors_with_issues': 0,
            'contributors_with_commits': 0,
            'parsing_errors': 0,
            'empty_timelines': 0,
            'successful_calculations': 0
        }

    def validate_and_install_dependencies(self):
        """Install and validate ALL required dependencies"""
        logger.info("Validating and installing dependencies...")
        
        required_packages = [
            ('emoji', 'emoji'),
            ('textstat', 'textstat'),
            ('langdetect', 'langdetect')
        ]
        
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                logger.info(f"✓ {package_name} available")
            except ImportError:
                logger.info(f"Installing {package_name}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
                logger.info(f"✓ {package_name} installed")
        
        # Validate timeline directory
        if not self.timeline_dir.exists():
            raise FileNotFoundError(f"Timeline directory not found: {self.timeline_dir}")
        
        timeline_files = list(self.timeline_dir.glob("timeline_*.csv"))
        if len(timeline_files) == 0:
            raise ValueError(f"No timeline files found in: {self.timeline_dir}")
        
        logger.info(f"Found {len(timeline_files)} timeline files to process")

    def safe_parse_json(self, json_string):
        """Safely parse JSON with multiple fallback strategies"""
        if pd.isna(json_string) or json_string == '' or json_string is None:
            return {}
        
        try:
            # Try direct parsing
            return json.loads(json_string)
        except (json.JSONDecodeError, TypeError, ValueError):
            try:
                # Try with string conversion
                return json.loads(str(json_string))
            except:
                return {}

    def extract_conversations_comprehensive(self, data):
        """Extract conversations with ALL possible fallback methods"""
        conversations = []
        
        if not isinstance(data, dict):
            return conversations
        
        # Method 1: Direct conversations field
        if data.get('conversations') and isinstance(data['conversations'], list):
            return data['conversations']
        
        # Method 2: Comments with nodes structure (GraphQL)
        if isinstance(data.get('comments'), dict) and 'nodes' in data['comments']:
            for node in data['comments']['nodes']:
                if node:
                    conv = {
                        'author': self.extract_author(node),
                        'timestamp': node.get('createdAt') or node.get('created_at'),
                        'body': node.get('body', ''),
                        'role': node.get('authorAssociation', 'NONE')
                    }
                    conversations.append(conv)
        
        # Method 3: Direct comments list (REST API)
        elif isinstance(data.get('comments'), list):
            for comment in data['comments']:
                if comment:
                    conv = {
                        'author': self.extract_author(comment),
                        'timestamp': comment.get('createdAt') or comment.get('created_at'),
                        'body': comment.get('body', ''),
                        'role': comment.get('authorAssociation', 'NONE')
                    }
                    conversations.append(conv)
        
        # Method 4: Timeline events as conversations
        if data.get('timelineEvents') and isinstance(data['timelineEvents'], list):
            for event in data['timelineEvents']:
                if event and event.get('__typename') in ['IssueComment', 'PullRequestReview']:
                    conv = {
                        'author': self.extract_author(event),
                        'timestamp': event.get('createdAt') or event.get('created_at'),
                        'body': event.get('body', ''),
                        'role': event.get('authorAssociation', 'NONE')
                    }
                    conversations.append(conv)
        
        return conversations

    def extract_author(self, item):
        """Extract author with multiple fallback strategies"""
        if not item:
            return None
            
        # Method 1: author.login
        author = item.get('author')
        if isinstance(author, dict) and author.get('login'):
            return author['login']
        
        # Method 2: direct author string
        if isinstance(author, str):
            return author
        
        # Method 3: user.login
        user = item.get('user')
        if isinstance(user, dict) and user.get('login'):
            return user['login']
        
        # Method 4: actor.login
        actor = item.get('actor')
        if isinstance(actor, dict) and actor.get('login'):
            return actor['login']
        
        return None

    def calculate_response_timing_metrics_complete(self, pr_events, issue_events):
        """Calculate ALL response timing metrics - COMPLETE IMPLEMENTATION"""
        
        # Initialize collectors
        first_response_times = []
        all_response_times = []
        response_delays = []
        total_items = 0
        items_with_responses = 0
        weekend_responses = 0
        off_hours_responses = 0
        business_hours_responses = 0
        same_day_responses = 0
        within_hour_responses = 0
        total_responses = 0
        
        # Process PRs
        for pr_json in pr_events:
            pr_data = self.safe_parse_json(pr_json)
            if not pr_data:
                continue
                
            total_items += 1
            created_at = pd.to_datetime(pr_data.get('created_at') or pr_data.get('createdAt'))
            if pd.isna(created_at):
                continue
            
            pr_author = self.extract_author(pr_data)
            conversations = self.extract_conversations_comprehensive(pr_data)
            
            if conversations:
                items_with_responses += 1
                first_response_found = False
                
                for conv in conversations:
                    conv_author = conv.get('author')
                    if not conv_author or conv_author == pr_author:
                        continue
                    
                    try:
                        response_time = pd.to_datetime(conv.get('timestamp'))
                        if pd.isna(response_time):
                            continue
                        
                        time_diff_hours = (response_time - created_at).total_seconds() / 3600
                        if time_diff_hours < 0:
                            continue
                        
                        # First response time
                        if not first_response_found:
                            first_response_times.append(time_diff_hours)
                            first_response_found = True
                        
                        # All response times
                        all_response_times.append(time_diff_hours)
                        total_responses += 1
                        
                        # Time-based classifications
                        if response_time.weekday() >= 5:
                            weekend_responses += 1
                        
                        hour = response_time.hour
                        if hour < 9 or hour >= 18:
                            off_hours_responses += 1
                        else:
                            business_hours_responses += 1
                        
                        if time_diff_hours <= 1:
                            within_hour_responses += 1
                        
                        if time_diff_hours <= 24:
                            same_day_responses += 1
                            
                    except Exception as e:
                        logger.debug(f"Error processing PR conversation: {e}")
                        continue
        
        # Process Issues (same logic)
        for issue_json in issue_events:
            issue_data = self.safe_parse_json(issue_json)
            if not issue_data:
                continue
                
            total_items += 1
            created_at = pd.to_datetime(issue_data.get('created_at') or issue_data.get('createdAt'))
            if pd.isna(created_at):
                continue
            
            issue_author = self.extract_author(issue_data)
            conversations = self.extract_conversations_comprehensive(issue_data)
            
            if conversations:
                items_with_responses += 1
                first_response_found = False
                
                for conv in conversations:
                    conv_author = conv.get('author')
                    if not conv_author or conv_author == issue_author:
                        continue
                    
                    try:
                        response_time = pd.to_datetime(conv.get('timestamp'))
                        if pd.isna(response_time):
                            continue
                        
                        time_diff_hours = (response_time - created_at).total_seconds() / 3600
                        if time_diff_hours < 0:
                            continue
                        
                        if not first_response_found:
                            first_response_times.append(time_diff_hours)
                            first_response_found = True
                        
                        all_response_times.append(time_diff_hours)
                        total_responses += 1
                        
                        if response_time.weekday() >= 5:
                            weekend_responses += 1
                        
                        hour = response_time.hour
                        if hour < 9 or hour >= 18:
                            off_hours_responses += 1
                        else:
                            business_hours_responses += 1
                        
                        if time_diff_hours <= 1:
                            within_hour_responses += 1
                        
                        if time_diff_hours <= 24:
                            same_day_responses += 1
                            
                    except Exception as e:
                        logger.debug(f"Error processing issue conversation: {e}")
                        continue
        
        # Calculate ALL metrics
        metrics = {}
        
        # Basic response metrics
        metrics['total_items'] = total_items
        metrics['items_with_responses'] = items_with_responses
        metrics['response_rate'] = items_with_responses / total_items if total_items > 0 else 0.0
        metrics['total_responses'] = total_responses
        
        # First response time metrics
        if first_response_times:
            metrics['first_response_count'] = len(first_response_times)
            metrics['first_response_mean_hours'] = float(np.mean(first_response_times))
            metrics['first_response_median_hours'] = float(np.median(first_response_times))
            metrics['first_response_std_hours'] = float(np.std(first_response_times))
            metrics['first_response_min_hours'] = float(np.min(first_response_times))
            metrics['first_response_max_hours'] = float(np.max(first_response_times))
            metrics['first_response_25th_percentile'] = float(np.percentile(first_response_times, 25))
            metrics['first_response_75th_percentile'] = float(np.percentile(first_response_times, 75))
        else:
            for key in ['first_response_count', 'first_response_mean_hours', 'first_response_median_hours', 
                       'first_response_std_hours', 'first_response_min_hours', 'first_response_max_hours',
                       'first_response_25th_percentile', 'first_response_75th_percentile']:
                metrics[key] = 0 if 'count' in key else np.nan
        
        # All response time metrics
        if all_response_times:
            metrics['avg_response_mean_hours'] = float(np.mean(all_response_times))
            metrics['avg_response_median_hours'] = float(np.median(all_response_times))
            metrics['avg_response_std_hours'] = float(np.std(all_response_times))
        else:
            metrics['avg_response_mean_hours'] = np.nan
            metrics['avg_response_median_hours'] = np.nan
            metrics['avg_response_std_hours'] = np.nan
        
        # Time-based response rates
        if total_responses > 0:
            metrics['weekend_response_rate'] = weekend_responses / total_responses
            metrics['off_hours_response_rate'] = off_hours_responses / total_responses
            metrics['business_hours_response_rate'] = business_hours_responses / total_responses
            metrics['within_hour_response_rate'] = within_hour_responses / total_responses
            metrics['same_day_response_rate'] = same_day_responses / total_responses
        else:
            for key in ['weekend_response_rate', 'off_hours_response_rate', 'business_hours_response_rate',
                       'within_hour_response_rate', 'same_day_response_rate']:
                metrics[key] = 0.0
        
        return metrics

    def calculate_engagement_breadth_metrics_complete(self, pr_events, issue_events):
        """Calculate ALL engagement breadth metrics - COMPLETE IMPLEMENTATION"""
        
        # Initialize collectors
        all_responders = []
        maintainer_responses = 0
        peer_responses = 0
        external_responses = 0
        total_responses = 0
        pr_responders = {}
        unique_pr_responders = set()
        responder_pr_counts = {}
        responder_roles = {}
        
        # Process PRs
        for pr_json in pr_events:
            pr_data = self.safe_parse_json(pr_json)
            if not pr_data:
                continue
            
            pr_number = pr_data.get('number', f'pr_{len(pr_responders)}')
            pr_responders[pr_number] = set()
            pr_author = self.extract_author(pr_data)
            
            # Process conversations
            conversations = self.extract_conversations_comprehensive(pr_data)
            for conv in conversations:
                author = conv.get('author')
                role = conv.get('role', 'NONE')
                
                if author and author != pr_author:
                    all_responders.append(author)
                    pr_responders[pr_number].add(author)
                    unique_pr_responders.add(author)
                    total_responses += 1
                    
                    # Track responder roles
                    if author not in responder_roles:
                        responder_roles[author] = role
                    
                    # Categorize responses
                    if role in ['MEMBER', 'OWNER', 'MAINTAINER', 'ADMIN']:
                        maintainer_responses += 1
                    elif role in ['CONTRIBUTOR', 'COLLABORATOR']:
                        peer_responses += 1
                    else:
                        external_responses += 1
            
            # Process reviews
            reviews = pr_data.get('reviews', [])
            if isinstance(reviews, dict) and 'nodes' in reviews:
                review_nodes = reviews['nodes']
            elif isinstance(reviews, list):
                review_nodes = reviews
            else:
                review_nodes = []
            
            for review in review_nodes:
                if review:
                    reviewer = self.extract_author(review)
                    if reviewer and reviewer != pr_author:
                        all_responders.append(reviewer)
                        pr_responders[pr_number].add(reviewer)
                        unique_pr_responders.add(reviewer)
                        total_responses += 1
                        maintainer_responses += 1  # Reviewers are typically maintainers
        
        # Process Issues
        for issue_json in issue_events:
            issue_data = self.safe_parse_json(issue_json)
            if not issue_data:
                continue
            
            issue_author = self.extract_author(issue_data)
            conversations = self.extract_conversations_comprehensive(issue_data)
            
            for conv in conversations:
                author = conv.get('author')
                role = conv.get('role', 'NONE')
                
                if author and author != issue_author:
                    all_responders.append(author)
                    total_responses += 1
                    
                    if author not in responder_roles:
                        responder_roles[author] = role
                    
                    if role in ['MEMBER', 'OWNER', 'MAINTAINER', 'ADMIN']:
                        maintainer_responses += 1
                    elif role in ['CONTRIBUTOR', 'COLLABORATOR']:
                        peer_responses += 1
                    else:
                        external_responses += 1
        
        # Count repeat engagers
        for pr_num, responders in pr_responders.items():
            for responder in responders:
                responder_pr_counts[responder] = responder_pr_counts.get(responder, 0) + 1
        
        # Calculate ALL metrics
        metrics = {}
        
        unique_responders = list(set(all_responders))
        metrics['unique_responders'] = len(unique_responders)
        metrics['total_responses'] = total_responses
        metrics['unique_pr_responders'] = len(unique_pr_responders)
        
        # Response type rates
        if total_responses > 0:
            metrics['maintainer_response_rate'] = maintainer_responses / total_responses
            metrics['peer_response_rate'] = peer_responses / total_responses
            metrics['external_response_rate'] = external_responses / total_responses
        else:
            metrics['maintainer_response_rate'] = 0.0
            metrics['peer_response_rate'] = 0.0
            metrics['external_response_rate'] = 0.0
        
        # Engagement patterns
        metrics['repeat_engagers'] = sum(1 for count in responder_pr_counts.values() if count > 1)
        metrics['avg_prs_per_responder'] = np.mean(list(responder_pr_counts.values())) if responder_pr_counts else 0.0
        metrics['max_prs_per_responder'] = max(responder_pr_counts.values()) if responder_pr_counts else 0
        
        # Diversity metrics
        if all_responders:
            responder_counts = pd.Series(all_responders).value_counts()
            proportions = responder_counts / len(all_responders)
            metrics['response_diversity_index'] = float(1 - (proportions ** 2).sum())  # Simpson's index
            metrics['response_concentration_ratio'] = proportions.iloc[0] if len(proportions) > 0 else 0.0
        else:
            metrics['response_diversity_index'] = 0.0
            metrics['response_concentration_ratio'] = 0.0
        
        # Role diversity
        role_counts = pd.Series(list(responder_roles.values())).value_counts()
        metrics['unique_responder_roles'] = len(role_counts)
        if len(role_counts) > 0:
            role_proportions = role_counts / len(responder_roles)
            metrics['role_diversity_index'] = float(1 - (role_proportions ** 2).sum())
        else:
            metrics['role_diversity_index'] = 0.0
        
        return metrics

    def calculate_interaction_patterns_metrics_complete(self, pr_events, issue_events):
        """Calculate ALL interaction pattern metrics - COMPLETE IMPLEMENTATION"""
        
        # Initialize collectors
        conversation_lengths = []
        back_forth_counts = []
        word_counts = []
        sentence_counts = []
        avg_words_per_message = []
        messages_with_questions = 0
        messages_with_links = 0
        messages_with_code = 0
        messages_with_mentions = 0
        total_messages = 0
        code_snippets = 0
        external_links = 0
        internal_links = 0
        
        # Process PRs
        for pr_json in pr_events:
            pr_data = self.safe_parse_json(pr_json)
            if not pr_data:
                continue
            
            pr_author = self.extract_author(pr_data)
            conversations = self.extract_conversations_comprehensive(pr_data)
            
            if conversations:
                conversation_lengths.append(len(conversations))
                
                # Count back-and-forth turns
                prev_author = pr_author
                turns = 0
                for conv in conversations:
                    curr_author = conv.get('author')
                    if curr_author and curr_author != prev_author:
                        turns += 1
                        prev_author = curr_author
                back_forth_counts.append(turns)
                
                # Analyze message content
                for conv in conversations:
                    text = conv.get('body', '')
                    if text:
                        total_messages += 1
                        
                        # Word and sentence analysis
                        words = str(text).split()
                        word_count = len(words)
                        word_counts.append(word_count)
                        
                        sentences = text.count('.') + text.count('!') + text.count('?')
                        sentence_counts.append(max(1, sentences))
                        
                        if word_count > 0:
                            avg_words_per_message.append(word_count)
                        
                        # Content pattern analysis
                        if '?' in text:
                            messages_with_questions += 1
                        
                        # Link analysis
                        if 'http://' in text or 'https://' in text:
                            messages_with_links += 1
                            if 'github.com' in text:
                                internal_links += 1
                            else:
                                external_links += 1
                        
                        # Code analysis
                        if '```' in text or '`' in text:
                            messages_with_code += 1
                            if '```' in text:
                                code_snippets += text.count('```') // 2
                        
                        # Mention analysis
                        if '@' in text:
                            mentions = len(re.findall(r'@[\w-]+', text))
                            if mentions > 0:
                                messages_with_mentions += 1
        
        # Process Issues (same logic)
        for issue_json in issue_events:
            issue_data = self.safe_parse_json(issue_json)
            if not issue_data:
                continue
            
            issue_author = self.extract_author(issue_data)
            conversations = self.extract_conversations_comprehensive(issue_data)
            
            if conversations:
                conversation_lengths.append(len(conversations))
                
                prev_author = issue_author
                turns = 0
                for conv in conversations:
                    curr_author = conv.get('author')
                    if curr_author and curr_author != prev_author:
                        turns += 1
                        prev_author = curr_author
                back_forth_counts.append(turns)
                
                for conv in conversations:
                    text = conv.get('body', '')
                    if text:
                        total_messages += 1
                        
                        words = str(text).split()
                        word_count = len(words)
                        word_counts.append(word_count)
                        
                        sentences = text.count('.') + text.count('!') + text.count('?')
                        sentence_counts.append(max(1, sentences))
                        
                        if word_count > 0:
                            avg_words_per_message.append(word_count)
                        
                        if '?' in text:
                            messages_with_questions += 1
                        
                        if 'http://' in text or 'https://' in text:
                            messages_with_links += 1
                            if 'github.com' in text:
                                internal_links += 1
                            else:
                                external_links += 1
                        
                        if '```' in text or '`' in text:
                            messages_with_code += 1
                            if '```' in text:
                                code_snippets += text.count('```') // 2
                        
                        if '@' in text:
                            mentions = len(re.findall(r'@[\w-]+', text))
                            if mentions > 0:
                                messages_with_mentions += 1
        
        # Calculate ALL metrics
        metrics = {}
        
        # Conversation structure metrics
        if conversation_lengths:
            metrics['conversation_length_mean'] = float(np.mean(conversation_lengths))
            metrics['conversation_length_median'] = float(np.median(conversation_lengths))
            metrics['conversation_length_std'] = float(np.std(conversation_lengths))
            metrics['conversation_length_max'] = float(np.max(conversation_lengths))
            metrics['conversation_length_min'] = float(np.min(conversation_lengths))
        else:
            for key in ['conversation_length_mean', 'conversation_length_median', 'conversation_length_std',
                       'conversation_length_max', 'conversation_length_min']:
                metrics[key] = np.nan
        
        if back_forth_counts:
            metrics['back_forth_turns_mean'] = float(np.mean(back_forth_counts))
            metrics['back_forth_turns_median'] = float(np.median(back_forth_counts))
            metrics['back_forth_turns_std'] = float(np.std(back_forth_counts))
            metrics['back_forth_turns_max'] = float(np.max(back_forth_counts))
        else:
            for key in ['back_forth_turns_mean', 'back_forth_turns_median', 'back_forth_turns_std', 'back_forth_turns_max']:
                metrics[key] = np.nan
        
        # Text analysis metrics
        if word_counts:
            metrics['word_count_mean'] = float(np.mean(word_counts))
            metrics['word_count_median'] = float(np.median(word_counts))
            metrics['word_count_std'] = float(np.std(word_counts))
            metrics['word_count_max'] = float(np.max(word_counts))
        else:
            for key in ['word_count_mean', 'word_count_median', 'word_count_std', 'word_count_max']:
                metrics[key] = np.nan
        
        if sentence_counts:
            metrics['sentence_count_mean'] = float(np.mean(sentence_counts))
            metrics['sentence_count_median'] = float(np.median(sentence_counts))
        else:
            metrics['sentence_count_mean'] = np.nan
            metrics['sentence_count_median'] = np.nan
        
        # Content pattern rates
        metrics['total_messages'] = total_messages
        if total_messages > 0:
            metrics['question_rate'] = messages_with_questions / total_messages
            metrics['link_sharing_rate'] = messages_with_links / total_messages
            metrics['code_sharing_rate'] = messages_with_code / total_messages
            metrics['mention_rate'] = messages_with_mentions / total_messages
        else:
            for key in ['question_rate', 'link_sharing_rate', 'code_sharing_rate', 'mention_rate']:
                metrics[key] = 0.0
        
        # Advanced content metrics
        metrics['code_snippets_total'] = code_snippets
        metrics['external_links_total'] = external_links
        metrics['internal_links_total'] = internal_links
        metrics['avg_words_per_message'] = float(np.mean(avg_words_per_message)) if avg_words_per_message else 0.0
        
        return metrics

    def calculate_recognition_signals_metrics_complete(self, pr_events, issue_events):
        """Calculate ALL recognition signal metrics - COMPLETE IMPLEMENTATION"""
        
        # Import emoji with fallback
        try:
            import emoji
        except ImportError:
            logger.warning("Emoji library not available, skipping emoji analysis")
            emoji = None
        
        # Initialize collectors
        messages_with_thanks = 0
        messages_with_praise = 0
        messages_with_emoji = 0
        messages_with_positive_sentiment = 0
        total_messages = 0
        approval_speeds = []
        total_prs = 0
        merged_prs = 0
        closed_prs = 0
        merged_with_attribution = 0
        total_approvals = 0
        total_rejections = 0
        positive_emoji_count = 0
        negative_emoji_count = 0
        
        # Enhanced patterns
        thanks_patterns = re.compile(r'\b(thank|thanks|thx|appreciate|grateful|kudos)\b', re.IGNORECASE)
        praise_patterns = re.compile(r'\b(great|excellent|awesome|amazing|perfect|brilliant|fantastic|wonderful|good job|well done|nice work|impressive|outstanding)\b', re.IGNORECASE)
        positive_patterns = re.compile(r'\b(love|like|good|nice|helpful|useful|clear|clean|solid)\b', re.IGNORECASE)
        
        positive_emojis = ['😊', '👍', '❤️', '🎉', '✨', '👏', '🚀', '💯', '🙌', '🔥', '⭐', '👌', '😄', '😃', '🙂', '👊']
        negative_emojis = ['😞', '😢', '😭', '💔', '👎', '😠', '😡', '🤔', '😕', '☹️']
        
        # Process PRs
        for pr_json in pr_events:
            pr_data = self.safe_parse_json(pr_json)
            if not pr_data:
                continue
            
            total_prs += 1
            created_at = pd.to_datetime(pr_data.get('created_at') or pr_data.get('createdAt'))
            pr_author = self.extract_author(pr_data)
            
            # PR state analysis
            state = pr_data.get('state', pr_data.get('stateStr', ''))
            merged = pr_data.get('merged', False)
            
            if state == 'MERGED' or merged:
                merged_prs += 1
                
                merged_by = self.extract_author(pr_data.get('mergedBy', {}))
                if merged_by and merged_by != pr_author:
                    merged_with_attribution += 1
            
            if state == 'CLOSED':
                closed_prs += 1
            
            # Review analysis
            reviews = pr_data.get('reviews', [])
            if isinstance(reviews, dict) and 'nodes' in reviews:
                review_nodes = reviews['nodes']
            elif isinstance(reviews, list):
                review_nodes = reviews
            else:
                review_nodes = []
            
            for review in review_nodes:
                if review:
                    review_state = review.get('state', review.get('stateStr', ''))
                    if review_state == 'APPROVED':
                        total_approvals += 1
                        
                        # Calculate approval speed
                        submitted_at = pd.to_datetime(review.get('submittedAt', review.get('submitted_at')))
                        if pd.notna(created_at) and pd.notna(submitted_at):
                            speed_hours = (submitted_at - created_at).total_seconds() / 3600
                            if speed_hours >= 0:
                                approval_speeds.append(speed_hours)
                    
                    elif review_state in ['CHANGES_REQUESTED', 'REQUEST_CHANGES']:
                        total_rejections += 1
            
            # Conversation analysis
            conversations = self.extract_conversations_comprehensive(pr_data)
            for conv in conversations:
                text = conv.get('body', '')
                conv_author = conv.get('author')
                
                if text and conv_author and conv_author != pr_author:
                    total_messages += 1
                    
                    # Pattern matching
                    if thanks_patterns.search(text):
                        messages_with_thanks += 1
                    
                    if praise_patterns.search(text):
                        messages_with_praise += 1
                    
                    if positive_patterns.search(text):
                        messages_with_positive_sentiment += 1
                    
                    # Emoji analysis
                    if emoji:
                        try:
                            emoji_count = emoji.emoji_count(text)
                            if emoji_count > 0:
                                messages_with_emoji += 1
                        except:
                            pass
                    
                    # Manual emoji detection as fallback
                    for pos_emoji in positive_emojis:
                        if pos_emoji in text:
                            messages_with_emoji += 1
                            positive_emoji_count += text.count(pos_emoji)
                            break
                    
                    for neg_emoji in negative_emojis:
                        negative_emoji_count += text.count(neg_emoji)
        
        # Process Issues
        for issue_json in issue_events:
            issue_data = self.safe_parse_json(issue_json)
            if not issue_data:
                continue
            
            issue_author = self.extract_author(issue_data)
            conversations = self.extract_conversations_comprehensive(issue_data)
            
            for conv in conversations:
                text = conv.get('body', '')
                conv_author = conv.get('author')
                
                if text and conv_author and conv_author != issue_author:
                    total_messages += 1
                    
                    if thanks_patterns.search(text):
                        messages_with_thanks += 1
                    
                    if praise_patterns.search(text):
                        messages_with_praise += 1
                    
                    if positive_patterns.search(text):
                        messages_with_positive_sentiment += 1
                    
                    if emoji:
                        try:
                            emoji_count = emoji.emoji_count(text)
                            if emoji_count > 0:
                                messages_with_emoji += 1
                        except:
                            pass
                    
                    for pos_emoji in positive_emojis:
                        if pos_emoji in text:
                            messages_with_emoji += 1
                            positive_emoji_count += text.count(pos_emoji)
                            break
                    
                    for neg_emoji in negative_emojis:
                        negative_emoji_count += text.count(neg_emoji)
        
        # Calculate ALL metrics
        metrics = {}
        
        # Message-based rates
        metrics['total_messages'] = total_messages
        if total_messages > 0:
            metrics['thanks_rate'] = messages_with_thanks / total_messages
            metrics['praise_rate'] = messages_with_praise / total_messages
            metrics['emoji_usage_rate'] = messages_with_emoji / total_messages
            metrics['positive_sentiment_rate'] = messages_with_positive_sentiment / total_messages
        else:
            for key in ['thanks_rate', 'praise_rate', 'emoji_usage_rate', 'positive_sentiment_rate']:
                metrics[key] = 0.0
        
        # PR-based metrics
        metrics['total_prs'] = total_prs
        if total_prs > 0:
            metrics['merge_rate'] = merged_prs / total_prs
            metrics['close_rate'] = closed_prs / total_prs
            metrics['approval_rate'] = total_approvals / total_prs
            metrics['rejection_rate'] = total_rejections / total_prs
        else:
            for key in ['merge_rate', 'close_rate', 'approval_rate', 'rejection_rate']:
                metrics[key] = 0.0
        
        if merged_prs > 0:
            metrics['author_attribution_rate'] = merged_with_attribution / merged_prs
        else:
            metrics['author_attribution_rate'] = 0.0
        
        # Approval speed metrics
        if approval_speeds:
            metrics['approval_speed_mean_hours'] = float(np.mean(approval_speeds))
            metrics['approval_speed_median_hours'] = float(np.median(approval_speeds))
            metrics['approval_speed_std_hours'] = float(np.std(approval_speeds))
            metrics['approval_speed_min_hours'] = float(np.min(approval_speeds))
            metrics['approval_speed_max_hours'] = float(np.max(approval_speeds))
        else:
            for key in ['approval_speed_mean_hours', 'approval_speed_median_hours', 'approval_speed_std_hours',
                       'approval_speed_min_hours', 'approval_speed_max_hours']:
                metrics[key] = np.nan
        
        # Emoji counts
        metrics['positive_emoji_count'] = positive_emoji_count
        metrics['negative_emoji_count'] = negative_emoji_count
        metrics['emoji_sentiment_ratio'] = positive_emoji_count / (negative_emoji_count + 1)  # +1 to avoid division by zero
        
        return metrics

    def calculate_trust_indicators_metrics_complete(self, pr_events, issue_events):
        """Calculate ALL trust indicator metrics - COMPLETE IMPLEMENTATION"""
        
        # Initialize collectors
        metrics = {
            'review_requests_received': 0,
            'at_mentions': 0,
            'issue_assignments': 0,
            'label_additions': 0,
            'milestone_assignments': 0,
            'project_assignments': 0,
            're_engagement_cycles': 0,
            'collaborative_edits': 0,
            'delegated_tasks': 0,
            'trusted_with_sensitive': 0
        }
        
        total_prs = 0
        total_issues = 0
        prs_with_multiple_cycles = 0
        issues_assigned_to_contributor = 0
        prs_with_labels = 0
        prs_with_milestones = 0
        administrative_actions = 0
        
        # Process PRs
        for pr_json in pr_events:
            pr_data = self.safe_parse_json(pr_json)
            if not pr_data:
                continue
            
            total_prs += 1
            pr_author = self.extract_author(pr_data)
            
            # Review requests
            requested_reviewers = pr_data.get('requestedReviewers', pr_data.get('requested_reviewers', []))
            if isinstance(requested_reviewers, dict) and 'nodes' in requested_reviewers:
                metrics['review_requests_received'] += len(requested_reviewers['nodes'])
            elif isinstance(requested_reviewers, list):
                metrics['review_requests_received'] += len(requested_reviewers)
            
            # Labels
            labels = pr_data.get('labels', [])
            if isinstance(labels, dict) and 'nodes' in labels:
                label_nodes = labels['nodes']
            elif isinstance(labels, list):
                label_nodes = labels
            else:
                label_nodes = []
                
            if label_nodes:
                prs_with_labels += 1
                metrics['label_additions'] += len(label_nodes)
                
                # Check for sensitive labels
                sensitive_labels = ['security', 'critical', 'breaking', 'release', 'hotfix']
                for label in label_nodes:
                    label_name = label.get('name', '').lower() if isinstance(label, dict) else str(label).lower()
                    if any(sensitive in label_name for sensitive in sensitive_labels):
                        metrics['trusted_with_sensitive'] += 1
            
            # Milestones
            milestone = pr_data.get('milestone')
            if milestone:
                prs_with_milestones += 1
                metrics['milestone_assignments'] += 1
            
            # Project assignments
            project_cards = pr_data.get('projectCards', pr_data.get('project_cards', []))
            if project_cards:
                metrics['project_assignments'] += len(project_cards) if isinstance(project_cards, list) else 1
            
            # Multiple review cycles
            reviews = pr_data.get('reviews', [])
            if isinstance(reviews, dict) and 'nodes' in reviews:
                review_nodes = reviews['nodes']
            elif isinstance(reviews, list):
                review_nodes = reviews
            else:
                review_nodes = []
            
            if len(review_nodes) > 1:
                prs_with_multiple_cycles += 1
                metrics['re_engagement_cycles'] += len(review_nodes) - 1
            
            # Conversations analysis for mentions
            conversations = self.extract_conversations_comprehensive(pr_data)
            for conv in conversations:
                text = conv.get('body', '')
                if text:
                    mentions = re.findall(r'@[\w-]+', text)
                    metrics['at_mentions'] += len(mentions)
                    
                    # Check for delegated tasks
                    delegation_patterns = ['can you', 'could you', 'please', 'would you mind', 'assign', 'take care']
                    if any(pattern in text.lower() for pattern in delegation_patterns):
                        metrics['delegated_tasks'] += 1
            
            # Administrative actions
            if pr_data.get('closedBy') or pr_data.get('mergedBy'):
                administrative_actions += 1
        
        # Process Issues
        for issue_json in issue_events:
            issue_data = self.safe_parse_json(issue_json)
            if not issue_data:
                continue
            
            total_issues += 1
            issue_author = self.extract_author(issue_data)
            
            # Issue assignments
            assignees = issue_data.get('assignees', [])
            if isinstance(assignees, dict) and 'nodes' in assignees:
                assignee_nodes = assignees['nodes']
            elif isinstance(assignees, list):
                assignee_nodes = assignees
            else:
                assignee_nodes = []
            
            if assignee_nodes:
                issues_assigned_to_contributor += 1
                metrics['issue_assignments'] += len(assignee_nodes)
            
            # Labels
            labels = issue_data.get('labels', [])
            if labels:
                if isinstance(labels, dict) and 'nodes' in labels:
                    label_count = len(labels['nodes'])
                elif isinstance(labels, list):
                    label_count = len(labels)
                else:
                    label_count = 0
                metrics['label_additions'] += label_count
            
            # Milestone
            milestone = issue_data.get('milestone')
            if milestone:
                metrics['milestone_assignments'] += 1
            
            # Conversations analysis
            conversations = self.extract_conversations_comprehensive(issue_data)
            for conv in conversations:
                text = conv.get('body', '')
                if text:
                    mentions = re.findall(r'@[\w-]+', text)
                    metrics['at_mentions'] += len(mentions)
                    
                    delegation_patterns = ['can you', 'could you', 'please', 'would you mind', 'assign', 'take care']
                    if any(pattern in text.lower() for pattern in delegation_patterns):
                        metrics['delegated_tasks'] += 1
        
        # Calculate rates and ratios
        metrics['total_prs'] = total_prs
        metrics['total_issues'] = total_issues
        
        if total_prs > 0:
            metrics['re_engagement_rate'] = prs_with_multiple_cycles / total_prs
            metrics['pr_with_labels_rate'] = prs_with_labels / total_prs
            metrics['pr_with_milestones_rate'] = prs_with_milestones / total_prs
        else:
            metrics['re_engagement_rate'] = 0.0
            metrics['pr_with_labels_rate'] = 0.0
            metrics['pr_with_milestones_rate'] = 0.0
        
        if total_issues > 0:
            metrics['issue_assignment_rate'] = issues_assigned_to_contributor / total_issues
        else:
            metrics['issue_assignment_rate'] = 0.0
        
        metrics['administrative_actions'] = administrative_actions
        metrics['trust_score'] = (metrics['review_requests_received'] + 
                                 metrics['issue_assignments'] + 
                                 metrics['label_additions'] + 
                                 metrics['trusted_with_sensitive'] * 2)
        
        return metrics

    def process_contributor_timeline(self, timeline_file):
        """Process one contributor timeline with ALL metrics"""
        
        try:
            # Load and validate
            timeline_df = pd.read_csv(timeline_file)
            if timeline_df.empty:
                self.debug_stats['empty_timelines'] += 1
                return None
            
            # Filter to pre-core events
            pre_core_df = timeline_df[timeline_df['is_pre_core'] == True].copy()
            if pre_core_df.empty:
                self.debug_stats['empty_timelines'] += 1
                return None
            
            # Extract events
            pr_events = pre_core_df[pre_core_df['event_type'] == 'pull_request']['event_data'].tolist()
            issue_events = pre_core_df[pre_core_df['event_type'] == 'issue']['event_data'].tolist()
            commit_events = pre_core_df[pre_core_df['event_type'] == 'commit']['event_data'].tolist()
            
            # Update debug stats
            self.debug_stats['total_contributors'] += 1
            if pr_events:
                self.debug_stats['contributors_with_prs'] += 1
            if issue_events:
                self.debug_stats['contributors_with_issues'] += 1
            if commit_events:
                self.debug_stats['contributors_with_commits'] += 1
            
            # Calculate ALL metric categories
            metrics = {}
            
            # Category 1: Response Timing (Complete)
            response_metrics = self.calculate_response_timing_metrics_complete(pr_events, issue_events)
            metrics.update(response_metrics)
            
            # Category 2: Engagement Breadth (Complete)
            engagement_metrics = self.calculate_engagement_breadth_metrics_complete(pr_events, issue_events)
            metrics.update(engagement_metrics)
            
            # Category 3: Interaction Patterns (Complete)
            interaction_metrics = self.calculate_interaction_patterns_metrics_complete(pr_events, issue_events)
            metrics.update(interaction_metrics)
            
            # Category 4: Recognition Signals (Complete)
            recognition_metrics = self.calculate_recognition_signals_metrics_complete(pr_events, issue_events)
            metrics.update(recognition_metrics)
            
            # Category 5: Trust Indicators (Complete)
            trust_metrics = self.calculate_trust_indicators_metrics_complete(pr_events, issue_events)
            metrics.update(trust_metrics)
            
            # Add metadata
            project_name = timeline_df['project_name'].iloc[0] if not timeline_df.empty else 'unknown'
            contributor_email = timeline_df['contributor_email'].iloc[0] if 'contributor_email' in timeline_df.columns and not timeline_df.empty else 'unknown'
            
            metrics['contributor_email'] = contributor_email
            metrics['project_name'] = project_name
            
            # Get project type
            contributor_info = self.transitions_df[
                (self.transitions_df['contributor_email'].str.lower() == str(contributor_email).lower()) &
                (self.transitions_df['project_name'] == project_name)
            ]
            
            if not contributor_info.empty:
                metrics['project_type'] = contributor_info['project_type'].iloc[0]
            else:
                metrics['project_type'] = 'unknown'
            
            # Add event counts
            metrics['total_pr_events'] = len(pr_events)
            metrics['total_issue_events'] = len(issue_events)
            metrics['total_commit_events'] = len(commit_events)
            
            self.debug_stats['successful_calculations'] += 1
            return metrics
            
        except Exception as e:
            logger.error(f"Error processing {timeline_file}: {e}")
            self.debug_stats['parsing_errors'] += 1
            return None

    def process_all_contributors(self):
        """Process all contributor timelines"""
        logger.info("Starting COMPLETE treatment metrics calculation...")
        
        timeline_files = list(self.timeline_dir.glob("timeline_*.csv"))
        logger.info(f"Found {len(timeline_files)} timeline files to process")
        
        all_metrics = []
        
        # Process with progress bar
        for timeline_file in tqdm(timeline_files, desc="Processing contributors"):
            try:
                metrics = self.process_contributor_timeline(timeline_file)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Failed to process {timeline_file}: {e}")
                self.debug_stats['parsing_errors'] += 1
                continue
        
        # Create DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save complete dataset
        output_file = self.output_dir / "complete_treatment_metrics_dataset.csv"
        metrics_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved COMPLETE treatment metrics for {len(metrics_df)} contributors")
        logger.info(f"Dataset contains {len(metrics_df.columns)} total columns")
        
        # Print debug statistics
        self.print_debug_statistics()
        
        # Generate summary
        self.generate_summary_statistics(metrics_df)
        
        return metrics_df

    def print_debug_statistics(self):
        """Print comprehensive debug statistics"""
        print("\n" + "="*80)
        print("COMPLETE TREATMENT METRICS - DEBUG STATISTICS")
        print("="*80)
        print(f"Total contributors processed: {self.debug_stats['total_contributors']}")
        print(f"Successful calculations: {self.debug_stats['successful_calculations']}")
        print(f"Contributors with PRs: {self.debug_stats['contributors_with_prs']}")
        print(f"Contributors with issues: {self.debug_stats['contributors_with_issues']}")
        print(f"Contributors with commits: {self.debug_stats['contributors_with_commits']}")
        print(f"Empty timelines: {self.debug_stats['empty_timelines']}")
        print(f"Parsing errors: {self.debug_stats['parsing_errors']}")
        
        if self.debug_stats['total_contributors'] > 0:
            success_rate = self.debug_stats['successful_calculations'] / self.debug_stats['total_contributors'] * 100
            pr_rate = self.debug_stats['contributors_with_prs'] / self.debug_stats['total_contributors'] * 100
            issue_rate = self.debug_stats['contributors_with_issues'] / self.debug_stats['total_contributors'] * 100
            print(f"Success rate: {success_rate:.1f}%")
            print(f"PR participation rate: {pr_rate:.1f}%")
            print(f"Issue participation rate: {issue_rate:.1f}%")

    def generate_summary_statistics(self, metrics_df):
        """Generate comprehensive summary statistics"""
        logger.info("Generating comprehensive summary statistics...")
        
        # Basic summary
        summary_stats = {
            'total_contributors': len(metrics_df),
            'oss_contributors': len(metrics_df[metrics_df['project_type'] == 'OSS']),
            'oss4sg_contributors': len(metrics_df[metrics_df['project_type'] == 'OSS4SG']),
            'total_columns': len(metrics_df.columns),
            'numeric_columns': len(metrics_df.select_dtypes(include=[np.number]).columns),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        # Column categorization
        metric_columns = [col for col in metrics_df.columns if col not in ['contributor_email', 'project_name', 'project_type']]
        
        column_categories = {
            'response_timing': [col for col in metric_columns if any(term in col.lower() for term in ['response', 'first_response', 'timing'])],
            'engagement_breadth': [col for col in metric_columns if any(term in col.lower() for term in ['responder', 'engagement', 'diversity', 'maintainer', 'peer'])],
            'interaction_patterns': [col for col in metric_columns if any(term in col.lower() for term in ['conversation', 'back_forth', 'word', 'message', 'question', 'link'])],
            'recognition_signals': [col for col in metric_columns if any(term in col.lower() for term in ['thanks', 'praise', 'emoji', 'approval', 'merge', 'sentiment'])],
            'trust_indicators': [col for col in metric_columns if any(term in col.lower() for term in ['review_request', 'mention', 'assignment', 'label', 'trust', 'milestone'])],
            'other': []
        }
        
        # Identify uncategorized columns
        categorized_cols = set()
        for category_cols in column_categories.values():
            categorized_cols.update(category_cols)
        
        column_categories['other'] = [col for col in metric_columns if col not in categorized_cols]
        
        summary_stats['column_categories'] = {k: len(v) for k, v in column_categories.items()}
        
        # Save summary
        summary_file = self.output_dir / "complete_dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Save column mapping
        columns_file = self.output_dir / "column_categories_mapping.json"
        with open(columns_file, 'w') as f:
            json.dump(column_categories, f, indent=2)
        
        logger.info(f"Summary statistics saved to {summary_file}")
        
        # Print key findings
        print(f"\n{'='*80}")
        print("COMPLETE DATASET SUMMARY")
        print(f"{'='*80}")
        print(f"Total contributors: {summary_stats['total_contributors']}")
        print(f"OSS contributors: {summary_stats['oss_contributors']}")
        print(f"OSS4SG contributors: {summary_stats['oss4sg_contributors']}")
        print(f"Total columns: {summary_stats['total_columns']}")
        print(f"Numeric columns: {summary_stats['numeric_columns']}")
        print("\nMetric categories:")
        for category, count in summary_stats['column_categories'].items():
            print(f"  {category}: {count} metrics")

def main():
    """Main execution function"""
    print("COMPLETE TREATMENT METRICS DATASET CREATOR")
    print("=" * 80)
    print("This will calculate ALL possible treatment metrics - no shortcuts!")
    print("=" * 80)
    
    try:
        base_path = "/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026"
        
        calculator = CompleteTreatmentMetricsCalculator(base_path)
        metrics_df = calculator.process_all_contributors()
        
        print(f"\n{'='*80}")
        print("COMPLETE TREATMENT METRICS CALCULATION FINISHED")
        print(f"{'='*80}")
        print(f"Generated dataset with {len(metrics_df)} contributors")
        print(f"Dataset contains {len(metrics_df.columns)} metrics")
        print(f"Files saved to: {calculator.output_dir}")
        
        # List generated files
        print(f"\nGenerated files:")
        for file_path in sorted(calculator.output_dir.glob('complete_*')):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   {file_path.name} ({size_mb:.1f} MB)")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
