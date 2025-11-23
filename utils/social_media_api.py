import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class SocialMediaCollector:
    def __init__(self, twitter_bearer_token=None):
        self.twitter_bearer_token = twitter_bearer_token
        self.reddit_client_id = None
        self.reddit_client_secret = None
        
    def collect_twitter_data(self, news_url, max_tweets=100):
        """
        Collect Twitter data for a news URL
        Note: This requires Twitter API v2 access
        """
        if not self.twitter_bearer_token:
            print("❌ Twitter bearer token not provided")
            return self._get_mock_twitter_data()
        
        try:
            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}',
                'Content-Type': 'application/json'
            }
            
            # Search for tweets containing the URL
            query = f'url:"{news_url}"'
            url = f'https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={max_tweets}&tweet.fields=author_id,created_at,public_metrics,context_annotations'
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_twitter_data(data)
            else:
                print(f"❌ Twitter API error: {response.status_code}")
                return self._get_mock_twitter_data()
                
        except Exception as e:
            print(f"❌ Error collecting Twitter data: {e}")
            return self._get_mock_twitter_data()
    
    def collect_reddit_data(self, news_url, subreddit='all', limit=100):
        """
        Collect Reddit data for a news URL
        Note: This requires Reddit API credentials
        """
        if not self.reddit_client_id or not self.reddit_client_secret:
            print("❌ Reddit credentials not provided")
            return self._get_mock_reddit_data()
        
        try:
            # Reddit API implementation would go here
            # This is a placeholder for actual implementation
            return self._get_mock_reddit_data()
            
        except Exception as e:
            print(f"❌ Error collecting Reddit data: {e}")
            return self._get_mock_reddit_data()
    
    def build_propagation_graph(self, social_data):
        """
        Build propagation graph from social media data
        """
        # NEW: Import inside function to avoid dependency issues
        try:
            from models.graph_model import SocialGraphBuilder
            graph_builder = SocialGraphBuilder()
        except ImportError:
            print("SocialGraphBuilder not available, using simple graph builder")
            return self._build_simple_propagation_graph(social_data)
        
        if 'twitter_data' in social_data:
            post_data = {
                'user_interactions': self._extract_twitter_interactions(social_data['twitter_data']),
                'timestamps': self._extract_timestamps(social_data['twitter_data'])
            }
        else:
            post_data = None
        
        return graph_builder.build_propagation_graph(post_data)
    
    def _process_twitter_data(self, twitter_data):
        """Process raw Twitter API response"""
        processed_data = {
            'tweet_count': len(twitter_data.get('data', [])),
            'engagement_metrics': {
                'total_likes': 0,
                'total_retweets': 0,
                'total_replies': 0,
                'total_quotes': 0
            },
            'user_data': [],
            'temporal_data': []
        }
        
        for tweet in twitter_data.get('data', []):
            metrics = tweet.get('public_metrics', {})
            processed_data['engagement_metrics']['total_likes'] += metrics.get('like_count', 0)
            processed_data['engagement_metrics']['total_retweets'] += metrics.get('retweet_count', 0)
            processed_data['engagement_metrics']['total_replies'] += metrics.get('reply_count', 0)
            processed_data['engagement_metrics']['total_quotes'] += metrics.get('quote_count', 0)
            
            user_data = {
                'user_id': tweet.get('author_id'),
                'created_at': tweet.get('created_at'),
                'interaction_type': 'tweet'
            }
            processed_data['user_data'].append(user_data)
            
            temporal_data = {
                'timestamp': tweet.get('created_at'),
                'engagement': metrics.get('like_count', 0) + metrics.get('retweet_count', 0)
            }
            processed_data['temporal_data'].append(temporal_data)
        
        return processed_data
    
    def _extract_twitter_interactions(self, twitter_data):
        """Extract user interactions from Twitter data"""
        interactions = []
        
        for user_data in twitter_data.get('user_data', []):
            interactions.append({
                'user_id': user_data.get('user_id', 'unknown'),
                'type': user_data.get('interaction_type', 'tweet'),
                'timestamp': user_data.get('created_at', ''),
                'parent_user_id': None  # Would be filled for retweets/replies
            })
        
        return interactions
    
    def _extract_timestamps(self, social_data):
        """Extract timestamps for temporal analysis"""
        timestamps = []
        
        for temporal_data in social_data.get('temporal_data', []):
            timestamp_str = temporal_data.get('timestamp', '')
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(dt.timestamp())
                except:
                    continue
        
        return timestamps if timestamps else [time.time()]
    
    def _get_mock_twitter_data(self):
        """Generate mock Twitter data for testing"""
        return {
            'tweet_count': 25,
            'engagement_metrics': {
                'total_likes': 150,
                'total_retweets': 45,
                'total_replies': 12,
                'total_quotes': 8
            },
            'user_data': [
                {'user_id': f'user_{i}', 'created_at': '2024-01-01T10:00:00Z', 'interaction_type': 'tweet'} 
                for i in range(25)
            ],
            'temporal_data': [
                {'timestamp': '2024-01-01T10:00:00Z', 'engagement': 10},
                {'timestamp': '2024-01-01T10:05:00Z', 'engagement': 25},
                {'timestamp': '2024-01-01T10:10:00Z', 'engagement': 45}
            ]
        }
    
    def _get_mock_reddit_data(self):
        """Generate mock Reddit data for testing"""
        return {
            'post_count': 15,
            'engagement_metrics': {
                'total_upvotes': 320,
                'total_comments': 45,
                'total_shares': 12
            },
            'user_data': [
                {'user_id': f'redditor_{i}', 'created_at': '2024-01-01T10:00:00Z', 'interaction_type': 'post'} 
                for i in range(15)
            ]
        }
    
    # NEW METHOD: Simple propagation graph builder
    def _build_simple_propagation_graph(self, social_data):
        """Build simple propagation graph when main builder is unavailable"""
        graph_data = {
            'nodes': [],
            'edges': [],
            'propagation_metrics': {
                'virality_score': 0.5,
                'spread_velocity': 0.3,
                'engagement_density': 0.4
            }
        }
        
        # Add nodes from user data
        if 'twitter_data' in social_data:
            for user in social_data['twitter_data'].get('user_data', []):
                graph_data['nodes'].append({
                    'id': user['user_id'],
                    'type': 'user',
                    'engagement': 1
                })
        
        # Add simple edges (placeholder)
        if len(graph_data['nodes']) > 1:
            for i in range(len(graph_data['nodes']) - 1):
                graph_data['edges'].append({
                    'source': graph_data['nodes'][i]['id'],
                    'target': graph_data['nodes'][i + 1]['id'],
                    'weight': 0.5
                })
        
        return graph_data

class BehavioralAnalyzer:
    """Analyze user behavior patterns"""
    
    def __init__(self):
        self.suspicious_patterns = {
            'bot_like_behavior': ['high_frequency', 'low_variance', 'repetitive_content'],
            'coordination': ['synchronized_posting', 'similar_content', 'network_clusters'],
            'astroturfing': ['fake_engagement', 'inauthentic_profiles', 'sudden_spikes']
        }
    
    def analyze_user_behavior(self, user_data):
        """Analyze user behavior patterns for authenticity"""
        features = []
        
        # Account age analysis (if available)
        account_age = self._analyze_account_age(user_data.get('account_created'))
        features.append(account_age)
        
        # Posting frequency analysis
        posting_freq = self._analyze_posting_frequency(user_data.get('post_history', []))
        features.append(posting_freq)
        
        # Engagement patterns
        engagement_score = self._analyze_engagement_patterns(user_data.get('engagement_metrics', {}))
        features.append(engagement_score)
        
        # Network analysis
        network_score = self._analyze_network_patterns(
            user_data.get('followers', 0), 
            user_data.get('friends', 0)
        )
        features.append(network_score)
        
        return features
    
    def detect_coordinated_behavior(self, posts):
        """Detect coordinated inauthentic behavior"""
        coordination_signals = []
        
        # Check for synchronized posting times
        if len(posts) > 5:
            time_differences = self._analyze_timing_patterns(posts)
            if time_differences < 300:  # Posts within 5 minutes
                coordination_signals.append('synchronized_timing')
        
        # Check for similar content patterns
        if self._detect_content_similarity(posts):
            coordination_signals.append('similar_content')
        
        # Check for network clusters
        if self._detect_network_clusters(posts):
            coordination_signals.append('network_clustering')
        
        return coordination_signals
    
    def _analyze_account_age(self, account_created):
        """Analyze account age for authenticity"""
        if not account_created:
            return 0.5  # Neutral if unknown
        
        try:
            # Convert to datetime and calculate age
            if isinstance(account_created, str):
                created_dt = datetime.fromisoformat(account_created.replace('Z', '+00:00'))
            else:
                created_dt = account_created
            
            account_age_days = (datetime.now() - created_dt).days
            
            # Normalize to 0-1 scale (older accounts are more trustworthy)
            if account_age_days > 365:  # More than 1 year
                return 1.0
            elif account_age_days > 30:  # More than 1 month
                return 0.7
            else:  # New account
                return 0.3
                
        except:
            return 0.5
    
    def _analyze_posting_frequency(self, post_history):
        """Analyze posting frequency patterns"""
        if len(post_history) < 5:
            return 0.5  # Not enough data
        
        # Calculate posting frequency variance
        frequencies = []
        for i in range(1, len(post_history)):
            time_diff = post_history[i] - post_history[i-1]
            frequencies.append(time_diff)
        
        if frequencies:
            variance = np.var(frequencies)
            # Low variance might indicate bot-like behavior
            return min(variance / 3600, 1.0)  # Normalize by hour
        else:
            return 0.5
    
    def _analyze_engagement_patterns(self, engagement_metrics):
        """Analyze engagement patterns for authenticity"""
        total_engagement = sum(engagement_metrics.values()) if engagement_metrics else 0
        
        if total_engagement == 0:
            return 0.5
        
        # High engagement with low diversity might indicate artificial boosting
        engagement_diversity = len(engagement_metrics) / total_engagement if total_engagement > 0 else 0
        return min(engagement_diversity * 10, 1.0)
    
    def _analyze_network_patterns(self, followers, friends):
        """Analyze follower/friend patterns"""
        if followers == 0:
            return 0.3  # Suspicious if no followers
        
        ratio = friends / followers if followers > 0 else 1.0
        
        # Normal accounts typically have more followers than friends
        if ratio < 1.0:
            return 0.8  # Healthy ratio
        elif ratio > 10.0:
            return 0.2  # Suspicious ratio
        else:
            return 0.5  # Neutral
    
    def _analyze_timing_patterns(self, posts):
        """Analyze timing patterns between posts"""
        if len(posts) < 2:
            return float('inf')
        
        timestamps = [post.get('timestamp', 0) for post in posts if 'timestamp' in post]
        if len(timestamps) < 2:
            return float('inf')
        
        timestamps.sort()
        differences = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        return np.mean(differences) if differences else float('inf')
    
    def _detect_content_similarity(self, posts):
        """Detect if posts have very similar content"""
        if len(posts) < 3:
            return False
        
        # Simple content similarity check
        contents = [post.get('content', '') for post in posts]
        unique_contents = set(contents)
        
        # If many posts have identical content
        return len(unique_contents) < len(contents) * 0.7
    
    def _detect_network_clusters(self, posts):
        """Detect if posts come from tightly connected users"""
        if len(posts) < 3:
            return False
        
        # Simple network clustering detection
        user_ids = [post.get('user_id') for post in posts]
        unique_users = set(user_ids)
        
        # If few users are responsible for many posts
        return len(unique_users) < len(posts) * 0.5
    
    # NEW METHOD: Calculate overall behavior score
    def calculate_behavior_score(self, user_data, posts):
        """Calculate overall behavior authenticity score"""
        features = self.analyze_user_behavior(user_data)
        coordination_signals = self.detect_coordinated_behavior(posts)
        
        # Average the feature scores
        behavior_score = np.mean(features) if features else 0.5
        
        # Penalize for coordination signals
        coordination_penalty = len(coordination_signals) * 0.1
        behavior_score = max(0, behavior_score - coordination_penalty)
        
        return behavior_score