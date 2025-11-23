import numpy as np
from datetime import datetime, timedelta
import statistics
from collections import Counter
import torch

class SocialMediaTrendAnalyzer:
    def __init__(self):
        self.bot_patterns = self.load_bot_patterns()
        self.coordination_indicators = self.load_coordination_indicators()
        
    def load_bot_patterns(self):
        """Define patterns associated with bot behavior"""
        return {
            'high_frequency_posting': 0.8,
            'low_diversity_content': 0.7,
            'coordinated_timing': 0.9,
            'suspicious_engagement_ratios': 0.6,
            'repetitive_behavior': 0.8,
            'network_clustering': 0.7
        }
    
    def load_coordination_indicators(self):
        """Define coordination detection patterns"""
        return {
            'temporal_synchronization': 0.8,
            'content_similarity': 0.7,
            'network_structure': 0.6,
            'behavioral_patterns': 0.7
        }
    
    def analyze_engagement(self, engagement_data):
        """Enhanced social media engagement pattern analysis"""
        analysis = {
            'bot_score': 0,
            'virality_score': 0,
            'coordination_score': 0,
            'authenticity_score': 0,
            'network_health': 0,
            'explanations': [],
            'risk_factors': []
        }
        
        # Enhanced bot detection
        analysis['bot_score'] = self.detect_bot_activity(engagement_data)
        if analysis['bot_score'] > 0.7:
            analysis['explanations'].append("High likelihood of automated/bot activity detected.")
            analysis['risk_factors'].append('BOT_NETWORK')
        
        # Virality analysis with enhanced metrics
        analysis['virality_score'] = self.analyze_virality(engagement_data)
        if analysis['virality_score'] > 0.8:
            analysis['explanations'].append("Content shows viral spreading patterns.")
        
        # Advanced coordination detection
        analysis['coordination_score'] = self.detect_coordinated_behavior(engagement_data)
        if analysis['coordination_score'] > 0.6:
            analysis['explanations'].append("Possible coordinated amplification campaign detected.")
            analysis['risk_factors'].append('COORDINATED_BEHAVIOR')
        
        # Network health analysis
        analysis['network_health'] = self.assess_network_health(engagement_data)
        
        # Overall authenticity
        analysis['authenticity_score'] = self.calculate_authenticity(analysis)
        
        # Add confidence metrics
        analysis['confidence_metrics'] = self.calculate_confidence_metrics(engagement_data)
        
        return analysis
    
    def detect_bot_activity(self, engagement_data):
        """Enhanced bot detection with multiple indicators"""
        bot_score = 0
        indicators_triggered = 0
        
        # Check engagement ratios
        retweets = engagement_data.get('retweets', 0)
        likes = engagement_data.get('likes', 0)
        replies = engagement_data.get('replies', 0)
        quotes = engagement_data.get('quotes', 0)
        
        total_engagements = retweets + likes + replies + quotes
        
        if total_engagements > 0:
            # High retweet-to-like ratio can indicate bots
            rt_like_ratio = retweets / max(likes, 1)
            if rt_like_ratio > 5:
                bot_score += 0.3
                indicators_triggered += 1
            
            # Low reply-to-retweet ratio
            reply_rt_ratio = replies / max(retweets, 1)
            if reply_rt_ratio < 0.1:
                bot_score += 0.2
                indicators_triggered += 1
            
            # Quote behavior analysis
            quote_ratio = quotes / max(retweets, 1)
            if quote_ratio > 2.0:  # Unusually high quoting
                bot_score += 0.1
        
        # Enhanced posting frequency analysis
        post_frequency = engagement_data.get('posts_per_hour', 0)
        if post_frequency > 15:  # More than 15 posts per hour
            bot_score += 0.3
            indicators_triggered += 1
        elif post_frequency > 8:  # 8-15 posts per hour
            bot_score += 0.15
        
        # Content diversity analysis
        content_diversity = engagement_data.get('content_diversity', 1.0)
        if content_diversity < 0.3:  # Low diversity
            bot_score += 0.2
            indicators_triggered += 1
        
        # Temporal pattern analysis
        temporal_consistency = engagement_data.get('temporal_consistency', 0.5)
        if temporal_consistency > 0.8:  # Highly consistent timing
            bot_score += 0.2
        
        # Normalize by number of indicators
        if indicators_triggered > 0:
            bot_score = min(1.0, bot_score * (1 + (indicators_triggered - 1) * 0.2))
        
        return bot_score
    
    def analyze_virality(self, engagement_data):
        """Enhanced virality analysis with network effects"""
        shares = engagement_data.get('shares', 0)
        views = engagement_data.get('views', 0)
        unique_users = engagement_data.get('unique_users', 0)
        time_hours = engagement_data.get('time_since_post_hours', 1)
        
        if time_hours == 0:
            time_hours = 1
        
        # Calculate shares per hour with decay
        shares_per_hour = shares / time_hours
        
        # Engagement rate with user diversity
        engagement_rate = shares / max(views, 1)
        user_diversity = unique_users / max(shares, 1)
        
        # Network effect multiplier
        network_effect = engagement_data.get('network_effect', 1.0)
        
        # Enhanced virality score
        base_virality = min(1.0, (shares_per_hour / 1000) + (engagement_rate * 10))
        diversity_bonus = user_diversity * 0.3
        network_bonus = (network_effect - 1.0) * 0.2
        
        virality_score = base_virality + diversity_bonus + network_bonus
        
        return min(1.0, virality_score)
    
    def detect_coordinated_behavior(self, engagement_data):
        """Enhanced coordinated behavior detection"""
        coordination_score = 0
        indicators = 0
        
        # Enhanced temporal coordination
        timestamps = engagement_data.get('engagement_timestamps', [])
        if len(timestamps) > 10:
            time_diffs = self.calculate_time_differences(timestamps)
            if time_diffs:
                cv = statistics.stdev(time_diffs) / statistics.mean(time_diffs) if statistics.mean(time_diffs) > 0 else 0
                
                # Low coefficient of variation indicates regular timing
                if cv < 0.2:
                    coordination_score += 0.4
                    indicators += 1
                elif cv < 0.3:
                    coordination_score += 0.2
        
        # User diversity analysis
        unique_users = engagement_data.get('unique_users', 0)
        total_engagements = engagement_data.get('total_engagements', 0)
        
        if total_engagements > 0:
            user_diversity = unique_users / total_engagements
            if user_diversity < 0.1:  # Very low diversity
                coordination_score += 0.3
                indicators += 1
            elif user_diversity < 0.2:  # Low diversity
                coordination_score += 0.15
        
        # Content similarity analysis
        content_similarity = engagement_data.get('content_similarity', 0)
        if content_similarity > 0.8:
            coordination_score += 0.3
            indicators += 1
        
        # Network clustering
        clustering_coefficient = engagement_data.get('clustering_coefficient', 0)
        if clustering_coefficient > 0.7:
            coordination_score += 0.2
            indicators += 1
        
        # Boost score if multiple indicators present
        if indicators >= 2:
            coordination_score *= 1.2
        
        return min(1.0, coordination_score)
    
    def assess_network_health(self, engagement_data):
        """Assess overall network health and authenticity"""
        health_score = 0.5
        
        # User authenticity
        verified_ratio = engagement_data.get('verified_ratio', 0)
        health_score += verified_ratio * 0.2
        
        # Engagement quality
        reply_ratio = engagement_data.get('reply_ratio', 0)
        health_score += min(reply_ratio * 0.3, 0.3)  # Higher reply ratio is good
        
        # Content diversity
        content_diversity = engagement_data.get('content_diversity', 1.0)
        health_score += (content_diversity - 0.5) * 0.2
        
        # Temporal distribution
        temporal_spread = engagement_data.get('temporal_spread', 0.5)
        health_score += (temporal_spread - 0.5) * 0.3
        
        return max(0, min(1, health_score))
    
    def calculate_time_differences(self, timestamps):
        """Calculate time differences between engagements with enhanced processing"""
        if len(timestamps) < 2:
            return []
        
        # Convert to datetime objects if they're strings
        try:
            if isinstance(timestamps[0], str):
                timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        except:
            return []
        
        sorted_times = sorted(timestamps)
        diffs = []
        for i in range(1, len(sorted_times)):
            diff = (sorted_times[i] - sorted_times[i-1]).total_seconds()
            diffs.append(diff)
        
        return diffs
    
    def calculate_authenticity(self, analysis):
        """Calculate overall authenticity score"""
        bot_score = analysis['bot_score']
        coordination_score = analysis['coordination_score']
        network_health = analysis['network_health']
        
        # Weighted combination
        authenticity = (
            (1.0 - bot_score) * 0.4 +
            (1.0 - coordination_score) * 0.3 +
            network_health * 0.3
        )
        
        return max(0.0, authenticity)
    
    def calculate_confidence_metrics(self, engagement_data):
        """Calculate confidence metrics for the analysis"""
        data_completeness = self.assess_data_completeness(engagement_data)
        temporal_coverage = self.assess_temporal_coverage(engagement_data)
        user_coverage = self.assess_user_coverage(engagement_data)
        
        overall_confidence = (data_completeness + temporal_coverage + user_coverage) / 3
        
        return {
            'data_completeness': data_completeness,
            'temporal_coverage': temporal_coverage,
            'user_coverage': user_coverage,
            'overall_confidence': overall_confidence
        }
    
    def assess_data_completeness(self, engagement_data):
        """Assess completeness of engagement data"""
        required_fields = ['shares', 'likes', 'retweets', 'replies', 'unique_users']
        present_fields = sum(1 for field in required_fields if field in engagement_data and engagement_data[field] is not None)
        
        return present_fields / len(required_fields)
    
    def assess_temporal_coverage(self, engagement_data):
        """Assess temporal coverage of the data"""
        timestamps = engagement_data.get('engagement_timestamps', [])
        if len(timestamps) < 5:
            return 0.3
        elif len(timestamps) < 20:
            return 0.6
        else:
            return 0.9
    
    def assess_user_coverage(self, engagement_data):
        """Assess user coverage in the data"""
        unique_users = engagement_data.get('unique_users', 0)
        total_engagements = engagement_data.get('total_engagements', 0)
        
        if total_engagements == 0:
            return 0.1
        
        user_ratio = unique_users / total_engagements
        
        if user_ratio > 0.5:
            return 0.9
        elif user_ratio > 0.2:
            return 0.6
        else:
            return 0.3
    
    def generate_risk_report(self, analysis):
        """Generate comprehensive risk assessment report"""
        risk_level = "LOW"
        if analysis['bot_score'] > 0.7 or analysis['coordination_score'] > 0.7:
            risk_level = "HIGH"
        elif analysis['bot_score'] > 0.5 or analysis['coordination_score'] > 0.5:
            risk_level = "MEDIUM"
        
        recommendations = []
        if analysis['bot_score'] > 0.6:
            recommendations.append("Investigate potential bot networks")
        if analysis['coordination_score'] > 0.6:
            recommendations.append("Monitor for coordinated inauthentic behavior")
        if analysis['authenticity_score'] < 0.4:
            recommendations.append("Verify content authenticity through multiple sources")
        
        return {
            'risk_level': risk_level,
            'authenticity_score': analysis['authenticity_score'],
            'key_risks': analysis.get('risk_factors', []),
            'recommendations': recommendations,
            'confidence': analysis.get('confidence_metrics', {}).get('overall_confidence', 0.5)
        }