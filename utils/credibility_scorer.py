import re
from urllib.parse import urlparse
import torch
import numpy as np
from datetime import datetime

class SourceCredibilityScorer:
    def __init__(self, config):
        self.config = config
        self.credible_sources = self.load_credible_sources()
        self.blacklisted_sources = self.load_blacklisted_sources()
        self.suspicious_patterns = self.load_suspicious_patterns()
        
    def load_credible_sources(self):
        """Load list of credible news sources with enhanced database"""
        credible_sources = [
            'reuters.com', 'associatedpress.com', 'apnews.com',
            'bbc.com', 'bbc.co.uk', 'npr.org', 'pbs.org',
            'theguardian.com', 'wsj.com', 'nytimes.com',
            'washingtonpost.com', 'bloomberg.com', 'ft.com',
            'ap.org', 'bbcnews.com', 'cnn.com', 'abcnews.go.com',
            'cbsnews.com', 'nbcnews.com', 'news.sky.com'
        ]
        return set(credible_sources)
    
    def load_blacklisted_sources(self):
        """Load list of known unreliable sources"""
        blacklisted_sources = [
            'infowars.com', 'naturalnews.com', 'beforeitsnews.com',
            'worldtruth.tv', 'veteranstoday.com', 'yournewswire.com',
            'christianfightback.com', 'stateofthenation.co'
        ]
        return set(blacklisted_sources)
    
    def load_suspicious_patterns(self):
        """Load patterns that indicate suspicious sources"""
        return {
            'unusual_domains': ['.tk', '.ml', '.ga', '.cf', '.gq'],  # Free domains
            'suspicious_keywords': ['free', 'win', 'prize', 'click', 'bonus', 'secret'],
            'political_extremes': ['far-right', 'far-left', 'extremist', 'conspiracy']
        }
    
    def extract_domain(self, url):
        """Enhanced domain extraction with validation"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www prefix and port numbers
            if domain.startswith('www.'):
                domain = domain[4:]
            if ':' in domain:
                domain = domain.split(':')[0]
                
            return domain
        except:
            return ""
    
    def score_source(self, source_info):
        """Enhanced source credibility scoring with multiple factors"""
        if isinstance(source_info, str):
            domain = self.extract_domain(source_info)
            author = ''
            history = {}
            content_analysis = {}
        elif isinstance(source_info, dict):
            domain = self.extract_domain(source_info.get('url', ''))
            author = source_info.get('author', '')
            history = source_info.get('history', {})
            content_analysis = source_info.get('content_analysis', {})
        else:
            return 0.5  # Neutral score for unknown sources
        
        score = 0.5  # Start with neutral score
        
        # Domain reputation (strong signal)
        if domain in self.credible_sources:
            score += 0.4
        elif domain in self.blacklisted_sources:
            score -= 0.5
        
        # Domain structure analysis
        domain_score = self.analyze_domain_structure(domain)
        score += domain_score * 0.2
        
        # Author credibility
        author_score = self.assess_author_credibility(author)
        score += author_score * 0.15
        
        # Content history analysis
        history_score = self.analyze_content_history(history)
        score += history_score * 0.1
        
        # Content quality analysis
        content_score = self.analyze_content_quality(content_analysis)
        score += content_score * 0.15
        
        # Temporal factors
        temporal_score = self.analyze_temporal_factors(history)
        score += temporal_score * 0.1
        
        return max(0, min(1, score))
    
    def analyze_domain_structure(self, domain):
        """Analyze domain structure for credibility indicators"""
        if not domain:
            return 0.0
            
        score = 0.5
        
        # Check for suspicious TLDs
        suspicious_tlds = self.suspicious_patterns['unusual_domains']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            score -= 0.4
        
        # Check domain age (placeholder - would use WHOIS in production)
        domain_complexity = domain.count('.')
        if domain_complexity >= 2:
            score += 0.1  # More complex domains often more established
        
        # Check for suspicious keywords in domain
        suspicious_keywords = self.suspicious_patterns['suspicious_keywords']
        if any(keyword in domain for keyword in suspicious_keywords):
            score -= 0.3
        
        # Domain length (very short domains can be suspicious)
        if len(domain) < 5:
            score -= 0.2
        elif len(domain) > 20:
            score += 0.1  # Longer domains often more specific
            
        return max(0, min(1, score))
    
    def assess_author_credibility(self, author):
        """Enhanced author credibility assessment"""
        if not author or len(author.strip()) == 0:
            return 0.3  # Anonymous authors less credible
        
        author_lower = author.lower()
        
        # Verified authors
        verified_indicators = [
            'associated press', 'reuters', 'bbc news', 'ap', 
            'staff reporter', 'correspondent', 'editor'
        ]
        if any(verified in author_lower for verified in verified_indicators):
            return 0.9
        
        # Suspicious author patterns
        suspicious_patterns = [
            r'.*\d+.*',  # Numbers in name
            r'^[A-Z]+\s*$',  # All caps
            r'^[a-z]+\s*$',  # All lowercase
            r'.*@.*',  # Email-like
            r'^user\_\d+',  # User_123 pattern
        ]
        
        for pattern in suspicious_patterns:
            if re.match(pattern, author, re.IGNORECASE):
                return 0.2
        
        # Author name length and structure
        if len(author) < 3:
            return 0.2
        elif ' ' in author and len(author) > 8:  # Has space and reasonable length
            return 0.7
        else:
            return 0.5
    
    def analyze_content_history(self, history):
        """Analyze content history and accuracy"""
        if not history:
            return 0.5
            
        accuracy_rate = history.get('accuracy_rate', 0.5)
        fact_check_rating = history.get('fact_check_rating', 0.5)
        correction_rate = history.get('correction_rate', 0.1)
        
        # Weighted combination
        history_score = (
            accuracy_rate * 0.6 +
            fact_check_rating * 0.3 +
            (1 - correction_rate) * 0.1  # Lower correction rate is better
        )
        
        return max(0, min(1, history_score))
    
    def analyze_content_quality(self, content_analysis):
        """Analyze content quality indicators"""
        if not content_analysis:
            return 0.5
            
        quality_score = 0.5
        
        # Writing quality
        readability = content_analysis.get('readability_score', 0.5)
        quality_score += (readability - 0.5) * 0.3
        
        # Source diversity
        source_diversity = content_analysis.get('source_diversity', 0.5)
        quality_score += (source_diversity - 0.5) * 0.2
        
        # Evidence citation
        citations = content_analysis.get('citation_count', 0)
        quality_score += min(citations / 5, 0.3)  # Up to 0.3 for good citations
        
        return max(0, min(1, quality_score))
    
    def analyze_temporal_factors(self, history):
        """Analyze temporal factors like domain age and update frequency"""
        if not history:
            return 0.5
            
        domain_age = history.get('domain_age_years', 1)
        update_frequency = history.get('update_frequency', 0.5)
        
        # Older domains generally more credible
        age_score = min(domain_age / 10, 1.0)  # Cap at 10 years
        
        # Regular updates indicate active maintenance
        frequency_score = min(update_frequency * 2, 1.0)
        
        return (age_score * 0.7 + frequency_score * 0.3)
    
    def get_credibility_explanation(self, source_info, score):
        """Generate detailed explanation for credibility score"""
        if isinstance(source_info, str):
            domain = self.extract_domain(source_info)
        else:
            domain = self.extract_domain(source_info.get('url', ''))
            
        explanations = []
        
        # Domain-based explanations
        if domain in self.credible_sources:
            explanations.append(f"✓ Domain '{domain}' is a known credible news source.")
        elif domain in self.blacklisted_sources:
            explanations.append(f"✗ Domain '{domain}' has a history of publishing unreliable content.")
        
        # Score-based explanations
        if score >= 0.8:
            explanations.append("✓ Source shows excellent credibility indicators with established reputation.")
        elif score >= 0.6:
            explanations.append("✓ Source shows good credibility indicators.")
        elif score >= 0.4:
            explanations.append("~ Source credibility is neutral or mixed.")
        else:
            explanations.append("✗ Source shows poor credibility indicators.")
        
        # Additional factors
        if isinstance(source_info, dict):
            author = source_info.get('author', '')
            if author:
                author_score = self.assess_author_credibility(author)
                if author_score > 0.7:
                    explanations.append("✓ Author appears credible.")
                elif author_score < 0.3:
                    explanations.append("✗ Author information appears suspicious.")
        
        return explanations
    
    def analyze_url_structure(self, url):
        """Enhanced URL structure analysis for credibility"""
        if not url:
            return 0.5
            
        score = 0.5
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Security protocol
            if url.startswith('https://'):
                score += 0.2
            elif url.startswith('http://'):
                score -= 0.1
            
            # Suspicious patterns
            suspicious_terms = ['free', 'win', 'prize', 'click', 'bonus', 'secret', 'viral']
            if any(term in domain or term in path for term in suspicious_terms):
                score -= 0.3
            
            # News-related terms (positive indicator)
            news_terms = ['news', 'press', 'media', 'journal', 'times', 'post']
            if any(term in domain for term in news_terms):
                score += 0.2
            
            # Domain complexity
            if domain.count('.') >= 2:
                score += 0.1
                
            # Path length (very long paths can be suspicious)
            if len(path) > 50:
                score -= 0.1
                
        except:
            pass
        
        return max(0, min(1, score))
    
    def batch_score_sources(self, sources_list):
        """Score multiple sources efficiently"""
        scores = {}
        explanations = {}
        
        for source in sources_list:
            score = self.score_source(source)
            explanation = self.get_credibility_explanation(source, score)
            
            if isinstance(source, str):
                key = source
            else:
                key = source.get('url', str(source))
                
            scores[key] = score
            explanations[key] = explanation
        
        return scores, explanations