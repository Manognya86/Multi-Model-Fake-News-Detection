import re
import nltk
from textstat import flesch_reading_ease, smog_index, flesch_kincaid_grade
from textblob import TextBlob
import numpy as np
import torch
from urllib.parse import urlparse
import emoji
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    def __init__(self):
        self.sia = self._initialize_sentiment_analyzer()
        self._setup_nltk()
        
        # Enhanced word lists for fake news detection
        self.sensational_words = [
            'shocking', 'amazing', 'unbelievable', 'miracle', 'secret', 'exposed',
            'cover-up', 'breaking', 'urgent', 'warning', 'alarming', 'outrageous',
            'massive', 'huge', 'epic', 'mind-blowing', 'explosive', 'bombshell',
            'you won\'t believe', 'what happened next', 'this will blow your mind'
        ]
        
        self.emotional_words = [
            'outrageous', 'disgusting', 'horrible', 'terrible', 'fantastic', 'wonderful',
            'angry', 'furious', 'devastating', 'heartbreaking', 'joyous', 'ecstatic',
            'disgraceful', 'appalling', 'shameful', 'miraculous', 'spectacular'
        ]
        
        self.certainty_words = [
            'certainly', 'definitely', 'undoubtedly', 'clearly', 'obviously',
            'absolutely', 'unquestionably', 'indisputably', 'guaranteed', 'proven'
        ]
        
        self.manipulation_phrases = [
            'they don\'t want you to know', 'the truth they\'re hiding',
            'what the media won\'t tell you', 'this will be deleted soon',
            'share before it\'s removed', 'viral for a reason',
            'mainstream media is lying', 'the establishment doesn\'t want you to know'
        ]
        
        self.conspiracy_terms = [
            'deep state', 'false flag', 'cover-up', 'conspiracy', 'they are lying',
            'wake up', 'sheeple', 'the truth is being suppressed'
        ]
        
        self.clickbait_patterns = [
            r'you won\'t believe what happened next',
            r'shocked when they saw this',
            r'what happened next will blow your mind',
            r'this changes everything',
            r'doctors are stunned'
        ]
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analyzer with error handling"""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except:
            print("⚠️ VADER sentiment analyzer not available")
            return None
    
    def _setup_nltk(self):
        """Setup NLTK resources with error handling"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt')
            except:
                print("⚠️ NLTK punkt not available")
        
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            try:
                nltk.download('vader_lexicon')
            except:
                print("⚠️ NLTK vader_lexicon not available")
    
    def extract_advanced_features(self, text, metadata=None):
        """Legacy method for backward compatibility"""
        return self.extract_enhanced_features(text, metadata, include_context=False)
    
    def extract_enhanced_features(self, text, metadata=None, include_context=True):
        """Extract comprehensive features for fake news detection with enhanced patterns"""
        features = {}
        
        # Basic text statistics
        features.update(self._extract_basic_features(text))
        
        # Readability and complexity
        features.update(self._extract_readability_features(text))
        
        # Sentiment and emotion analysis
        features.update(self._extract_sentiment_features(text))
        
        # Psychological and manipulation indicators
        features.update(self._extract_psychological_features(text))
        
        # Linguistic patterns
        features.update(self._extract_linguistic_features(text))
        
        # Fake news specific patterns
        features.update(self._extract_fake_news_patterns(text))
        
        # Metadata features if available
        if metadata:
            features.update(self._extract_enhanced_metadata_features(metadata))
        
        # Contextual features
        if include_context:
            features.update(self._extract_contextual_features(text, metadata))
        
        return features
    
    def _extract_basic_features(self, text):
        """Extract basic text statistics with enhanced metrics"""
        if not text or len(text.strip()) == 0:
            return {
                'text_length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'unique_word_ratio': 0,
                'paragraph_count': 0
            }
        
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'paragraph_count': len(paragraphs)
        }
    
    def _extract_readability_features(self, text):
        """Extract readability and complexity metrics with error handling"""
        try:
            flesch_ease = flesch_reading_ease(text)
            smog_idx = smog_index(text)
            fk_grade = flesch_kincaid_grade(text)
            
            return {
                'flesch_reading_ease': flesch_ease,
                'smog_index': smog_idx,
                'flesch_kincaid_grade': fk_grade,
                'complex_word_ratio': self._calculate_complex_word_ratio(text),
                'syllable_count': self._count_syllables(text),
                'readability_score': max(0, min(1, (100 - abs(flesch_ease - 60)) / 100))
            }
        except:
            return {k: 0 for k in ['flesch_reading_ease', 'smog_index', 
                                  'flesch_kincaid_grade', 'complex_word_ratio', 
                                  'syllable_count', 'readability_score']}
    
    def _extract_sentiment_features(self, text):
        """Extract comprehensive sentiment features"""
        features = {}
        
        if not text:
            return {
                'subjectivity': 0.5,
                'polarity': 0.0,
                'vader_neg': 0.0,
                'vader_neu': 1.0,
                'vader_pos': 0.0,
                'vader_compound': 0.0,
                'emotional_intensity': 0.0,
                'sentiment_variance': 0.0
            }
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            features['subjectivity'] = blob.sentiment.subjectivity
            features['polarity'] = blob.sentiment.polarity
        except:
            features['subjectivity'] = 0.5
            features['polarity'] = 0.0
        
        # VADER sentiment if available
        if self.sia:
            try:
                vader_scores = self.sia.polarity_scores(text)
                features.update({
                    'vader_neg': vader_scores['neg'],
                    'vader_neu': vader_scores['neu'],
                    'vader_pos': vader_scores['pos'],
                    'vader_compound': vader_scores['compound']
                })
            except:
                features.update({
                    'vader_neg': 0.0, 'vader_neu': 1.0, 
                    'vader_pos': 0.0, 'vader_compound': 0.0
                })
        else:
            features.update({
                'vader_neg': 0.0, 'vader_neu': 1.0, 
                'vader_pos': 0.0, 'vader_compound': 0.0
            })
        
        # Emotional intensity
        features['emotional_intensity'] = self._calculate_emotional_intensity(text)
        features['sentiment_variance'] = self._calculate_sentiment_variance(text)
        
        return features
    
    def _extract_psychological_features(self, text):
        """Extract psychological manipulation indicators"""
        if not text:
            return {
                'sensationalism_score': 0.0,
                'certainty_level': 0.0,
                'manipulation_indicators': 0.0,
                'urgency_indicators': 0.0,
                'authority_appeals': 0.0
            }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        return {
            'sensationalism_score': sum(1 for word in words if word in self.sensational_words) / len(words) if words else 0.0,
            'certainty_level': sum(1 for word in words if word in self.certainty_words) / len(words) if words else 0.0,
            'manipulation_indicators': sum(1 for phrase in self.manipulation_phrases if phrase in text_lower),
            'urgency_indicators': text_lower.count('urgent') + text_lower.count('immediately') + text_lower.count('now') + text_lower.count('quick'),
            'authority_appeals': text_lower.count('experts say') + text_lower.count('studies show') + text_lower.count('scientists prove') + text_lower.count('doctors recommend')
        }
    
    def _extract_linguistic_features(self, text):
        """Extract advanced linguistic patterns"""
        if not text:
            return {
                'passive_voice_ratio': 0.0,
                'modal_verb_ratio': 0.0,
                'punctuation_intensity': 0.0,
                'capitalization_ratio': 0.0,
                'emoji_count': 0.0,
                'url_count': 0.0,
                'mention_count': 0.0,
                'hashtag_count': 0.0
            }
        
        return {
            'passive_voice_ratio': self._detect_passive_voice(text),
            'modal_verb_ratio': self._count_modal_verbs(text),
            'punctuation_intensity': (text.count('!') + text.count('?')) / len(text) if text else 0.0,
            'capitalization_ratio': sum(1 for char in text if char.isupper()) / len(text) if text else 0.0,
            'emoji_count': emoji.emoji_count(text),
            'url_count': len(re.findall(r'http\S+', text)),
            'mention_count': len(re.findall(r'@\w+', text)),
            'hashtag_count': len(re.findall(r'#\w+', text))
        }
    
    def _extract_fake_news_patterns(self, text):
        """Extract patterns specifically indicative of fake news"""
        if not text:
            return {
                'clickbait_score': 0.0,
                'conspiracy_indicators': 0.0,
                'emotional_manipulation': 0.0,
                'polarization_score': 0.0,
                'sensational_headline': 0.0
            }
        
        text_lower = text.lower()
        
        # Clickbait detection
        clickbait_matches = sum(1 for pattern in self.clickbait_patterns if re.search(pattern, text_lower))
        clickbait_score = min(1.0, clickbait_matches / 2.0)
        
        # Conspiracy indicators
        conspiracy_count = sum(1 for term in self.conspiracy_terms if term in text_lower)
        conspiracy_score = min(1.0, conspiracy_count / 3.0)
        
        # Emotional manipulation
        emotional_words_count = sum(1 for word in self.emotional_words if word in text_lower)
        emotional_score = min(1.0, emotional_words_count / 5.0)
        
        # Polarization (us vs them language)
        polarization_terms = ['they', 'them', 'us', 'we', 'our', 'their']
        polarization_count = sum(1 for term in polarization_terms if term in text_lower.split())
        polarization_score = min(1.0, polarization_count / 10.0)
        
        # Sensational headline indicators (multiple exclamation/question marks)
        headline_intensity = (text.count('!') + text.count('?')) / max(1, len(text.split()))
        
        return {
            'clickbait_score': clickbait_score,
            'conspiracy_indicators': conspiracy_score,
            'emotional_manipulation': emotional_score,
            'polarization_score': polarization_score,
            'sensational_headline': min(1.0, headline_intensity * 10)
        }
    
    def _extract_enhanced_metadata_features(self, metadata):
        """Extract enhanced metadata features for fake news detection"""
        features = {}
        
        if isinstance(metadata, dict):
            # Source analysis
            source = metadata.get('source', '')
            features.update(self._analyze_source_credibility(source))
            
            # Temporal features
            timestamp = metadata.get('timestamp', '')
            features.update(self._analyze_temporal_features(timestamp))
            
            # Author analysis
            author = metadata.get('author', '')
            features['author_credibility'] = self._assess_author_credibility(author)
            
            # Engagement metrics
            features['engagement_velocity'] = min(1.0, metadata.get('engagement_velocity', 0) / 1000)
            features['user_diversity'] = metadata.get('user_diversity', 0.5)
            
            # Propagation patterns
            features['virality_score'] = metadata.get('virality_score', 0.5)
            features['bot_activity'] = metadata.get('bot_activity_score', 0.0)
        
        return features
    
    def _extract_contextual_features(self, text, metadata):
        """Extract contextual and network features"""
        return {
            'context_consistency': self._assess_context_consistency(text, metadata),
            'temporal_relevance': self._assess_temporal_relevance(metadata),
            'source_alignment': self._assess_source_alignment(text, metadata),
            'social_cohesion': self._assess_social_cohesion(metadata)
        }
    
    # Enhanced helper methods
    def _calculate_complex_word_ratio(self, text, syllable_threshold=3):
        """Calculate ratio of complex words"""
        words = text.split()
        if not words:
            return 0.0
        
        complex_count = sum(1 for word in words if self._count_syllables(word) >= syllable_threshold)
        return complex_count / len(words)
    
    def _count_syllables(self, word):
        """Enhanced syllable counting"""
        word = word.lower()
        if not word:
            return 0
        
        count = 0
        vowels = "aeiouy"
        
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        return max(1, count)
    
    def _calculate_emotional_intensity(self, text):
        """Calculate emotional intensity score"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        emotional_count = sum(1 for word in words if word in self.emotional_words)
        return min(1.0, emotional_count / 5.0)
    
    def _calculate_sentiment_variance(self, text):
        """Calculate sentiment variance across sentences"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 0.0
        
        sentiments = []
        for sent in sentences:
            try:
                blob = TextBlob(sent)
                sentiments.append(blob.sentiment.polarity)
            except:
                sentiments.append(0.0)
        
        return np.var(sentiments) if sentiments else 0.0
    
    def _detect_passive_voice(self, text):
        """Enhanced passive voice detection"""
        passive_indicators = [
            'was', 'were', 'been', 'being', 'is', 'are', 'am',
            'get', 'got', 'has been', 'have been', 'had been'
        ]
        words = text.lower().split()
        if not words:
            return 0.0
        
        return sum(1 for word in words if word in passive_indicators) / len(words)
    
    def _count_modal_verbs(self, text):
        """Count modal verbs indicating uncertainty"""
        modal_verbs = ['can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must']
        words = text.lower().split()
        if not words:
            return 0.0
        
        return sum(1 for word in words if word in modal_verbs) / len(words)
    
    def _analyze_source_credibility(self, source):
        """Analyze source credibility features"""
        features = {}
        
        try:
            parsed = urlparse(source)
            domain = parsed.netloc.lower()
            
            features['source_length'] = min(1.0, len(source) / 100)
            features['has_https'] = 1 if source.startswith('https') else 0
            features['domain_complexity'] = min(1.0, domain.count('.') / 3)
            features['is_common_domain'] = 1 if any(d in domain for d in ['.com', '.org', '.net', '.edu', '.gov']) else 0
            features['has_subdomain'] = 1 if domain.count('.') > 1 else 0
            
            # Domain trust indicator
            features['domain_trust_indicator'] = self._estimate_domain_trust(domain)
            
        except:
            features.update({k: 0 for k in ['source_length', 'has_https', 'domain_complexity', 
                                          'is_common_domain', 'has_subdomain', 'domain_trust_indicator']})
        
        return features
    
    def _estimate_domain_trust(self, domain):
        """Estimate domain trustworthiness (enhanced)"""
        trusted_domains = ['reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 
                          'washingtonpost.com', 'theguardian.com', 'wsj.com']
        suspicious_domains = ['blogspot.com', 'wordpress.com', 'weebly.com', 'tumblr.com',
                             'blogger.com', 'medium.com']
        
        if any(td in domain for td in trusted_domains):
            return 1.0
        elif any(sd in domain for sd in suspicious_domains):
            return 0.2
        else:
            return 0.5
    
    def _analyze_temporal_features(self, timestamp):
        """Analyze temporal features for fake news detection"""
        features = {}
        
        try:
            if timestamp:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                
                current_time = datetime.now()
                time_diff = current_time - dt
                
                # Recent content might be less verified
                features['is_recent'] = 1 if time_diff.total_seconds() < 3600 else 0  # 1 hour
                features['hours_since_post'] = min(1.0, time_diff.total_seconds() / 86400)  # Normalize to 1 day
                
                # Time of day analysis (placeholder for behavioral patterns)
                features['is_off_hours'] = 1 if dt.hour < 6 or dt.hour > 22 else 0
            else:
                features['is_recent'] = 0
                features['hours_since_post'] = 0
                features['is_off_hours'] = 0
                
        except:
            features['is_recent'] = 0
            features['hours_since_post'] = 0
            features['is_off_hours'] = 0
            
        return features
    
    def _assess_author_credibility(self, author):
        """Enhanced author credibility assessment"""
        if not author or len(author.strip()) == 0:
            return 0.3  # Anonymous authors less credible
        
        author_lower = author.lower()
        
        # Verified authors and reputable sources
        verified_indicators = [
            'associated press', 'reuters', 'bbc news', 'ap', 
            'staff reporter', 'correspondent', 'editor', 'journalist'
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
            r'^anonymous',  # Explicitly anonymous
        ]
        
        for pattern in suspicious_patterns:
            if re.match(pattern, author, re.IGNORECASE):
                return 0.2
        
        # Author name structure analysis
        if len(author) < 3:
            return 0.2
        elif ' ' in author and len(author) > 8:  # Has space and reasonable length
            return 0.7
        else:
            return 0.5
    
    def _assess_context_consistency(self, text, metadata):
        """Assess context consistency (placeholder implementation)"""
        # In practice, this would compare text content with metadata context
        return 0.7
    
    def _assess_temporal_relevance(self, metadata):
        """Assess temporal relevance (placeholder implementation)"""
        # In practice, this would check if the content is timely/relevant
        return 0.8
    
    def _assess_source_alignment(self, text, metadata):
        """Assess source alignment (placeholder implementation)"""
        # In practice, this would check if the source typically publishes this type of content
        return 0.6
    
    def _assess_social_cohesion(self, metadata):
        """Assess social cohesion (placeholder implementation)"""
        # In practice, this would analyze social network structure
        return 0.5
    
    def get_feature_vector(self, text, metadata=None):
        """Legacy method for backward compatibility"""
        features = self.extract_advanced_features(text, metadata)
        return torch.tensor(list(features.values()), dtype=torch.float32)
    
    def get_enhanced_feature_vector(self, text, metadata=None):
        """Get enhanced feature vector for model input"""
        features = self.extract_enhanced_features(text, metadata)
        return torch.tensor(list(features.values()), dtype=torch.float32)
    
    def get_feature_names(self):
        """Get comprehensive feature names"""
        sample_features = self.extract_enhanced_features("sample text")
        return list(sample_features.keys())
    
    def analyze_content_risk(self, text, metadata=None):
        """Analyze content risk level for fake news"""
        features = self.extract_enhanced_features(text, metadata)
        
        # Calculate risk score based on key indicators
        risk_indicators = [
            features.get('sensationalism_score', 0),
            features.get('clickbait_score', 0),
            features.get('conspiracy_indicators', 0),
            features.get('emotional_manipulation', 0),
            features.get('manipulation_indicators', 0) / 3,  # Normalize
            1.0 - features.get('domain_trust_indicator', 0.5)  # Low trust = higher risk
        ]
        
        risk_score = sum(risk_indicators) / len(risk_indicators)
        
        # Risk level classification
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'key_risk_factors': self._identify_risk_factors(features)
        }
    
    def _identify_risk_factors(self, features):
        """Identify key risk factors from features"""
        risk_factors = []
        
        if features.get('sensationalism_score', 0) > 0.6:
            risk_factors.append("SENSATIONALISM")
        if features.get('clickbait_score', 0) > 0.5:
            risk_factors.append("CLICKBAIT")
        if features.get('conspiracy_indicators', 0) > 0.4:
            risk_factors.append("CONSPIRACY_CONTENT")
        if features.get('emotional_manipulation', 0) > 0.5:
            risk_factors.append("EMOTIONAL_MANIPULATION")
        if features.get('domain_trust_indicator', 0.5) < 0.3:
            risk_factors.append("LOW_SOURCE_CREDIBILITY")
        
        return risk_factors

# Backward compatibility
AdvancedFeatureEngineer = EnhancedFeatureEngineer