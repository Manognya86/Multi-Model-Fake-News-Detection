import requests
import os
import time

class LiveFactChecking:
    def __init__(self, config):
        self.config = config
        self.api_keys = {
            'google': config.google_fact_check_api_key,
            'claimbuster': config.claimbuster_api_key
        }
        self.last_api_call = 0
        self.rate_limit_delay = 1  # seconds between API calls
    
    def google_fact_check(self, claim):
        """Check claim using Google Fact Check API"""
        if not self.api_keys['google']:
            return None
            
        # Rate limiting
        self._wait_for_rate_limit()
        
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            'query': claim,
            'key': self.api_keys['google']
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Google Fact Check API returned status: {response.status_code}")
        except Exception as e:
            print(f"Google Fact Check API error: {e}")
        
        return None
    
    def claimbuster_check(self, claim):
        """Check claim using ClaimBuster API"""
        if not self.api_keys['claimbuster']:
            return None
            
        # Rate limiting
        self._wait_for_rate_limit()
            
        url = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
        headers = {
            'x-api-key': self.api_keys['claimbuster']
        }
        data = {
            'text': claim
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"ClaimBuster API returned status: {response.status_code}")
        except Exception as e:
            print(f"ClaimBuster API error: {e}")
        
        return None
    
    def check_claim(self, claim):
        """Comprehensive fact-checking for a claim"""
        results = {
            'sources_checked': [],
            'rating_summary': None,
            'confidence': 0,
            'explanations': [],
            'overall_assessment': 'LOW_CONFIDENCE'
        }
        
        # Google Fact Check
        google_result = self.google_fact_check(claim)
        if google_result:
            results['sources_checked'].append('google_fact_check')
            rating = self.parse_google_result(google_result)
            if rating:
                results['rating_summary'] = rating
                results['confidence'] += 0.6
                results['explanations'].append(f"Google Fact Check: {rating}")
        
        # ClaimBuster
        claimbuster_result = self.claimbuster_check(claim)
        if claimbuster_result:
            results['sources_checked'].append('claimbuster')
            cb_score = self.parse_claimbuster_result(claimbuster_result)
            if cb_score > 0.7:
                results['confidence'] += 0.4
                results['explanations'].append("Claim matches patterns of check-worthy statements.")
            elif cb_score > 0.5:
                results['confidence'] += 0.2
                results['explanations'].append("Claim shows moderate check-worthiness.")
        
        # NEW: Fallback to internal analysis if no APIs available
        if not results['sources_checked']:
            internal_score = self.internal_claim_analysis(claim)
            results['confidence'] = internal_score
            results['explanations'].append("Using internal claim analysis (no external APIs available).")
            results['sources_checked'].append('internal_analysis')
        
        # Generate overall assessment
        if results['confidence'] > 0.7:
            results['overall_assessment'] = 'HIGH_CONFIDENCE'
        elif results['confidence'] > 0.4:
            results['overall_assessment'] = 'MEDIUM_CONFIDENCE'
        else:
            results['overall_assessment'] = 'LOW_CONFIDENCE'
        
        return results
    
    def parse_google_result(self, result):
        """Parse Google Fact Check API result"""
        if 'claims' in result and result['claims']:
            claim_review = result['claims'][0].get('claimReview', [])
            if claim_review:
                return claim_review[0].get('textualRating', '')
        return None
    
    def parse_claimbuster_result(self, result):
        """Parse ClaimBuster API result"""
        if 'results' in result and result['results']:
            return result['results'][0].get('score', 0)
        return 0
    
    # NEW METHOD: Internal claim analysis when APIs are unavailable
    def internal_claim_analysis(self, claim):
        """Internal analysis of claim using simple heuristics"""
        claim_lower = claim.lower()
        
        # Check for sensational language
        sensational_words = ['breaking', 'shocking', 'amazing', 'unbelievable', 'secret']
        sensational_count = sum(1 for word in sensational_words if word in claim_lower)
        
        # Check for excessive punctuation
        exclamation_count = claim.count('!')
        question_count = claim.count('?')
        
        # Calculate internal score
        score = min(1.0, (sensational_count * 0.2) + (exclamation_count * 0.1) + (question_count * 0.05))
        return score
    
    # NEW METHOD: Rate limiting helper
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)
        
        self.last_api_call = time.time()