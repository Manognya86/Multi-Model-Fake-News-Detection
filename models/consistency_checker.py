import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class CrossModalConsistency(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, graph_dim=256, hidden_dim=512):
        super().__init__()
        
        # Text-Image consistency network
        self.text_image_consistency = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Text-Graph alignment network
        self.text_graph_alignment = nn.Sequential(
            nn.Linear(text_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Image-Graph coherence network
        self.image_graph_coherence = nn.Sequential(
            nn.Linear(image_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Overall consistency classifier
        self.consistency_classifier = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Thresholds
        self.consistency_threshold = 0.7
        self.alignment_threshold = 0.6
    
    def forward(self, text_features, image_features, graph_features):
        consistency_scores = {}
        
        # Text-Image consistency
        if image_features is not None and text_features is not None:
            text_image = torch.cat([text_features, image_features], dim=1)
            consistency_scores['text_image'] = self.text_image_consistency(text_image)
        
        # Text-Graph alignment
        if graph_features is not None and text_features is not None:
            text_graph = torch.cat([text_features, graph_features], dim=1)
            consistency_scores['text_graph'] = self.text_graph_alignment(text_graph)
        
        # Image-Graph coherence
        if image_features is not None and graph_features is not None:
            image_graph = torch.cat([image_features, graph_features], dim=1)
            consistency_scores['image_graph'] = self.image_graph_coherence(image_graph)
        
        # Overall consistency score
        if len(consistency_scores) > 0:
            all_scores = torch.cat(list(consistency_scores.values()), dim=1)
            consistency_scores['overall'] = self.consistency_classifier(all_scores)
        
        return consistency_scores
    
    def detect_contradictions(self, text_features, image_features, graph_features, threshold=0.3):
        """Detect contradictions between modalities with enhanced analysis"""
        consistency_scores = self.forward(text_features, image_features, graph_features)
        
        contradictions = {}
        contradiction_details = {}
        
        for modality, score_tensor in consistency_scores.items():
            if modality != 'overall':
                score = score_tensor.item() if torch.is_tensor(score_tensor) else score_tensor
                is_contradiction = score < threshold
                contradictions[modality] = is_contradiction
                contradiction_details[modality] = {
                    'score': score,
                    'is_contradiction': is_contradiction,
                    'severity': 1.0 - score  # Higher severity for lower scores
                }
        
        # Overall contradiction assessment
        if 'overall' in consistency_scores:
            overall_score = consistency_scores['overall'].item()
            has_contradictions = any(contradictions.values())
            contradiction_details['overall'] = {
                'score': overall_score,
                'has_contradictions': has_contradictions,
                'contradiction_count': sum(contradictions.values())
            }
        
        return contradictions, contradiction_details
    
    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.5
    
    def analyze_modality_alignment(self, text, image_caption, graph_summary):
        """Analyze alignment between different modality descriptions"""
        alignment_scores = {}
        
        # Text-Image alignment
        if text and image_caption:
            alignment_scores['text_image'] = self.calculate_semantic_similarity(text, image_caption)
        
        # Text-Graph alignment
        if text and graph_summary:
            alignment_scores['text_graph'] = self.calculate_semantic_similarity(text, graph_summary)
        
        # Image-Graph alignment
        if image_caption and graph_summary:
            alignment_scores['image_graph'] = self.calculate_semantic_similarity(image_caption, graph_summary)
        
        return alignment_scores
    
    def get_consistency_report(self, text_features, image_features, graph_features, 
                             text=None, image_caption=None, graph_summary=None):
        """Generate comprehensive consistency report"""
        # Feature-based consistency
        feature_consistency = self.forward(text_features, image_features, graph_features)
        contradictions, contradiction_details = self.detect_contradictions(
            text_features, image_features, graph_features
        )
        
        # Semantic alignment (if text descriptions available)
        semantic_alignment = {}
        if text and (image_caption or graph_summary):
            semantic_alignment = self.analyze_modality_alignment(text, image_caption, graph_summary)
        
        # Overall assessment
        overall_score = feature_consistency.get('overall', torch.tensor([0.5])).item()
        has_contradictions = any(contradictions.values())
        
        report = {
            'overall_consistency': overall_score,
            'has_contradictions': has_contradictions,
            'contradiction_count': sum(contradictions.values()),
            'feature_consistency': {
                k: v.item() if torch.is_tensor(v) else v 
                for k, v in feature_consistency.items()
            },
            'contradiction_details': contradiction_details,
            'semantic_alignment': semantic_alignment,
            'assessment': self._get_consistency_assessment(overall_score, has_contradictions)
        }
        
        return report
    
    def _get_consistency_assessment(self, overall_score, has_contradictions):
        """Get human-readable consistency assessment"""
        if overall_score >= 0.8 and not has_contradictions:
            return "HIGH_CONSISTENCY"
        elif overall_score >= 0.6 and not has_contradictions:
            return "MODERATE_CONSISTENCY"
        elif has_contradictions:
            return "CONTRADICTIONS_DETECTED"
        else:
            return "LOW_CONSISTENCY"

class ModalityImportanceWeighter(nn.Module):
    """Dynamically weight modalities based on their reliability and consistency"""
    
    def __init__(self, text_dim=768, image_dim=768, graph_dim=256, metadata_dim=64):
        super().__init__()
        
        self.importance_predictor = nn.Sequential(
            nn.Linear(text_dim + image_dim + graph_dim + metadata_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # 4 modalities: text, image, graph, metadata
            nn.Softmax(dim=1)
        )
        
        # Reliability history
        self.reliability_history = {
            'text': 0.8,
            'image': 0.7,
            'graph': 0.6,
            'metadata': 0.7
        }
    
    def forward(self, text_features, image_features, graph_features, metadata_features):
        # Concatenate all features
        all_features = torch.cat([
            text_features, 
            image_features, 
            graph_features, 
            metadata_features
        ], dim=1)
        
        # Predict base importance weights
        base_weights = self.importance_predictor(all_features)
        
        # Apply reliability adjustments
        reliability_factors = torch.tensor([
            self.reliability_history['text'],
            self.reliability_history['image'], 
            self.reliability_history['graph'],
            self.reliability_history['metadata']
        ], device=base_weights.device)
        
        adjusted_weights = base_weights * reliability_factors
        normalized_weights = adjusted_weights / adjusted_weights.sum(dim=1, keepdim=True)
        
        return {
            'text_weight': normalized_weights[:, 0],
            'image_weight': normalized_weights[:, 1],
            'graph_weight': normalized_weights[:, 2],
            'metadata_weight': normalized_weights[:, 3],
            'base_weights': base_weights
        }
    
    def update_reliability(self, modality, accuracy, consistency):
        """Update modality reliability based on performance"""
        current = self.reliability_history.get(modality, 0.7)
        
        # Combined score from accuracy and consistency
        performance_score = (accuracy + consistency) / 2
        
        # Exponential moving average
        new_reliability = 0.9 * current + 0.1 * performance_score
        self.reliability_history[modality] = max(0.1, min(1.0, new_reliability))
    
    def get_reliability_report(self):
        """Get current reliability status of all modalities"""
        return self.reliability_history.copy()

class ConsistencyEnhancedFusion(nn.Module):
    """Fusion mechanism that considers cross-modal consistency"""
    
    def __init__(self, feature_dims, hidden_dim=512):
        super().__init__()
        
        self.consistency_checker = CrossModalConsistency()
        self.importance_weighter = ModalityImportanceWeighter()
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(sum(feature_dims.values()), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, modality_features, modality_descriptions=None):
        # Extract features
        text_features = modality_features.get('text')
        image_features = modality_features.get('image') 
        graph_features = modality_features.get('graph')
        metadata_features = modality_features.get('metadata')
        
        # Check consistency
        consistency_report = self.consistency_checker.get_consistency_report(
            text_features, image_features, graph_features,
            modality_descriptions.get('text') if modality_descriptions else None,
            modality_descriptions.get('image') if modality_descriptions else None,
            modality_descriptions.get('graph') if modality_descriptions else None
        )
        
        # Calculate importance weights
        importance_weights = self.importance_weighter(
            text_features, image_features, graph_features, metadata_features
        )
        
        # Apply consistency-adjusted weighting
        consistency_factors = self._calculate_consistency_factors(consistency_report)
        final_weights = self._apply_consistency_adjustment(importance_weights, consistency_factors)
        
        # Weighted feature combination
        weighted_features = (
            text_features * final_weights['text_weight'].unsqueeze(-1) +
            image_features * final_weights['image_weight'].unsqueeze(-1) +
            graph_features * final_weights['graph_weight'].unsqueeze(-1) +
            metadata_features * final_weights['metadata_weight'].unsqueeze(-1)
        )
        
        # Final classification
        logits = self.fusion_network(weighted_features)
        
        return {
            'logits': logits,
            'consistency_report': consistency_report,
            'importance_weights': importance_weights,
            'final_weights': final_weights
        }
    
    def _calculate_consistency_factors(self, consistency_report):
        """Calculate adjustment factors based on consistency"""
        factors = {}
        
        consistency_scores = consistency_report.get('feature_consistency', {})
        
        for modality_pair in ['text_image', 'text_graph', 'image_graph']:
            if modality_pair in consistency_scores:
                score = consistency_scores[modality_pair]
                # Higher consistency = higher trust in both modalities
                factors[modality_pair] = score
        
        return factors
    
    def _apply_consistency_adjustment(self, importance_weights, consistency_factors):
        """Apply consistency-based adjustments to importance weights"""
        adjusted_weights = importance_weights.copy()
        
        # Adjust weights based on pairwise consistency
        for pair, consistency in consistency_factors.items():
            mod1, mod2 = pair.split('_')
            
            # Increase weights for consistent modalities
            boost = (consistency - 0.5) * 0.2  # Adjust boost factor as needed
            
            if mod1 in adjusted_weights:
                adjusted_weights[mod1 + '_weight'] += boost
            if mod2 in adjusted_weights:
                adjusted_weights[mod2 + '_weight'] += boost
        
        # Renormalize weights
        weight_keys = [k for k in adjusted_weights.keys() if k.endswith('_weight')]
        total_weight = sum(adjusted_weights[k].sum() for k in weight_keys)
        
        if total_weight > 0:
            for key in weight_keys:
                adjusted_weights[key] = adjusted_weights[key] / total_weight
        
        return adjusted_weights