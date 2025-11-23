import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class CrossModalConsistency:
    def __init__(self):
        self.semantic_similarity_threshold = 0.7
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def check_text_image_consistency(self, text_embedding, image_embedding, text_description, image_caption=None):
        """Enhanced consistency check between text and image content"""
        if text_embedding is None or image_embedding is None:
            return {
                'consistency_score': 0.5,
                'is_consistent': True,
                'embedding_similarity': 0.5,
                'content_similarity': 0.5,
                'confidence': 0.1
            }
        
        # Convert to numpy if they are tensors
        if torch.is_tensor(text_embedding):
            text_embedding = text_embedding.detach().cpu().numpy()
        if torch.is_tensor(image_embedding):
            image_embedding = image_embedding.detach().cpu().numpy()
        
        # Ensure 2D arrays for similarity calculation
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)
        if image_embedding.ndim == 1:
            image_embedding = image_embedding.reshape(1, -1)
        
        # Semantic similarity between text and image embeddings
        embedding_similarity = cosine_similarity(text_embedding, image_embedding)[0][0]
        
        # Content consistency (if image caption is available)
        if image_caption and text_description:
            content_similarity = self._calculate_content_similarity(text_description, image_caption)
        else:
            # Fallback: use embedding similarity
            content_similarity = embedding_similarity
        
        # Combined consistency score with weighting
        consistency_score = (embedding_similarity * 0.6 + content_similarity * 0.4)
        
        # Confidence based on input quality
        confidence = min(1.0, (len(text_description) / 100) * 0.5 + 0.5)
        
        return {
            'consistency_score': float(consistency_score),
            'is_consistent': consistency_score > self.semantic_similarity_threshold,
            'embedding_similarity': float(embedding_similarity),
            'content_similarity': float(content_similarity),
            'confidence': float(confidence)
        }
    
    def check_text_graph_alignment(self, text_embedding, graph_embedding, propagation_pattern):
        """Check alignment between text content and propagation patterns"""
        if text_embedding is None or graph_embedding is None:
            return {
                'alignment_score': 0.5,
                'is_aligned': True,
                'embedding_similarity': 0.5,
                'propagation_consistency': 0.5
            }
        
        # Convert to numpy if they are tensors
        if torch.is_tensor(text_embedding):
            text_embedding = text_embedding.detach().cpu().numpy()
        if torch.is_tensor(graph_embedding):
            graph_embedding = graph_embedding.detach().cpu().numpy()
        
        # Ensure 2D arrays
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)
        if graph_embedding.ndim == 1:
            graph_embedding = graph_embedding.reshape(1, -1)
        
        # Embedding similarity
        embedding_similarity = cosine_similarity(text_embedding, graph_embedding)[0][0]
        
        # Propagation pattern analysis
        propagation_consistency = self._analyze_propagation_consistency(propagation_pattern)
        
        # Combined alignment score
        alignment_score = (embedding_similarity * 0.7 + propagation_consistency * 0.3)
        
        return {
            'alignment_score': float(alignment_score),
            'is_aligned': alignment_score > 0.6,
            'embedding_similarity': float(embedding_similarity),
            'propagation_consistency': float(propagation_consistency)
        }
    
    def detect_cross_modal_contradictions(self, text_features, image_features, graph_features, metadata):
        """Detect contradictions across different modalities"""
        contradictions = []
        contradiction_score = 0.0
        
        # Text-Image contradiction
        if text_features is not None and image_features is not None:
            text_image_consistency = self.check_text_image_consistency(
                text_features, image_features, "text", "image"
            )
            if not text_image_consistency['is_consistent']:
                contradictions.append("TEXT_IMAGE_MISMATCH")
                contradiction_score += 0.4
        
        # Text-Graph contradiction
        if text_features is not None and graph_features is not None:
            text_graph_alignment = self.check_text_graph_alignment(
                text_features, graph_features, metadata.get('propagation_pattern', {})
            )
            if not text_graph_alignment['is_aligned']:
                contradictions.append("TEXT_GRAPH_MISALIGNMENT")
                contradiction_score += 0.3
        
        # Metadata consistency
        metadata_consistency = self._check_metadata_consistency(metadata)
        if not metadata_consistency['is_consistent']:
            contradictions.append("METADATA_INCONSISTENCY")
            contradiction_score += 0.3
        
        return {
            'contradictions': contradictions,
            'contradiction_score': min(1.0, contradiction_score),
            'has_contradictions': len(contradictions) > 0
        }
    
    def _calculate_content_similarity(self, text1, text2):
        """Calculate semantic similarity between two text descriptions"""
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating content similarity: {e}")
            return 0.5
    
    def _analyze_propagation_consistency(self, propagation_pattern):
        """Analyze consistency of propagation patterns with content"""
        if not propagation_pattern:
            return 0.5
        
        # Analyze propagation velocity
        velocity = propagation_pattern.get('velocity', 0)
        velocity_score = 1.0 - min(velocity / 1000, 1.0)  # Lower velocity is more consistent
        
        # Analyze user diversity
        diversity = propagation_pattern.get('user_diversity', 0.5)
        diversity_score = diversity
        
        # Analyze temporal patterns
        temporal_consistency = propagation_pattern.get('temporal_consistency', 0.5)
        
        return (velocity_score * 0.4 + diversity_score * 0.3 + temporal_consistency * 0.3)
    
    def _check_metadata_consistency(self, metadata):
        """Check internal consistency of metadata"""
        if not metadata:
            return {'is_consistent': True, 'score': 0.5}
        
        consistency_indicators = []
        
        # Source-domain consistency
        source = metadata.get('source', '')
        domain = metadata.get('domain', '')
        if source and domain:
            source_in_domain = source.lower() in domain.lower()
            consistency_indicators.append(1.0 if source_in_domain else 0.5)
        
        # Temporal consistency
        created_time = metadata.get('created_time')
        modified_time = metadata.get('modified_time')
        if created_time and modified_time:
            time_consistent = modified_time >= created_time
            consistency_indicators.append(1.0 if time_consistent else 0.3)
        
        # Engagement consistency
        shares = metadata.get('shares', 0)
        likes = metadata.get('likes', 0)
        if shares > 0 and likes > 0:
            engagement_ratio = likes / shares
            # Normal engagement ratio is typically between 0.1 and 10
            engagement_consistent = 0.1 <= engagement_ratio <= 10
            consistency_indicators.append(1.0 if engagement_consistent else 0.4)
        
        if not consistency_indicators:
            return {'is_consistent': True, 'score': 0.5}
        
        avg_consistency = sum(consistency_indicators) / len(consistency_indicators)
        
        return {
            'is_consistent': avg_consistency > 0.7,
            'score': avg_consistency
        }

class ModalityFusion:
    """Advanced modality fusion techniques"""
    
    def __init__(self, fusion_type='weighted'):
        self.fusion_type = fusion_type
        
    def weighted_fusion(self, modality_features, modality_weights):
        """Weighted fusion of modality features"""
        fused_features = None
        
        for modality, features in modality_features.items():
            if features is not None:
                weight = modality_weights.get(modality, 0.0)
                weighted_features = features * weight
                
                if fused_features is None:
                    fused_features = weighted_features
                else:
                    # Ensure same dimension
                    min_dim = min(fused_features.shape[-1], weighted_features.shape[-1])
                    fused_features[..., :min_dim] += weighted_features[..., :min_dim]
        
        return fused_features
    
    def attention_fusion(self, modality_features, attention_weights):
        """Attention-based fusion of modalities"""
        # Convert to tensors if needed
        if not all(torch.is_tensor(feat) for feat in modality_features.values() if feat is not None):
            modality_features = {k: torch.tensor(v) if v is not None else None 
                               for k, v in modality_features.items()}
        
        # Apply attention weights
        attended_features = []
        for modality, features in modality_features.items():
            if features is not None and modality in attention_weights:
                attention = attention_weights[modality]
                if torch.is_tensor(attention):
                    attention = attention.unsqueeze(-1)
                attended = features * attention
                attended_features.append(attended)
        
        if attended_features:
            # Concatenate and reduce
            concatenated = torch.cat(attended_features, dim=-1)
            # Use mean pooling for fusion
            fused = concatenated.mean(dim=-1, keepdim=True)
            return fused
        else:
            return None
    
    def cross_modal_attention(self, query_modality, key_modalities, value_modalities):
        """Cross-modal attention mechanism"""
        # Simple implementation - in practice, this would use transformer layers
        similarities = {}
        
        for key_modality, key_features in key_modalities.items():
            if key_features is not None and query_modality is not None:
                # Calculate similarity between query and key
                similarity = F.cosine_similarity(query_modality, key_features, dim=-1)
                similarities[key_modality] = similarity
        
        # Normalize similarities to get attention weights
        if similarities:
            attention_weights = torch.softmax(torch.stack(list(similarities.values())), dim=0)
            
            # Apply attention to value modalities
            attended_values = []
            for (modality, value), weight in zip(value_modalities.items(), attention_weights):
                if value is not None:
                    attended = value * weight.unsqueeze(-1)
                    attended_values.append(attended)
            
            if attended_values:
                return torch.stack(attended_values).mean(dim=0)
        
        return query_modality  # Fallback to original query

def calculate_modality_reliability(modality_features, historical_accuracy):
    """Calculate reliability scores for each modality"""
    reliability_scores = {}
    
    for modality, features in modality_features.items():
        if features is not None:
            # Feature quality indicators
            if torch.is_tensor(features):
                feature_variance = features.var().item()
                feature_magnitude = features.norm().item()
            else:
                feature_variance = np.var(features)
                feature_magnitude = np.linalg.norm(features)
            
            # Historical accuracy for this modality
            historical_score = historical_accuracy.get(modality, 0.5)
            
            # Combine indicators
            reliability = (
                historical_score * 0.6 +
                (1 - min(feature_variance, 1.0)) * 0.2 +
                min(feature_magnitude / 10, 1.0) * 0.2
            )
            
            reliability_scores[modality] = max(0.1, min(1.0, reliability))
        else:
            reliability_scores[modality] = 0.1  # Low reliability if no features
    
    return reliability_scores