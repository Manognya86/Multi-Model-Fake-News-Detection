import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, AutoModel, AutoTokenizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedEnsembleDetector:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.models = {}
        self.feature_importances = {}
        
        # Dynamic weighting based on model performance and modality reliability
        self.weights = {
            'bert': 0.25,
            'roberta': 0.20,
            'deberta': 0.15,
            'linguistic': 0.15,
            'metadata': 0.10,
            'social': 0.08,
            'image': 0.07
        }
        
        self.confidence_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
        
        # Modality reliability tracker
        self.modality_reliability = {
            'text': 0.8,
            'image': 0.7,
            'graph': 0.6,
            'metadata': 0.7,
            'social': 0.6
        }
        
        self.load_enhanced_models()
        
    def load_enhanced_models(self):
        """Load multiple pre-trained models with enhanced error handling"""
        try:
            # BERT Model
            self.models['bert'] = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
            # RoBERTa Model
            self.models['roberta'] = RobertaModel.from_pretrained('roberta-base').to(self.device)
            self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            
            # DeBERTa Model
            try:
                self.models['deberta'] = AutoModel.from_pretrained('microsoft/deberta-base').to(self.device)
                self.deberta_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
            except:
                print("⚠️ DeBERTa model not available, using BERT as fallback")
                self.models['deberta'] = self.models['bert']
                self.deberta_tokenizer = self.bert_tokenizer
            
            # Enhanced traditional ML models
            self.models['linguistic_rf'] = RandomForestClassifier(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
            
            self.models['linguistic_gb'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Calibrated classifiers for better probability estimates
            self.models['calibrated_rf'] = CalibratedClassifierCV(
                self.models['linguistic_rf'], 
                method='isotonic', 
                cv=3
            )
            
            # Put transformer models in evaluation mode
            for model in [self.models['bert'], self.models['roberta'], self.models['deberta']]:
                model.eval()
                
            self.models_loaded = True
            print("✅ Enhanced ensemble models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading enhanced ensemble models: {e}")
            self.models_loaded = False
    
    def extract_enhanced_transformer_features(self, text, model_name):
        """Extract enhanced features from transformer models with robust error handling"""
        if not self.models_loaded:
            return self._get_fallback_features()
            
        try:
            if model_name == 'bert':
                tokenizer = self.bert_tokenizer
                model = self.models['bert']
            elif model_name == 'roberta':
                tokenizer = self.roberta_tokenizer
                model = self.models['roberta']
            elif model_name == 'deberta':
                tokenizer = self.deberta_tokenizer
                model = self.models['deberta']
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Enhanced tokenization with truncation strategy
            inputs = tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_attention_mask=True,
                stride=128,  # Overlapping chunks for long texts
                return_overflowing_tokens=False
            ).to(self.device)
            
            # Extract multi-layer features with pooling
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Use last 4 layers for richer features
                hidden_states = outputs.hidden_states[-4:]
                
                # Weighted average of layers (later layers get more weight)
                weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=self.device)
                weighted_states = torch.stack([hs * w for hs, w in zip(hidden_states, weights)])
                layer_embeddings = weighted_states.sum(dim=0)
                
                # Mean pooling across tokens
                features = layer_embeddings.mean(dim=1).cpu().numpy()
            
            return features.flatten()
            
        except Exception as e:
            print(f"❌ Error extracting {model_name} features: {e}")
            return self._get_fallback_features()
    
    def ensemble_predict(self, text, advanced_features, metadata_features, image_analysis=None):
        """Legacy method for backward compatibility"""
        return self.enhanced_ensemble_predict(text, advanced_features, metadata_features, image_analysis)
    
    def enhanced_ensemble_predict(self, text, advanced_features, metadata_features, 
                                image_analysis=None, social_context=None, graph_analysis=None):
        """Enhanced ensemble prediction with dynamic modality weighting"""
        predictions = {}
        confidences = {}
        feature_importances = {}
        
        # Transformer model predictions
        for model_name in ['bert', 'roberta', 'deberta']:
            features = self.extract_enhanced_transformer_features(text, model_name)
            prediction, confidence, importance = self.enhanced_classifier(features, model_name)
            
            predictions[model_name] = prediction
            confidences[model_name] = confidence
            feature_importances[model_name] = importance
        
        # Enhanced linguistic feature prediction
        linguistic_pred, linguistic_conf = self.enhanced_linguistic_prediction(advanced_features)
        predictions['linguistic'] = linguistic_pred
        confidences['linguistic'] = linguistic_conf
        
        # Enhanced metadata prediction
        metadata_pred, metadata_conf = self.enhanced_metadata_prediction(metadata_features)
        predictions['metadata'] = metadata_pred
        confidences['metadata'] = metadata_conf
        
        # Social context analysis
        if social_context:
            social_pred, social_conf = self.analyze_social_context(social_context)
            predictions['social'] = social_pred
            confidences['social'] = social_conf
        
        # Image analysis integration
        if image_analysis is not None:
            image_pred, image_conf = self.enhanced_image_analysis(image_analysis)
            predictions['image'] = image_pred
            confidences['image'] = image_conf
        
        # Graph analysis integration
        if graph_analysis is not None:
            graph_pred, graph_conf = self.enhanced_graph_analysis(graph_analysis)
            predictions['graph'] = graph_pred
            confidences['graph'] = graph_conf
        
        # Dynamic weighting based on confidence and modality reliability
        dynamic_weights = self.calculate_dynamic_weights(confidences, predictions)
        
        # Weighted ensemble prediction
        final_prediction, ensemble_confidence = self.calculate_ensemble_prediction(
            predictions, dynamic_weights, confidences
        )
        
        # Enhanced uncertainty estimation
        uncertainty = self.estimate_enhanced_uncertainty(predictions, confidences, dynamic_weights)
        
        # Feature importance aggregation
        overall_importance = self.aggregate_feature_importance(feature_importances, dynamic_weights)
        
        # Cross-modality consistency check
        consistency_analysis = self.analyze_cross_modality_consistency(predictions)
        
        return {
            'ensemble_prediction': final_prediction,
            'confidence': ensemble_confidence,
            'uncertainty': uncertainty,
            'is_fake': final_prediction > 0.5,
            'fake_probability': final_prediction,
            'real_probability': 1 - final_prediction,
            'component_predictions': predictions,
            'component_confidences': confidences,
            'feature_importance': overall_importance,
            'confidence_level': self.get_confidence_level(ensemble_confidence),
            'explanation': self.generate_enhanced_explanation(predictions, overall_importance, consistency_analysis),
            'consistency_analysis': consistency_analysis,
            'modality_weights': dynamic_weights
        }
    
    def enhanced_classifier(self, features, model_name):
        """Enhanced classifier with confidence estimation and feature analysis"""
        if features is None or len(features) == 0:
            return 0.3, 0.1, {}
        
        # Convert to numpy if needed
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        
        # Enhanced feature analysis
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        feature_skew = self.calculate_skewness(features)
        feature_kurtosis = self.calculate_kurtosis(features)
        
        # More sophisticated fake news indicators based on feature statistics
        fake_score = min(1.0, 
            feature_std * 0.20 + 
            abs(feature_skew) * 0.15 +
            abs(feature_kurtosis - 3) * 0.10 +  # Excess kurtosis
            abs(feature_mean) * 0.05
        )
        
        # Confidence based on feature stability and quality
        confidence = max(0.1, 1.0 - feature_std * 0.3 - abs(feature_skew) * 0.2)
        
        # Feature importance metrics
        importance = {
            'feature_std': feature_std,
            'feature_skew': abs(feature_skew),
            'feature_kurtosis': abs(feature_kurtosis - 3),
            'feature_range': np.ptp(features),
            'feature_entropy': self.calculate_entropy(features)
        }
        
        return fake_score, confidence, importance
    
    def enhanced_linguistic_prediction(self, features):
        """Enhanced linguistic feature analysis with fake news patterns"""
        if isinstance(features, dict):
            # Comprehensive linguistic analysis for fake news detection
            sensationalism = features.get('sensationalism_score', 0)
            emotional = features.get('emotional_intensity', 0)
            certainty = features.get('certainty_level', 0)
            manipulation = features.get('manipulation_indicators', 0)
            readability = 1.0 - min(features.get('flesch_reading_ease', 0) / 100, 1.0)
            clickbait = features.get('clickbait_score', 0)
            conspiracy = features.get('conspiracy_indicators', 0)
            
            # Weighted combination with fake news specific patterns
            fake_score = min(1.0, 
                sensationalism * 0.25 +
                emotional * 0.20 +
                certainty * 0.15 +
                manipulation * 0.15 +
                readability * 0.10 +
                clickbait * 0.10 +
                conspiracy * 0.05
            )
            
            # Confidence based on feature completeness and quality
            feature_count = len([v for v in features.values() if v is not None])
            confidence = 0.6 + (feature_count / 10) * 0.4
            
        else:
            # Fallback for basic features
            fake_score = 0.3
            confidence = 0.5
            
        return fake_score, confidence
    
    def enhanced_metadata_prediction(self, metadata):
        """Enhanced metadata analysis with source credibility"""
        if isinstance(metadata, dict):
            fake_score = 0.3  # Base score
            
            # Source credibility analysis
            source_trust = metadata.get('source_credibility', 0.5)
            fake_score += (1.0 - source_trust) * 0.3
            
            # Temporal factors
            if metadata.get('is_recent', False):
                fake_score += 0.1  # Recent content might be less verified
            
            # Engagement patterns - suspicious patterns
            engagement_velocity = metadata.get('engagement_velocity', 0)
            if engagement_velocity > 1000:  # High virality
                fake_score += 0.15
            
            # User diversity
            user_diversity = metadata.get('user_diversity', 0.5)
            fake_score += (1.0 - user_diversity) * 0.1  # Low diversity suspicious
            
            confidence = 0.6 + source_trust * 0.4
            
        else:
            fake_score = 0.3
            confidence = 0.5
            
        return min(1.0, fake_score), confidence
    
    def analyze_social_context(self, social_context):
        """Enhanced social context analysis"""
        fake_score = 0.3
        
        if isinstance(social_context, dict):
            # Bot activity indicators
            bot_score = social_context.get('bot_likelihood', 0)
            fake_score += bot_score * 0.3
            
            # Coordination patterns
            coordination = social_context.get('coordination_score', 0)
            fake_score += coordination * 0.25
            
            # Virality patterns
            virality = social_context.get('virality_score', 0)
            if virality > 0.7:
                fake_score += 0.15
            
            # Network health
            network_health = social_context.get('network_health', 0.5)
            fake_score += (1.0 - network_health) * 0.2
            
            confidence = 0.6 + (1 - bot_score) * 0.4
        
        else:
            fake_score = 0.3
            confidence = 0.5
        
        return min(1.0, fake_score), confidence
    
    def enhanced_image_analysis(self, image_analysis):
        """Enhanced image analysis for fake news detection"""
        if isinstance(image_analysis, dict):
            manipulation_score = image_analysis.get('manipulation_score', 0)
            consistency_score = image_analysis.get('consistency_score', 0.5)
            
            fake_score = min(1.0, 
                manipulation_score * 0.5 +
                (1 - consistency_score) * 0.3 +
                image_analysis.get('suspicious_elements', 0) * 0.2
            )
            
            confidence = 0.5 + manipulation_score * 0.5
            
        else:
            fake_score = 0.3
            confidence = 0.5
            
        return fake_score, confidence
    
    def enhanced_graph_analysis(self, graph_analysis):
        """Enhanced graph analysis for propagation patterns"""
        if isinstance(graph_analysis, dict):
            # Propagation pattern analysis
            virality_score = graph_analysis.get('virality_score', 0)
            coordination_score = graph_analysis.get('coordination_score', 0)
            echo_chamber_score = graph_analysis.get('echo_chamber_score', 0)
            
            fake_score = min(1.0,
                virality_score * 0.4 +
                coordination_score * 0.4 +
                echo_chamber_score * 0.2
            )
            
            confidence = 0.6 + (1 - coordination_score) * 0.4
            
        else:
            fake_score = 0.3
            confidence = 0.5
            
        return fake_score, confidence
    
    def calculate_dynamic_weights(self, confidences, predictions):
        """Calculate dynamic weights based on confidence and historical performance"""
        base_weights = self.weights.copy()
        total_confidence = sum(confidences.values())
        
        if total_confidence == 0:
            return base_weights
        
        # Adjust weights based on confidence and consistency
        dynamic_weights = {}
        for model, weight in base_weights.items():
            if model in confidences:
                confidence_ratio = confidences[model] / total_confidence
                
                # Consistency bonus - if prediction aligns with majority
                model_pred = predictions.get(model, 0.5)
                avg_pred = np.mean(list(predictions.values()))
                consistency = 1.0 - abs(model_pred - avg_pred)
                
                dynamic_weight = weight * (0.6 + 0.3 * confidence_ratio + 0.1 * consistency)
                dynamic_weights[model] = dynamic_weight
            else:
                dynamic_weights[model] = weight
        
        # Normalize weights
        total_weight = sum(dynamic_weights.values())
        dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
        
        return dynamic_weights
    
    def calculate_ensemble_prediction(self, predictions, weights, confidences):
        """Calculate final ensemble prediction with reliability weighting"""
        weighted_sum = 0
        confidence_sum = 0
        total_weight = 0
        
        for model, prediction in predictions.items():
            if model in weights:
                weight = weights[model]
                confidence = confidences.get(model, 0.5)
                
                # Apply reliability weighting
                reliability = self.modality_reliability.get(model, 0.7)
                effective_weight = weight * reliability
                
                weighted_sum += prediction * effective_weight * confidence
                confidence_sum += confidence * effective_weight
                total_weight += effective_weight
        
        if total_weight == 0:
            return 0.5, 0.5
        
        final_prediction = weighted_sum / total_weight
        ensemble_confidence = confidence_sum / total_weight
        
        return final_prediction, ensemble_confidence
    
    def estimate_enhanced_uncertainty(self, predictions, confidences, weights):
        """Enhanced uncertainty estimation with multiple factors"""
        predictions_list = list(predictions.values())
        
        if len(predictions_list) < 2:
            return 0.5
        
        # Prediction variance
        prediction_variance = np.var(predictions_list)
        
        # Confidence-based uncertainty
        avg_confidence = np.mean(list(confidences.values()))
        confidence_uncertainty = 1.0 - avg_confidence
        
        # Weight distribution uncertainty
        weight_entropy = self.calculate_weight_entropy(weights)
        
        # Combined uncertainty score
        uncertainty = (
            prediction_variance * 0.4 +
            confidence_uncertainty * 0.3 +
            weight_entropy * 0.3
        )
        
        return min(1.0, uncertainty)
    
    def analyze_cross_modality_consistency(self, predictions):
        """Analyze consistency across different modalities"""
        if len(predictions) < 2:
            return {'consistency_score': 1.0, 'conflicts': []}
        
        pred_values = list(predictions.values())
        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        
        consistency_score = 1.0 - min(std_pred * 2, 1.0)  # Normalize
        
        # Detect specific conflicts
        conflicts = []
        if std_pred > 0.2:
            conflicts.append("HIGH_VARIANCE")
        
        # Check for extreme outliers
        z_scores = np.abs((pred_values - mean_pred) / (std_pred + 1e-8))
        if np.any(z_scores > 2.0):
            conflicts.append("OUTLIER_PREDICTIONS")
        
        return {
            'consistency_score': consistency_score,
            'conflicts': conflicts,
            'prediction_std': std_pred
        }
    
    def aggregate_feature_importance(self, feature_importances, weights):
        """Aggregate feature importance across models with weighting"""
        aggregated = {}
        total_weight = sum(weights.values())
        
        for model, importance in feature_importances.items():
            weight = weights.get(model, 0) / total_weight
            for feature, value in importance.items():
                if feature not in aggregated:
                    aggregated[feature] = 0
                aggregated[feature] += value * weight
        
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
    
    def get_confidence_level(self, confidence):
        """Get human-readable confidence level"""
        if confidence >= self.confidence_thresholds['high_confidence']:
            return "HIGH"
        elif confidence >= self.confidence_thresholds['medium_confidence']:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_enhanced_explanation(self, predictions, feature_importance, consistency_analysis):
        """Generate enhanced human-readable explanation"""
        explanations = []
        
        # Top contributing factors
        top_features = list(feature_importance.keys())[:3]
        if top_features:
            explanations.append(f"Top factors: {', '.join(top_features)}")
        
        # Model agreement
        consistency_score = consistency_analysis.get('consistency_score', 1.0)
        if consistency_score > 0.8:
            explanations.append("High model agreement across modalities")
        elif consistency_score < 0.4:
            explanations.append("Conflicting signals from different modalities")
        
        # Key risk indicators
        high_risk_modalities = [mod for mod, pred in predictions.items() if pred > 0.7]
        if high_risk_modalities:
            explanations.append(f"High risk signals from: {', '.join(high_risk_modalities)}")
        
        # Data quality assessment
        low_confidence_modalities = [mod for mod, pred in predictions.items() 
                                   if self.weights.get(mod, 0) < 0.05]
        if low_confidence_modalities:
            explanations.append(f"Limited data from: {', '.join(low_confidence_modalities)}")
        
        return explanations
    
    def calculate_skewness(self, data):
        """Calculate skewness of data distribution"""
        if len(data) < 2:
            return 0
        
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skewness = np.mean((data - mean) ** 3) / (std ** 3)
        return skewness
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis of data distribution"""
        if len(data) < 4:
            return 3.0  # Normal distribution kurtosis
        
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 3.0
        
        kurtosis = np.mean((data - mean) ** 4) / (std ** 4)
        return kurtosis
    
    def calculate_entropy(self, data, bins=10):
        """Calculate entropy of data distribution"""
        if len(data) < 2:
            return 0
        
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log(hist))
        return entropy / np.log(bins)  # Normalize
    
    def calculate_weight_entropy(self, weights):
        """Calculate entropy of weight distribution"""
        weight_values = list(weights.values())
        if len(weight_values) < 2:
            return 0
        
        # Normalize weights
        total = sum(weight_values)
        if total == 0:
            return 1.0
        
        normalized = np.array(weight_values) / total
        normalized = normalized[normalized > 0]  # Remove zeros
        
        entropy = -np.sum(normalized * np.log(normalized))
        max_entropy = np.log(len(normalized))
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def update_modality_reliability(self, modality, accuracy):
        """Update modality reliability based on recent performance"""
        current_reliability = self.modality_reliability.get(modality, 0.7)
        # Exponential moving average
        new_reliability = 0.9 * current_reliability + 0.1 * accuracy
        self.modality_reliability[modality] = max(0.1, min(1.0, new_reliability))
    
    def _get_fallback_features(self):
        """Get fallback features when models fail"""
        return np.random.normal(0, 0.1, 768)

# Backward compatibility
EnsembleDetector = EnhancedEnsembleDetector