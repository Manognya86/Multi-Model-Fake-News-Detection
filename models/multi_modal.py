import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np

class MultiModalIntegrator:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        try:
            # Initialize Vision Transformer for image analysis
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(self.device)
            self.vit_model.eval()
            self.image_analysis_available = True
        except Exception as e:
            print(f"Image analysis models not available: {e}")
            self.image_analysis_available = False
        
    def analyze_image(self, image_url_or_path):
        """Analyze image for manipulation detection and context analysis"""
        if not self.image_analysis_available:
            return None
            
        try:
            # Load image
            if image_url_or_path.startswith('http'):
                response = requests.get(image_url_or_path, timeout=10)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_url_or_path)
            
            # Preprocess image
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                image_features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
            
            # Analyze image characteristics
            analysis = {
                'image_features': image_features.cpu(),
                'image_size': image.size,
                'format': image.format,
                'mode': image.mode,
                'manipulation_score': self.detect_manipulation_indicators(image)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Image analysis error: {e}")
            return None
    
    def detect_manipulation_indicators(self, image):
        """Simple heuristic for image manipulation detection"""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV
            open_cv_image = np.array(image)
            if len(open_cv_image.shape) == 3:
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
            
            # Simple edge analysis (crude manipulation detection)
            edges = cv2.Canny(open_cv_image, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # High edge density might indicate splicing or manipulation
            manipulation_score = min(1.0, edge_density * 10)
            
            return manipulation_score
        except:
            return 0.0
    
    def check_multimodal_consistency(self, text, image_analysis):
        """Check consistency between text and image content"""
        if image_analysis is None:
            return 0.5  # Neutral if no image
        
        # Simple keyword-based consistency check
        text_lower = text.lower()
        image_keywords = {
            'person': ['person', 'people', 'man', 'woman', 'face'],
            'object': ['object', 'thing', 'item', 'product'],
            'scene': ['scene', 'landscape', 'building', 'street'],
            'document': ['document', 'text', 'paper', 'screenshot']
        }
        
        # Check if text mentions image content
        consistency_score = 0.0
        matches = 0
        
        for category, keywords in image_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                matches += 1
        
        consistency_score = matches / len(image_keywords)
        return consistency_score
    
    # NEW METHOD: Enhanced multimodal analysis
    def enhanced_multimodal_analysis(self, text, image_url_or_path):
        """Enhanced analysis combining text and image"""
        image_analysis = self.analyze_image(image_url_or_path)
        consistency_score = self.check_multimodal_consistency(text, image_analysis)
        
        return {
            'image_analysis': image_analysis,
            'consistency_score': consistency_score,
            'multimodal_trust_score': self.calculate_multimodal_trust(
                image_analysis, consistency_score
            )
        }
    
    # NEW METHOD: Calculate multimodal trust score
    def calculate_multimodal_trust(self, image_analysis, consistency_score):
        """Calculate overall trust score from multimodal analysis"""
        if image_analysis is None:
            return 0.5
        
        manipulation_score = image_analysis.get('manipulation_score', 0)
        
        # High consistency and low manipulation = high trust
        trust_score = (consistency_score * 0.6 + (1 - manipulation_score) * 0.4)
        return trust_score