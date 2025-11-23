import torch
import torch.nn as nn
from transformers import BertModel
from .image_model import ImageModel
from .graph_model import EnhancedGNNModel
from .consistency_checker import CrossModalConsistency, ModalityImportanceWeighter

class UnifiedMultimodalModel(nn.Module):
    """Unified model that integrates all modalities with dynamic weighting"""
    
    def __init__(self, text_model_name='bert-base-uncased', metadata_dim=6, 
                 image_dim=768, graph_dim=256, n_classes=2, dropout_prob=0.3):
        super().__init__()
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        # Image encoder
        self.image_encoder = ImageModel()
        
        # Graph encoder  
        self.graph_encoder = EnhancedGNNModel()
        
        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Calculate total input dimension for classifier
        self.total_feature_dim = text_dim + image_dim + graph_dim + 64  # 768 + 768 + 256 + 64 = 1856
        
        # Modality importance weighter
        self.importance_weighter = ModalityImportanceWeighter(
            text_dim=text_dim,
            image_dim=image_dim,
            graph_dim=graph_dim,
            metadata_dim=64
        )
        
        # Cross-modal consistency checker
        self.consistency_checker = CrossModalConsistency()
        
        # FIXED: Unified classifier with correct dimensions
        self.feature_gate = nn.Sequential(
            nn.Linear(self.total_feature_dim, self.total_feature_dim),  # Match input dimension
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.total_feature_dim, 512),  # 1856 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(128, n_classes)
        )
        
        # Modality dropout for robustness
        self.modality_dropout = nn.Dropout(0.1)
        
        print(f"🚀 Unified Multimodal Model initialized")
        print(f"📊 Modalities: Text({text_dim}), Image({image_dim}), Graph({graph_dim}), Metadata(64)")
        print(f"🎯 Total feature dimension: {self.total_feature_dim}")
        print(f"🔧 Classifier input: {self.total_feature_dim} -> 512")
        
    def forward(self, input_ids, attention_mask, metadata, image_data=None, graph_data=None):
        batch_size = input_ids.size(0)
        
        # Encode text
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.pooler_output
        
        # Encode metadata
        metadata_features = self.metadata_encoder(metadata)
        
        # Encode image (with fallback)
        if image_data is not None and not isinstance(image_data, list):
            # FIX: Check if image_data is valid tensor
            if torch.is_tensor(image_data) and image_data.dim() == 4:  # [batch, channels, height, width]
                image_features, _ = self.image_encoder(image_data)
            else:
                image_features = torch.zeros(batch_size, 768, device=input_ids.device)
        else:
            image_features = torch.zeros(batch_size, 768, device=input_ids.device)
        
        # Encode graph (with fallback)
        graph_features = torch.zeros(batch_size, 256, device=input_ids.device)
        if graph_data is not None and not isinstance(graph_data, list):
            try:
                # FIX: Handle graph data properly
                if isinstance(graph_data, tuple) and len(graph_data) >= 3:
                    x, edge_index, batch = graph_data
                    if torch.is_tensor(x) and torch.is_tensor(edge_index) and torch.is_tensor(batch):
                        graph_features, _ = self.graph_encoder(x, edge_index, batch)
            except Exception as e:
                print(f"⚠️ Graph encoding failed: {e}")
                graph_features = torch.zeros(batch_size, 256, device=input_ids.device)
        
        # Apply modality dropout for robustness
        if self.training:
            if torch.rand(1) < 0.1:  # 10% chance to drop image
                image_features = torch.zeros_like(image_features)
            if torch.rand(1) < 0.1:  # 10% chance to drop graph
                graph_features = torch.zeros_like(graph_features)
        
        # Calculate modality importance weights
        importance_weights = self.importance_weighter(
            text_features, image_features, graph_features, metadata_features
        )
        
        # Weight features by importance
        text_weighted = text_features * importance_weights['text_weight'].unsqueeze(1)
        image_weighted = image_features * importance_weights['image_weight'].unsqueeze(1)
        graph_weighted = graph_features * importance_weights['graph_weight'].unsqueeze(1)
        metadata_weighted = metadata_features * importance_weights['metadata_weight'].unsqueeze(1)
        
        # Combine all modalities - THIS IS 1856 DIMENSIONS
        combined_features = torch.cat([
            text_weighted, 
            image_weighted, 
            graph_weighted, 
            metadata_weighted
        ], dim=1)
        
        # Apply feature gating - FIXED: Now dimensions match (1856 -> 1856)
        gate_weights = self.feature_gate(combined_features)
        gated_features = combined_features * gate_weights
        
        # Check cross-modal consistency
        consistency_scores = self.consistency_checker(
            text_features, image_features, graph_features
        )
        
        # Final classification - FIXED: Now dimensions match (1856 -> 512)
        logits = self.classifier(gated_features)
        
        return {
            'logits': logits,
            'consistency_scores': consistency_scores,
            'importance_weights': importance_weights,
            'modality_features': {
                'text': text_features,
                'image': image_features, 
                'graph': graph_features,
                'metadata': metadata_features
            }
        }
    
    def get_modality_contributions(self, input_ids, attention_mask, metadata, image_data=None, graph_data=None):
        """Get contribution of each modality to the final decision"""
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask, metadata, image_data, graph_data)
            
            # Calculate individual modality contributions
            modalities = ['text', 'image', 'graph', 'metadata']
            contributions = {}
            
            for modality in modalities:
                if modality + '_weight' in output['importance_weights']:
                    weight = output['importance_weights'][modality + '_weight'].mean().item()
                    feature_norm = torch.norm(output['modality_features'][modality], dim=1).mean().item()
                    contributions[modality] = weight * feature_norm
            
            # Normalize contributions
            total = sum(contributions.values())
            if total > 0:
                contributions = {k: v/total for k, v in contributions.items()}
            
            return contributions