import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedMultimodalFusion(nn.Module):
    """Unified fusion for all 4 modalities with cross-attention and dynamic weighting"""
    
    def __init__(self, text_dim=768, image_dim=768, graph_dim=256, metadata_dim=64, 
                 hidden_dim=512, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = 4
        
        # Project all modalities to same dimension
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.graph_projection = nn.Linear(graph_dim, hidden_dim)
        self.metadata_projection = nn.Linear(metadata_dim, hidden_dim)
        
        # Layer normalization for each modality
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.metadata_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer encoder for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-modal attention with multiple heads
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Modality importance weighting
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Feature gating mechanism
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, text_features, image_features, graph_features, metadata_features):
        batch_size = text_features.size(0)
        
        # Project all modalities to common space with normalization
        text_proj = self.text_norm(self.text_projection(text_features)).unsqueeze(1)
        image_proj = self.image_norm(self.image_projection(image_features)).unsqueeze(1)
        graph_proj = self.graph_norm(self.graph_projection(graph_features)).unsqueeze(1)
        metadata_proj = self.metadata_norm(self.metadata_projection(metadata_features)).unsqueeze(1)
        
        # Combine features (batch_size, 4, hidden_dim)
        combined = torch.cat([text_proj, image_proj, graph_proj, metadata_proj], dim=1)
        
        # Apply transformer fusion
        fused = self.transformer_encoder(combined)
        
        # Apply cross-attention
        attended, attention_weights = self.cross_attention(fused, fused, fused)
        
        # Calculate modality importance weights
        flattened = attended.view(batch_size, -1)
        importance_weights = self.importance_predictor(flattened)
        
        # Apply importance weights
        weighted_modalities = []
        for i in range(self.num_modalities):
            weighted = attended[:, i] * importance_weights[:, i].unsqueeze(-1)
            weighted_modalities.append(weighted)
        
        # Combine weighted modalities
        combined_weighted = torch.stack(weighted_modalities, dim=1).sum(dim=1)
        
        # Apply feature gating
        gate_weights = self.feature_gate(combined_weighted)
        gated_features = combined_weighted * gate_weights
        
        # Final projection and normalization
        output_features = self.output_norm(self.output_projection(gated_features))
        
        # Classification
        logits = self.classifier(output_features)
        
        return {
            'logits': logits,
            'features': output_features,
            'attention_weights': attention_weights,
            'importance_weights': importance_weights,
            'modality_embeddings': {
                'text': text_proj.squeeze(1),
                'image': image_proj.squeeze(1),
                'graph': graph_proj.squeeze(1),
                'metadata': metadata_proj.squeeze(1)
            }
        }

class TransformerFusion(nn.Module):
    def __init__(self, text_dim, metadata_dim, hidden_dim=512, num_heads=8, num_layers=3):
        super(TransformerFusion, self).__init__()
        
        # Project all modalities to same dimension
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.metadata_projection = nn.Linear(metadata_dim, hidden_dim)
        
        # Transformer encoder for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, text_features, metadata_features, image_features=None, graph_features=None):
        # Project to common space
        text_proj = self.text_projection(text_features).unsqueeze(1)
        metadata_proj = self.metadata_projection(metadata_features).unsqueeze(1)
        
        modalities = [text_proj, metadata_proj]
        
        # Optional additional modalities
        if image_features is not None:
            if not hasattr(self, 'image_projection'):
                self.image_projection = nn.Linear(image_features.shape[-1], text_proj.shape[-1]).to(text_features.device)
            image_proj = self.image_projection(image_features).unsqueeze(1)
            modalities.append(image_proj)
            
        if graph_features is not None:
            if not hasattr(self, 'graph_projection'):
                self.graph_projection = nn.Linear(graph_features.shape[-1], text_proj.shape[-1]).to(text_features.device)
            graph_proj = self.graph_projection(graph_features).unsqueeze(1)
            modalities.append(graph_proj)
        
        combined = torch.cat(modalities, dim=1)
        fused = self.transformer_encoder(combined)
        fused_pooled = fused.mean(dim=1)
        output = self.classifier(fused_pooled)
        
        return output

class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim, metadata_dim, hidden_dim=512):
        super(CrossAttentionFusion, self).__init__()
        
        # Projections
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.metadata_projection = nn.Linear(metadata_dim, hidden_dim)
        
        # Cross-attention layers
        self.text_to_metadata_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.metadata_to_text_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, text_features, metadata_features, image_features=None, graph_features=None):
        # Project features
        text_proj = self.text_projection(text_features).unsqueeze(1)
        metadata_proj = self.metadata_projection(metadata_features).unsqueeze(1)
        
        # Cross attention
        text_attended, _ = self.text_to_metadata_attention(
            text_proj, metadata_proj, metadata_proj
        )
        metadata_attended, _ = self.metadata_to_text_attention(
            metadata_proj, text_proj, text_proj
        )
        
        # Combine attended features
        text_attended = text_attended.squeeze(1)
        metadata_attended = metadata_attended.squeeze(1)
        
        combined = torch.cat([text_attended, metadata_attended], dim=1)
        
        # Fusion and classification
        fused = self.fusion_layer(combined)
        output = self.classifier(fused)
        
        return output

class DynamicModalityFusion(nn.Module):
    """Dynamically weights modalities based on their reliability"""
    
    def __init__(self, modality_dims, hidden_dim=512):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.projections[modality] = nn.Linear(dim, hidden_dim)
        
        # Reliability predictor
        self.reliability_predictor = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, modality_features):
        # Project all modalities
        projected = {}
        for modality, features in modality_features.items():
            if features is not None:
                projected[modality] = self.projections[modality](features)
            else:
                # Use zeros if modality missing
                projected[modality] = torch.zeros(
                    list(modality_features.values())[0].size(0), 
                    self.projections[list(self.projections.keys())[0]].out_features
                ).to(next(self.parameters()).device)
        
        # Concatenate for reliability prediction
        all_features = torch.cat([projected[modality] for modality in self.modality_dims.keys()], dim=1)
        reliability_weights = self.reliability_predictor(all_features)
        
        # Weighted combination
        weighted_sum = None
        for i, modality in enumerate(self.modality_dims.keys()):
            weight = reliability_weights[:, i].unsqueeze(-1)
            weighted_feature = projected[modality] * weight
            
            if weighted_sum is None:
                weighted_sum = weighted_feature
            else:
                weighted_sum += weighted_feature
        
        # Fusion and classification
        fused = self.fusion(weighted_sum)
        output = self.classifier(fused)
        
        return {
            'logits': output,
            'reliability_weights': reliability_weights,
            'modality_features': projected
        }