import torch
import torch.nn as nn
from transformers import BertModel, AutoConfig
import torch.nn.functional as F

class SimpleMultimodalFakeNewsDetector(nn.Module):
    """Simple multimodal model with basic fusion"""
    
    def __init__(self, text_model_name, metadata_dim, n_classes):
        super(SimpleMultimodalFakeNewsDetector, self).__init__()
        
        # BERT text encoder
        self.bert = BertModel.from_pretrained(text_model_name)
        self.bert_config = AutoConfig.from_pretrained(text_model_name)
        text_dim = self.bert_config.hidden_size
        
        # Metadata processing
        self.metadata_dim = metadata_dim
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion and classification
        fusion_dim = text_dim + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, input_ids, attention_mask, metadata):
        # Text features
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        
        # Metadata features
        metadata_features = self.metadata_encoder(metadata)
        
        # Fusion
        combined_features = torch.cat([text_features, metadata_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits

class EnhancedMultimodalFakeNewsDetector(nn.Module):
    """Enhanced multimodal model with attention and gating mechanisms"""
    
    def __init__(self, text_model_name, metadata_dim, n_classes, hidden_dim=512, dropout=0.3):
        super(EnhancedMultimodalFakeNewsDetector, self).__init__()
        
        # BERT text encoder
        self.bert = BertModel.from_pretrained(text_model_name)
        self.bert_config = AutoConfig.from_pretrained(text_model_name)
        text_dim = self.bert_config.hidden_size  # 768
        
        # Metadata processing
        self.metadata_dim = metadata_dim
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Enhanced fusion with attention - FIXED DIMENSIONS
        self.fusion_dim = text_dim + 64  # 768 + 64 = 832
        self.attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim, 
            num_heads=8,
            batch_first=True  # Add batch_first for compatibility
        )
        
        # Gating mechanism with PROPER DIMENSIONS - FIXED
        self.gate_text = nn.Linear(text_dim, self.fusion_dim)  # 768 -> 832
        self.gate_metadata = nn.Linear(64, self.fusion_dim)    # 64 -> 832
        self.gate_activation = nn.Sigmoid()
        
        # Advanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"🚀 Enhanced Model initialized")
        print(f"📊 Text dim: {text_dim}, Metadata dim: {metadata_dim}")
        print(f"🎯 Fusion dim: {self.fusion_dim}")
        
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, metadata):
        # Text features from BERT
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # [batch_size, 768]
        
        # Metadata features
        metadata_features = self.metadata_encoder(metadata)  # [batch_size, 64]
        
        # Concatenate features
        combined_features = torch.cat([text_features, metadata_features], dim=1)  # [batch_size, 832]
        
        # Enhanced fusion with self-attention
        # Reshape for attention: [batch_size, 1, fusion_dim]
        combined_features_attn = combined_features.unsqueeze(1)
        attended_features, _ = self.attention(
            combined_features_attn, 
            combined_features_attn, 
            combined_features_attn
        )
        combined_features = attended_features.squeeze(1)  # [batch_size, 832]
        
        # Adaptive gating mechanism - FIXED: Now dimensions match
        gate_text = self.gate_text(text_features)  # [batch_size, 832]
        gate_metadata = self.gate_metadata(metadata_features)  # [batch_size, 832]
        
        # Gating weights
        gate_weights = self.gate_activation(gate_text + gate_metadata)  # [batch_size, 832]
        
        # Apply gating - NOW DIMENSIONS MATCH: [batch_size, 832] * [batch_size, 832]
        gated_features = combined_features * gate_weights  # [batch_size, 832]
        
        # Final classification
        logits = self.classifier(gated_features)  # [batch_size, n_classes]
        
        return logits

class UnifiedMultimodalFakeNewsDetector(nn.Module):
    """Unified multimodal model with cross-modal attention"""
    
    def __init__(self, text_model_name, metadata_dim, n_classes):
        super(UnifiedMultimodalFakeNewsDetector, self).__init__()
        
        # BERT text encoder
        self.bert = BertModel.from_pretrained(text_model_name)
        self.bert_config = AutoConfig.from_pretrained(text_model_name)
        text_dim = self.bert_config.hidden_size
        
        # Metadata processing
        self.metadata_dim = metadata_dim
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, text_dim),  # Project to same dimension as text
            nn.ReLU()
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(text_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        
    def forward(self, input_ids, attention_mask, metadata):
        # Text features
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        
        # Metadata features projected to text dimension
        metadata_features = self.metadata_encoder(metadata)
        
        # Cross-modal attention
        text_features_attn = text_features.unsqueeze(1)
        metadata_features_attn = metadata_features.unsqueeze(1)
        
        # Attend metadata to text
        attended_features, _ = self.cross_attention(
            metadata_features_attn,
            text_features_attn,
            text_features_attn
        )
        
        # Combine features
        enhanced_metadata = attended_features.squeeze(1)
        combined_features = torch.cat([text_features, enhanced_metadata], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits

def create_model_from_registry(model_type, text_model_name, metadata_dim, n_classes):
    """
    Model registry for creating different model architectures
    """
    available_models = ['simple', 'enhanced', 'unified']
    
    if model_type == 'simple':
        model = SimpleMultimodalFakeNewsDetector(text_model_name, metadata_dim, n_classes)
        print("🎯 Creating simple model from registry")
        
    elif model_type == 'enhanced':
        model = EnhancedMultimodalFakeNewsDetector(text_model_name, metadata_dim, n_classes)
        print("🎯 Creating enhanced model from registry")
        
    elif model_type == 'unified':
        model = UnifiedMultimodalFakeNewsDetector(text_model_name, metadata_dim, n_classes)
        print("🎯 Creating unified model from registry")
        
    else:
        raise ValueError(f"Model type '{model_type}' not found. Available: {available_models}")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🚀 {model_type.capitalize()} Model initialized on: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model