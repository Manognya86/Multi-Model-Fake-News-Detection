import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from PIL import Image
import numpy as np
from transformers import RobertaTokenizer
from utils.preprocessing import preprocess_text, extract_metadata_features
from models.image_model import ImageModel
import re

def preprocess_text(text):
    """Text preprocessing function"""
    if not text or pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class UnifiedMultimodalDataset(Dataset):
    def __init__(self, data_df, text_model_name='bert-base-uncased', 
                 max_length=256, include_images=False, include_graphs=False,
                 image_size=224, use_advanced_features=True):
        self.data = data_df.reset_index(drop=True)
        self.max_length = max_length
        self.include_images = include_images
        self.include_graphs = include_graphs
        self.image_size = image_size
        self.use_advanced_features = use_advanced_features
        
        # Initialize tokenizer
        self.text_tokenizer = RobertaTokenizer.from_pretrained(text_model_name)        
        # Initialize image model if needed
        self.image_model = None
        if self.include_images:
            try:
                self.image_model = ImageModel()
                for param in self.image_model.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"⚠️ Image model initialization failed: {e}")
                self.include_images = False
        
        # Initialize feature engineer for advanced features
        self.feature_engineer = None
        if self.use_advanced_features:
            try:
                from models.feature_engineer import EnhancedFeatureEngineer
                self.feature_engineer = EnhancedFeatureEngineer()
            except Exception as e:
                print(f"⚠️ Feature engineer initialization failed: {e}")
                self.use_advanced_features = False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Text processing
        raw_text = item.get('text', item.get('title', ''))
        if pd.isna(raw_text) or raw_text is None:
            text = ""
        else:
            text = str(raw_text)
        
        text = preprocess_text(text)
        
        # Tokenize text
        text_inputs = self.text_tokenizer(
            text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Metadata processing - ensure 6 features
        metadata_features = extract_metadata_features(item)
        if len(metadata_features) != 6:
            # Pad or truncate to 6 features
            metadata_features = np.pad(metadata_features, (0, max(0, 6 - len(metadata_features))))[:6]
        
        # Advanced features if enabled
        advanced_features = None
        if self.feature_engineer:
            try:
                advanced_features = self.feature_engineer.extract_enhanced_features(text, item)
            except Exception as e:
                print(f"Error extracting advanced features: {e}")
                advanced_features = {}
        
        # Image features
        image_features = None
        image_pixel_values = None
        if self.include_images and self.image_model:
            try:
                # Check for image path in data
                image_path = item.get('image_path', '')
                if image_path and os.path.exists(str(image_path)):
                    with torch.no_grad():
                        pixel_values = self.image_model.preprocess_image(image_path)
                        if pixel_values is not None:
                            image_features, _ = self.image_model(pixel_values.unsqueeze(0))
                            image_features = image_features.squeeze(0)
                            image_pixel_values = pixel_values
            except Exception as e:
                print(f"Error processing image: {e}")
                image_features = None
                image_pixel_values = None
        
        # Graph features (simplified without twitter_tree_parser)
        graph_data = None
        if self.include_graphs:
            try:
                # Create simple graph data structure
                graph_data = self.create_simple_graph_data(item, idx)
            except Exception as e:
                print(f"Error building graph: {e}")
                graph_data = None
        
        # Social context features
        social_context = None
        if 'social_engagement' in item:
            social_context = item['social_engagement']
        
        # Label
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'metadata': torch.tensor(metadata_features, dtype=torch.float),
            'advanced_features': advanced_features,
            'image_features': image_features,
            'image_pixel_values': image_pixel_values,
            'graph_data': graph_data,
            'social_context': social_context,
            'label': label,
            'text': text,
            'post_id': item.get('post_id', f'post_{idx}')
        }
    
    def create_simple_graph_data(self, item, idx):
        """Create simple graph data for testing"""
        try:
            # Create a simple graph with 3 nodes
            num_nodes = 3
            x = torch.randn(num_nodes, 64)  # Node features
            
            # Create edges: 0->1, 1->2
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
            
            # Create batch indices
            batch = torch.zeros(num_nodes, dtype=torch.long)
            
            graph_data = Data(x=x, edge_index=edge_index, batch=batch)
            return graph_data
        except Exception as e:
            print(f"Error in simple graph creation: {e}")
            return None

def collate_multimodal_batch(batch):
    """Enhanced collate function for unified multimodal data"""
    if not batch:
        return {}
    
    # Text data
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Metadata
    metadata = torch.stack([item['metadata'] for item in batch])
    
    # Labels
    labels = torch.stack([item['label'] for item in batch])
    
    # Image features
    image_features_list = []
    image_pixel_list = []
    for item in batch:
        if item['image_features'] is not None:
            image_features_list.append(item['image_features'])
        else:
            # Create zero features with correct dimension
            dummy_features = torch.zeros(768)  # ViT base features
            image_features_list.append(dummy_features)
        
        if item['image_pixel_values'] is not None:
            image_pixel_list.append(item['image_pixel_values'])
        else:
            image_pixel_list.append(torch.zeros(3, 224, 224))
    
    image_features = torch.stack(image_features_list)
    image_pixel_values = torch.stack(image_pixel_list)
    
    # Graph data - handle None cases
    graph_data_list = [item['graph_data'] for item in batch if item['graph_data'] is not None]
    if graph_data_list:
        try:
            graph_batch = Batch.from_data_list(graph_data_list)
        except Exception as e:
            print(f"Error batching graph data: {e}")
            graph_batch = None
    else:
        graph_batch = None
    
    # Advanced features
    advanced_features = [item.get('advanced_features', {}) for item in batch]
    
    # Social context
    social_contexts = [item.get('social_context', {}) for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'metadata': metadata,
        'image_features': image_features,
        'image_pixel_values': image_pixel_values,
        'graph_batch': graph_batch,
        'advanced_features': advanced_features,
        'social_contexts': social_contexts,
        'labels': labels,
        'texts': [item['text'] for item in batch],
        'post_ids': [item['post_id'] for item in batch]
    }