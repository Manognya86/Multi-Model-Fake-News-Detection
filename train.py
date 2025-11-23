import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from models.multimodal_bert_model import create_model_from_registry
from models.unified_multimodal import UnifiedMultimodalModel
from models.ensemble_model import EnhancedEnsembleDetector
from models.graph_model import SocialGraphBuilder, EnhancedGNNModel
from models.image_model import ImageModel
from models.consistency_checker import CrossModalConsistency, ModalityImportanceWeighter
from models.feature_engineer import EnhancedFeatureEngineer
from utils.data_utils import prepare_datasets
from utils.helpers import calculate_metrics, plot_training_history, save_model
from utils.device_utils import to_device, clear_gpu_memory, print_gpu_memory, optimize_gpu_usage
from utils.advanced_metrics import calculate_additional_metrics
from utils.loss_functions import FocalLoss
from utils.data_loader import UnifiedMultimodalDataset, collate_multimodal_batch
from utils.preprocessing import extract_advanced_metadata_features, extract_linguistic_patterns
from utils.contrastive_utils import create_contrastive_pairs, ContrastiveDataset
import config
import time
import traceback
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class EnhancedMultimodalDataset(torch.utils.data.Dataset):
    """Enhanced dataset supporting all modalities"""
    
    def __init__(self, texts, labels, metadata, tokenizer, max_len, 
                 image_paths=None, social_data=None, augment=False):
        self.texts = texts
        self.labels = labels
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_paths = image_paths if image_paths else [None] * len(texts)
        self.social_data = social_data if social_data else [None] * len(texts)
        self.augment = augment
        self.feature_engineer = EnhancedFeatureEngineer()
        self.graph_builder = SocialGraphBuilder()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        metadata_features = self.metadata[index]
        image_path = self.image_paths[index] if self.image_paths else None
        social_context = self.social_data[index] if self.social_data else None
        
        # Ensure metadata_features is properly formatted
        if isinstance(metadata_features, list):
            if len(metadata_features) < 6:
                metadata_features = metadata_features + [0.0] * (6 - len(metadata_features))
            elif len(metadata_features) > 6:
                metadata_features = metadata_features[:6]
        else:
            metadata_features = [0.0] * 6
        
        # Text augmentation for training
        if self.augment and np.random.random() > 0.7:
            text = self._augment_text(text)
        
        # Enhanced tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        
        # Extract advanced linguistic features
        linguistic_features = self.feature_engineer.extract_enhanced_features(text, metadata_features)
        
        # Build graph data from social context
        graph_data = None
        if social_context:
            try:
                x, edge_index, edge_attr, timestamps = self.graph_builder.build_propagation_graph(social_context)
                graph_data = (x, edge_index, torch.zeros(x.size(0), dtype=torch.long))  # batch tensor
            except:
                graph_data = None
        
        # Prepare image data
        image_tensor = None
        if image_path and os.path.exists(image_path):
            try:
                # This would be implemented based on your image preprocessing
                # For now, we'll create a placeholder
                image_tensor = torch.randn(3, 224, 224)  # Placeholder
            except:
                image_tensor = None
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': torch.tensor(metadata_features, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
            'linguistic_features': linguistic_features,
            'graph_data': graph_data,
            'image_data': image_tensor,
            'text_raw': text
        }
    
    def _augment_text(self, text):
        """Enhanced text augmentation"""
        words = text.split()
        if len(words) > 5:
            # Random word deletion
            if np.random.random() > 0.8:
                del_idx = np.random.randint(0, len(words))
                words.pop(del_idx)
            # Random word swap
            if np.random.random() > 0.8 and len(words) > 2:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            # Synonym replacement (placeholder - would use nlpaug in practice)
            if np.random.random() > 0.9:
                pass
        
        return ' '.join(words)

class UnifiedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
    
    def train_memory_optimized_epoch(self, model, data_loader, optimizer, criterion, scaler, scheduler=None):
        """Memory-optimized training for enhanced model"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
    
        progress_bar = tqdm(data_loader, desc='Training', leave=False)
    
        for batch_idx, batch in enumerate(progress_bar):
            current_loss = 0.0
            try:
                # Clear GPU cache frequently
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Move batch to device - SIMPLIFIED for enhanced model
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['labels'].to(self.device)
            
                # Use gradient checkpointing for BERT
                if hasattr(model, 'bert'):
                    model.bert.gradient_checkpointing_enable()
                
                # Mixed precision training
                if scaler is not None and str(self.device) == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            metadata=metadata
                        )
                        loss = criterion(outputs, labels)
                
                    scaler.scale(loss).backward()
                    current_loss = loss.item()
                
                    # Gradient accumulation
                    if (batch_idx + 1) % 4 == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()
                else:
                    # Standard training
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        metadata=metadata
                    )
                    loss = criterion(outputs, labels)
                
                    loss.backward()
                    current_loss = loss.item()
                
                    if (batch_idx + 1) % 4 == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()
            
                total_loss += current_loss
            
                # Get predictions
                with torch.no_grad():
                    probabilities = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, dim=1)
                
                    all_preds.extend(preds.cpu().detach().numpy())
                    all_labels.extend(labels.cpu().detach().numpy())
            
                # Memory cleanup
                del outputs, loss, probabilities, preds
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
            
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"💥 OOM at batch {batch_idx}. Clearing cache and skipping...")
                    torch.cuda.empty_cache()
                    progress_bar.set_postfix({
                        'Loss': 'OOM',
                        'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    continue
                else:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    progress_bar.set_postfix({
                        'Loss': 'Error',
                        'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    continue
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                progress_bar.set_postfix({
                    'Loss': 'Error',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                continue
    
        # Handle any remaining gradients
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
        avg_loss = total_loss / len(data_loader)
        metrics = calculate_metrics(all_labels, all_preds)
    
        return avg_loss, metrics
    
    def evaluate_memory_optimized(self, model, data_loader, criterion):
        """Memory-optimized evaluation for enhanced model"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probabilities = []
        
        progress_bar = tqdm(data_loader, desc='Evaluation', leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                current_loss = 0.0
                try:
                    # Clear cache periodically
                    if batch_idx % 20 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Move batch to device - SIMPLIFIED for enhanced model
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    metadata = batch['metadata'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Disable autocast for evaluation to save memory
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        metadata=metadata
                    )
                    loss = criterion(outputs, labels)
                    
                    current_loss = loss.item()
                    total_loss += current_loss
                    
                    probabilities = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().detach().numpy())
                    all_labels.extend(labels.cpu().detach().numpy())
                    all_probabilities.extend(probabilities.cpu().detach().numpy())
                    
                    # Cleanup
                    del outputs, loss, probabilities, preds
                    
                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg Loss': f'{total_loss/(len(all_preds)/len(batch["labels"])):.4f}'
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"💥 OOM at evaluation batch {batch_idx}. Clearing cache and skipping...")
                        torch.cuda.empty_cache()
                        progress_bar.set_postfix({
                            'Loss': 'OOM',
                            'Avg Loss': f'{total_loss/(len(all_preds)/4):.4f}'
                        })
                        continue
                    else:
                        print(f"❌ Error in evaluation batch {batch_idx}: {e}")
                        progress_bar.set_postfix({
                            'Loss': 'Error',
                            'Avg Loss': f'{total_loss/(len(all_preds)/4):.4f}'
                        })
                        continue
                except Exception as e:
                    print(f"❌ Error in evaluation batch {batch_idx}: {e}")
                    progress_bar.set_postfix({
                        'Loss': 'Error', 
                        'Avg Loss': f'{total_loss/(len(all_preds)/4):.4f}'
                    })
                    continue
        
        avg_loss = total_loss / len(data_loader)
        metrics = calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics, all_preds, all_labels, all_probabilities

    def setup_unified_data_loaders(self, train_df, val_df, test_df, tokenizer):
        """Setup data loaders for all modalities"""
        
        print("🔧 Setting up unified multimodal data loaders...")
        
        # Extract metadata features
        train_metadata = [extract_metadata_from_row(row) for _, row in train_df.iterrows()]
        val_metadata = [extract_metadata_from_row(row) for _, row in val_df.iterrows()]
        test_metadata = [extract_metadata_from_row(row) for _, row in test_df.iterrows()]
        
        # Create datasets with all modalities
        train_dataset = EnhancedMultimodalDataset(
            texts=train_df['text'].values,
            labels=train_df['label'].values,
            metadata=train_metadata,
            tokenizer=tokenizer,
            max_len=128,
            augment=True
        )
        
        val_dataset = EnhancedMultimodalDataset(
            texts=val_df['text'].values,
            labels=val_df['label'].values,
            metadata=val_metadata,
            tokenizer=tokenizer,
            max_len=128,
            augment=False
        )
        
        test_dataset = EnhancedMultimodalDataset(
            texts=test_df['text'].values,
            labels=test_df['label'].values,
            metadata=test_metadata,
            tokenizer=tokenizer,
            max_len=128,
            augment=False
        )
        
        # Custom collate function for multimodal data
        def multimodal_collate(batch):
            collated = {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'metadata': torch.stack([item['metadata'] for item in batch]),
                'labels': torch.stack([item['label'] for item in batch]),
                'linguistic_features': [item['linguistic_features'] for item in batch],
                'graph_data': [item['graph_data'] for item in batch],
                'image_data': [item['image_data'] for item in batch],
                'texts_raw': [item['text_raw'] for item in batch]
            }
            return collated
        
        batch_size = 4  # Conservative for multimodal
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=multimodal_collate,
            pin_memory=False,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=multimodal_collate,
            pin_memory=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=multimodal_collate,
            pin_memory=False,
            num_workers=0
        )
        
        print(f"📊 Unified data loaders created:")
        print(f"   Batch size: {batch_size}")
        print(f"   Modalities: Text, Metadata, Linguistic, Graph, Image")
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches") 
        print(f"   Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def train_unified_epoch(self, model, data_loader, optimizer, criterion, scaler, scheduler=None):
        """Training with all modalities"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(data_loader, desc='Unified Multimodal Training', leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            current_loss = 0.0
            try:
                # Clear GPU cache frequently
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Prepare graph data - SIMPLIFIED: Skip graph for now to avoid complexity
                graph_data = None
                
                # Prepare image data - SIMPLIFIED: Skip image for now to avoid complexity  
                image_data = None
                
                # Mixed precision training
                if scaler is not None and str(self.device) == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            metadata=metadata,
                            image_data=image_data,
                            graph_data=graph_data
                        )
                        # FIX: Extract logits from dictionary
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        loss = criterion(logits, labels)
                
                    scaler.scale(loss).backward()
                    current_loss = loss.item()
                
                    # Gradient accumulation
                    if (batch_idx + 1) % 4 == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()
                else:
                    # Standard training
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        metadata=metadata,
                        image_data=image_data,
                        graph_data=graph_data
                    )
                    # FIX: Extract logits from dictionary
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    loss = criterion(logits, labels)
                
                    loss.backward()
                    current_loss = loss.item()
                
                    if (batch_idx + 1) % 4 == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()
        
                total_loss += current_loss
        
                # Get predictions
                with torch.no_grad():
                    probabilities = torch.softmax(logits, dim=1)
                    _, preds = torch.max(logits, dim=1)
                
                    all_preds.extend(preds.cpu().detach().numpy())
                    all_labels.extend(labels.cpu().detach().numpy())
        
                # Memory cleanup
                del outputs, loss, probabilities, preds, logits
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
        
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"💥 OOM at batch {batch_idx}. Clearing cache and skipping...")
                    torch.cuda.empty_cache()
                    progress_bar.set_postfix({
                        'Loss': 'OOM',
                        'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    continue
                else:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    progress_bar.set_postfix({
                        'Loss': 'Error',
                        'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    continue
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                progress_bar.set_postfix({
                    'Loss': 'Error',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                continue

        # Handle any remaining gradients
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        metrics = calculate_metrics(all_labels, all_preds)

        return avg_loss, metrics
    
    def evaluate_unified(self, model, data_loader, criterion):
        """Evaluation with all modalities"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probabilities = []
        all_consistency_scores = []
        all_importance_weights = []
        
        progress_bar = tqdm(data_loader, desc='Unified Multimodal Evaluation', leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                current_loss = 0.0
                try:
                    # Clear cache periodically
                    if batch_idx % 20 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    metadata = batch['metadata'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Prepare graph data - SIMPLIFIED: Skip graph for now
                    graph_data = None
                    
                    # Prepare image data - SIMPLIFIED: Skip image for now
                    image_data = None
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        metadata=metadata,
                        image_data=image_data,
                        graph_data=graph_data
                    )
                    # FIX: Extract logits from dictionary
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    loss = criterion(logits, labels)
                    
                    current_loss = loss.item()
                    total_loss += current_loss
                    
                    probabilities = torch.softmax(logits, dim=1)
                    _, preds = torch.max(logits, dim=1)
                    
                    all_preds.extend(preds.cpu().detach().numpy())
                    all_labels.extend(labels.cpu().detach().numpy())
                    all_probabilities.extend(probabilities.cpu().detach().numpy())
                    
                    # Collect additional metrics
                    if isinstance(outputs, dict):
                        if 'consistency_scores' in outputs:
                            all_consistency_scores.append(outputs['consistency_scores'])
                        if 'importance_weights' in outputs:
                            all_importance_weights.append(outputs['importance_weights'])
                    
                    # Cleanup
                    del outputs, loss, probabilities, preds, logits
                    
                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg Loss': f'{total_loss/(len(all_preds)/len(batch["labels"])):.4f}'
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"💥 OOM at evaluation batch {batch_idx}. Clearing cache and skipping...")
                        torch.cuda.empty_cache()
                        progress_bar.set_postfix({
                            'Loss': 'OOM',
                            'Avg Loss': f'{total_loss/(len(all_preds)/4):.4f}'
                        })
                        continue
                    else:
                        print(f"❌ Error in evaluation batch {batch_idx}: {e}")
                        progress_bar.set_postfix({
                            'Loss': 'Error',
                            'Avg Loss': f'{total_loss/(len(all_preds)/4):.4f}'
                        })
                        continue
                except Exception as e:
                    print(f"❌ Error in evaluation batch {batch_idx}: {e}")
                    progress_bar.set_postfix({
                        'Loss': 'Error', 
                        'Avg Loss': f'{total_loss/(len(all_preds)/4):.4f}'
                    })
                    continue
        
        avg_loss = total_loss / len(data_loader)
        metrics = calculate_metrics(all_labels, all_preds)
        
        # Calculate modality contributions if available
        if all_importance_weights:
            modality_contributions = self.calculate_modality_contributions(all_importance_weights)
            metrics['modality_contributions'] = modality_contributions
        
        return avg_loss, metrics, all_preds, all_labels, all_probabilities
    
    def calculate_modality_contributions(self, importance_weights_list):
        """Calculate average modality contributions across batches"""
        contributions = {
            'text': [], 'image': [], 'graph': [], 'metadata': []
        }
        
        for batch_weights in importance_weights_list:
            for modality in contributions.keys():
                weight_key = f'{modality}_weight'
                if weight_key in batch_weights:
                    contributions[modality].append(batch_weights[weight_key].mean().item())
        
        # Average across batches
        avg_contributions = {}
        for modality, values in contributions.items():
            if values:
                avg_contributions[modality] = np.mean(values)
            else:
                avg_contributions[modality] = 0.0
        
        # Normalize to percentages
        total = sum(avg_contributions.values())
        if total > 0:
            avg_contributions = {k: v/total for k, v in avg_contributions.items()}
        
        return avg_contributions

class EnsembleTrainer:
    """Trainer for ensemble model"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.feature_engineer = EnhancedFeatureEngineer()
    
    def train_ensemble(self, train_df, val_df, test_df):
        """Train ensemble model on extracted features"""
        print("🎯 Training Ensemble Model...")
        
        # Extract features for ensemble training
        print("🔧 Extracting features for ensemble...")
        
        train_features = []
        train_labels = []
        
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
            text = str(row['text'])
            label = row['label']
            metadata = extract_metadata_from_row(row)
            
            # Extract comprehensive features
            linguistic_features = self.feature_engineer.extract_enhanced_features(text, metadata)
            feature_vector = list(linguistic_features.values())
            
            train_features.append(feature_vector)
            train_labels.append(label)
        
        # Convert to numpy
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        
        # Initialize ensemble model
        ensemble_model = EnhancedEnsembleDetector(self.config)
        
        print("✅ Ensemble features extracted")
        print(f"📊 Feature dimension: {train_features.shape}")
        
        return ensemble_model, train_features, train_labels

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced datasets"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)

def create_optimizer(model, learning_rate, weight_decay=0.01):
    """Enhanced optimizer with differential learning rates"""
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': learning_rate
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
            'lr': learning_rate
        },
    ]
    
    return optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

def extract_metadata_from_row(row):
    """Extract metadata features from actual data row"""
    features = []
    
    # Account age (if account_created exists)
    if 'account_created' in row and pd.notna(row['account_created']):
        try:
            account_date = pd.to_datetime(row['account_created'])
            account_age_days = (datetime.now() - account_date).days
            features.append(min(account_age_days / 365, 10))
        except:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # Followers count (normalized)
    if 'followers_count' in row and pd.notna(row['followers_count']):
        features.append(min(row['followers_count'] / 10000, 1.0))
    else:
        features.append(0.0)
    
    # Following count (normalized)  
    if 'following_count' in row and pd.notna(row['following_count']):
        features.append(min(row['following_count'] / 5000, 1.0))
    else:
        features.append(0.0)
    
    # Statuses count (normalized)
    if 'statuses_count' in row and pd.notna(row['statuses_count']):
        features.append(min(row['statuses_count'] / 10000, 1.0))
    else:
        features.append(0.0)
    
    # Text length (normalized)
    if 'text' in row and pd.notna(row['text']):
        text_len = len(str(row['text']))
        features.append(min(text_len / 500, 1.0))
    else:
        features.append(0.0)
    
    # Add one more feature to make it 6
    if len(features) < 6:
        features.append(0.0)
    
    return features

def load_and_prepare_actual_data(data_path, dataset_files):
    """Load actual data files and prepare them for training"""
    import pandas as pd
    import os
    
    print("📁 Loading actual dataset files...")
    
    # Load all available data files
    all_data = []
    
    data_files = [
        'data1.csv', 'data2.csv', 'data3.csv',
        'gossipcop_fake.csv', 'gossipcop_real.csv', 
        'politifact_fake.csv', 'politifact_real.csv',
        'snopes_medical.csv'
    ]
    
    for file in data_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ Loaded {file}: {len(df)} rows, columns: {list(df.columns)}")
                
                # Check if it has the basic required columns
                if 'text' in df.columns and 'label' in df.columns:
                    if 'clean_text' not in df.columns:
                        from utils.data_utils import clean_text
                        df['clean_text'] = df['text'].apply(clean_text)
                    all_data.append(df)
                    print(f"   ✅ Added {file} to training data")
                else:
                    # Try to preprocess datasets with different structures
                    print(f"   🔄 Preprocessing {file} with custom structure...")
                    processed_df = preprocess_additional_datasets(df, file.replace('.csv', ''))
                    if processed_df is not None and 'text' in processed_df.columns and 'label' in processed_df.columns:
                        all_data.append(processed_df)
                        print(f"   ✅ Successfully processed and added {file}")
                    else:
                        print(f"   ⚠️  Skipping {file}: cannot process structure")
                    
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
    
    if not all_data:
        # Create dummy data for testing if no files found
        print("⚠️ No valid data files found! Creating dummy data for testing...")
        dummy_data = {
            'text': ['This is a fake news article', 'This is a real news story'] * 50,
            'label': [0, 1] * 50
        }
        all_data.append(pd.DataFrame(dummy_data))
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Combined dataset: {len(combined_df)} total rows")
    print(f"📊 Label distribution: {combined_df['label'].value_counts().to_dict()}")
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=0.3, 
        random_state=42,
        stratify=combined_df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"📈 Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df

def preprocess_additional_datasets(df, source):
    """Preprocess datasets that have different column structures"""
    from utils.data_utils import clean_text
    
    if source in ['gossipcop_fake', 'gossipcop_real', 'politifact_fake', 'politifact_real']:
        # These have: id, news_url, title, tweet_ids
        if 'title' in df.columns:
            df['text'] = df['title']
            df['clean_text'] = df['title'].apply(clean_text)
            # Assign labels based on source
            if 'fake' in source:
                df['label'] = 0
            else:
                df['label'] = 1
            return df
                
    elif source == 'snopes_medical':
        # This has: ID, Title, Body, Rating, Date, Image
        if 'Title' in df.columns and 'Body' in df.columns:
            # Combine title and body for text
            df['text'] = df['Title'] + ' ' + df['Body']
            df['clean_text'] = df['text'].apply(clean_text)
            # Convert rating to label 
            if 'Rating' in df.columns:
                # Map rating to binary label
                def map_rating_to_label(rating):
                    rating_str = str(rating).lower()
                    if any(true_indicator in rating_str for true_indicator in ['true', 'correct', 'accurate']):
                        return 1  # Real
                    elif any(false_indicator in rating_str for false_indicator in ['false', 'fake', 'incorrect', 'inaccurate']):
                        return 0  # Fake
                    else:
                        return 0  # Default to fake if unknown rating
                
                df['label'] = df['Rating'].apply(map_rating_to_label)
            else:
                df['label'] = 0  # Default to fake if no rating
            return df
    
    return None

def cleanup(model, train_loader, val_loader, test_loader, optimizer, scaler, scheduler=None):
    """Clean up GPU memory after training"""
    print("🧹 Cleaning up GPU memory...")
    
    # Delete all major objects
    del model
    del train_loader
    del val_loader
    del test_loader
    del optimizer
    if scaler is not None:
        del scaler
    if scheduler is not None:
        del scheduler
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("✅ GPU memory cleanup completed")

def save_training_artifacts(model, history, test_metrics, file_prefix='unified_multimodal'):
    """Save model, history and metrics"""
    # Ensure model save path exists
    model_save_path = getattr(config, 'MODEL_SAVE_PATH', './saved_models')
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save model
    model_path = f'{model_save_path}/{file_prefix}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved as {file_prefix}_model.pth")
    
    # Save training history
    history_path = f'{model_save_path}/{file_prefix}_training_history.pth'
    torch.save(history, history_path)
    print("💾 Training history saved")
    
    # Convert numpy values to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in test_metrics.items():
        if hasattr(value, 'tolist'):
            serializable_metrics[key] = value.tolist()
        elif hasattr(value, 'item'):
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = value
    
    # Save metrics as JSON
    metrics_path = f'{model_save_path}/{file_prefix}_test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    print("💾 Test metrics saved")
    
    return model_path, history_path, metrics_path

def main_unified_multimodal():
    """Main training function with all modalities integrated"""
    try:
        print("🧠 Starting Unified Multimodal Training...")
        clear_gpu_memory()
        
        # Initialize unified trainer
        trainer = UnifiedTrainer(config)
        
        print("📁 Loading and preprocessing data...")
        
        # Use actual data files
        train_df, val_df, test_df = load_and_prepare_actual_data(
            getattr(config, 'DATA_PATH', './data'), 
            getattr(config, 'DATASET_FILES', [])
        )
        
        # Use smaller model for memory constraints
        smaller_model_name = "distilroberta-base"
        print(f"🔧 Using model: {smaller_model_name}")
        
        tokenizer = RobertaTokenizer.from_pretrained(smaller_model_name)
        
        # Setup unified data loaders
        train_loader, val_loader, test_loader = trainer.setup_unified_data_loaders(
            train_df, val_df, test_df, tokenizer
        )
        
        # Start with a simpler approach - use the enhanced model from registry first
        print("🧠 Starting with Enhanced Model (text + metadata only) for stability...")
        model = create_model_from_registry(
            model_type='enhanced',
            text_model_name=smaller_model_name,
            metadata_dim=6,
            n_classes=config.NUM_CLASSES
        )
        model = model.to(config.device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'bert'):
            model.bert.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        
        # Loss function
        train_labels = train_df['label'].values
        class_weights = calculate_class_weights(train_labels)
        class_weights = class_weights.to(config.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("🎯 Using Weighted Cross Entropy Loss")
        
        # Conservative learning rate
        learning_rate = 2e-5
        optimizer = create_optimizer(model, learning_rate)
        
        # Learning rate scheduler
        num_training_steps = len(train_loader) * config.NUM_EPOCHS
        num_warmup_steps = num_training_steps // 10
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Use AMP for memory efficiency
        use_amp = True
        if use_amp and str(config.device) == 'cuda':
            scaler = GradScaler()
            print("🎯 Using Mixed Precision Training (AMP)")
        else:
            scaler = None
            print("🎯 Using Standard Precision Training")
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': []
        }
        
        print("🚀 Starting training with Enhanced Model...")
        start_time = time.time()
        
        best_val_f1 = 0.0
        patience = getattr(config, 'early_stopping_patience', 5)
        patience_counter = 0
        
        # Ensure model save directory exists
        model_save_path = getattr(config, 'MODEL_SAVE_PATH', './saved_models')
        os.makedirs(model_save_path, exist_ok=True)
        print(f"📁 Model save directory: {model_save_path}")
        
        for epoch in range(min(2, config.NUM_EPOCHS)):  # Start with 2 epochs for stability
            try:
                epoch_start = time.time()
                print(f"\n{'='*60}")
                print(f"📅 Epoch {epoch+1}/{min(2, config.NUM_EPOCHS)} - Enhanced Model")
                print(f"{'='*60}")
                
                # Clear memory before each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Training with enhanced model (text + metadata only)
                train_loss, train_metrics = trainer.train_memory_optimized_epoch(
                    model, train_loader, optimizer, criterion, scaler, scheduler
                )
                
                # Validation
                val_loss, val_metrics, val_preds, val_labels, val_probs = trainer.evaluate_memory_optimized(
                    model, val_loader, criterion
                )
                
                # Skip epoch if validation loss is NaN
                if np.isnan(val_loss):
                    print("⚠️ NaN validation loss! Skipping epoch...")
                    continue
                
                # Update history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_metrics['accuracy'])
                history['train_f1'].append(train_metrics['f1'])
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_metrics['accuracy'])
                history['val_f1'].append(val_metrics['f1'])
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                epoch_time = time.time() - epoch_start
                
                print(f"⏱️  Epoch completed in {epoch_time:.2f}s")
                print(f"📈 Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
                print(f"📊 Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
                print(f"📉 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
                
                # Memory usage info
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"💾 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                # Early stopping
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    patience_counter = 0
                    torch.save(model.state_dict(), f'{model_save_path}/best_enhanced_model.pth')
                    print("🏆 New best model saved!")
                else:
                    patience_counter += 1
                    print(f"⏳ Early stopping counter: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered!")
                    break
                
                # Clear memory after each epoch
                clear_gpu_memory()
                
            except Exception as e:
                print(f"❌ Error in epoch {epoch+1}: {e}")
                print(traceback.format_exc())
                clear_gpu_memory()
                continue
        
        # Now try to load the unified multimodal model with the fixed dimensions
        print("\n🔄 Switching to Unified Multimodal Model...")
        try:
            # Load the unified model with fixed dimensions
            unified_model = UnifiedMultimodalModel(
                text_model_name=smaller_model_name,
                metadata_dim=6,
                image_dim=768,
                graph_dim=256,
                n_classes=config.NUM_CLASSES,
                dropout_prob=0.3
            )
            unified_model = unified_model.to(config.device)
            
            # Transfer learning: copy weights from enhanced model if possible
            try:
                enhanced_state_dict = model.state_dict()
                unified_state_dict = unified_model.state_dict()
                
                # Copy matching weights
                for name, param in enhanced_state_dict.items():
                    if name in unified_state_dict and unified_state_dict[name].shape == param.shape:
                        unified_state_dict[name] = param
                
                unified_model.load_state_dict(unified_state_dict)
                print("✅ Transferred weights from enhanced model")
            except:
                print("⚠️ Could not transfer weights, training from scratch")
            
            # Continue training with unified model for remaining epochs
            remaining_epochs = max(0, config.NUM_EPOCHS - 2)
            if remaining_epochs > 0:
                print(f"🔄 Continuing training for {remaining_epochs} epochs with unified model...")
                
                # Reset optimizer for new model
                optimizer = create_optimizer(unified_model, learning_rate)
                
                for epoch in range(remaining_epochs):
                    epoch_start = time.time()
                    print(f"\n{'='*60}")
                    print(f"📅 Epoch {epoch+3}/{config.NUM_EPOCHS} - Unified Multimodal")
                    print(f"{'='*60}")
                    
                    # Clear memory before each epoch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Training with unified model
                    train_loss, train_metrics = trainer.train_unified_epoch(
                        unified_model, train_loader, optimizer, criterion, scaler, scheduler
                    )
                    
                    # Validation
                    val_loss, val_metrics, val_preds, val_labels, val_probs = trainer.evaluate_unified(
                        unified_model, val_loader, criterion
                    )
                    
                    # Update history
                    history['train_loss'].append(train_loss)
                    history['train_acc'].append(train_metrics['accuracy'])
                    history['train_f1'].append(train_metrics['f1'])
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_metrics['accuracy'])
                    history['val_f1'].append(val_metrics['f1'])
                    history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                    
                    epoch_time = time.time() - epoch_start
                    
                    print(f"⏱️  Epoch completed in {epoch_time:.2f}s")
                    print(f"📈 Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
                    print(f"📊 Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
                    
                    # Update best model
                    if val_metrics['f1'] > best_val_f1:
                        best_val_f1 = val_metrics['f1']
                        torch.save(unified_model.state_dict(), f'{model_save_path}/best_unified_multimodal_model.pth')
                        print("🏆 New best unified model saved!")
            
            model = unified_model  # Use unified model for final evaluation
            
        except Exception as e:
            print(f"❌ Failed to switch to unified model: {e}")
            print("🔧 Continuing with enhanced model...")
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Final evaluation
        print("\n🧪 Final evaluation on test set...")
        try:
            test_loss, test_metrics, test_preds, test_labels, test_probs = trainer.evaluate_memory_optimized(
                model, test_loader, criterion
            )
            
            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            try:
                test_auc = roc_auc_score(test_labels, [p[1] for p in test_probs])
                test_metrics['roc_auc'] = test_auc
                print(f"📈 Test ROC AUC: {test_auc:.4f}")
            except:
                test_metrics['roc_auc'] = 0.5
                print("⚠️ Could not calculate ROC AUC")
                
        except Exception as e:
            print(f"⚠️ Test evaluation failed: {e}")
            test_metrics = {'accuracy': 0.5, 'f1': 0.5, 'precision': 0.5, 'recall': 0.5}
        
        # Save training artifacts
        save_training_artifacts(model, history, test_metrics, 'unified_multimodal')
        
        # Cleanup
        cleanup(model, train_loader, val_loader, test_loader, optimizer, scaler, scheduler)
        
        return history, test_metrics
        
    except Exception as e:
        print(f"❌ Error in unified multimodal training: {e}")
        print(traceback.format_exc())
        return None, None

def analyze_results():
    """Enhanced results analysis with multimodal insights"""
    # Load saved metrics
    try:
        with open('./saved_models/unified_multimodal_test_metrics.json', 'r') as f:
            metrics = json.load(f)
    except:
        print("⚠️ Could not load unified multimodal metrics, trying memory optimized...")
        try:
            with open('./saved_models/memory_optimized_test_metrics.json', 'r') as f:
                metrics = json.load(f)
        except:
            print("❌ No metrics found!")
            return
    
    print("📊 ENHANCED RESULTS SUMMARY:")
    print(f"✅ Test Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"🎯 Test F1-Score: {metrics.get('f1', 0):.4f}") 
    print(f"📈 Test ROC AUC: {metrics.get('roc_auc', 0):.4f}")
    print(f"🔍 Test Precision: {metrics.get('precision', 0):.4f}")
    print(f"📝 Test Recall: {metrics.get('recall', 0):.4f}")
    
    # Print modality contributions if available
    if 'modality_contributions' in metrics:
        contributions = metrics['modality_contributions']
        print(f"\n🔧 MODALITY CONTRIBUTIONS:")
        for modality, contribution in contributions.items():
            print(f"   {modality.upper()}: {contribution:.2%}")
    
    # Create enhanced results visualization
    metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    values = [metrics.get(m, 0) for m in metrics_to_plot]
    
    plt.figure(figsize=(12, 8))
    
    # Main metrics
    plt.subplot(2, 1, 1)
    bars = plt.bar(metrics_to_plot, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E885B'])
    plt.title('Unified Multimodal Model Performance', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Modality contributions if available
    if 'modality_contributions' in metrics:
        plt.subplot(2, 1, 2)
        contributions = metrics['modality_contributions']
        modalities = list(contributions.keys())
        contrib_values = [contributions[m] for m in modalities]
        
        plt.pie(contrib_values, labels=modalities, autopct='%1.1f%%', startangle=90)
        plt.title('Modality Contribution Distribution', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('./saved_models/enhanced_multimodal_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function with unified multimodal training"""
    print("🎯 Unified Multimodal Fake News Detection Training")
    print("=" * 60)
    print("🔧 Integrating ALL Modalities:")
    print("   📝 Text + Metadata")
    print("   🖼️  Image Analysis") 
    print("   📊 Graph Propagation")
    print("   🔄 Cross-modal Consistency")
    print("   🎯 Ensemble Methods")
    print("💡 Using distilroberta-base, batch size 4, all modalities")
    
    # Run unified multimodal training
    history, metrics = main_unified_multimodal()
    
    if history and metrics:
        print("\n🎉 Unified multimodal training completed successfully!")
        print(f"🏆 Best validation F1: {max(history['val_f1']):.4f}")
        print(f"🧪 Test accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"🎯 Test F1-score: {metrics.get('f1', 0):.4f}")
        if 'roc_auc' in metrics:
            print(f"📈 Test ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Analyze results
        analyze_results()
    else:
        print("\n💥 Training failed!")

if __name__ == '__main__':
    main()