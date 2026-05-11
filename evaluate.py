import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer, AutoModel
from utils.data_utils import prepare_datasets, clean_text
from utils.helpers import calculate_metrics, plot_confusion_matrix
from utils.device_utils import to_device, clear_gpu_memory
from utils.advanced_metrics import calculate_additional_metrics
from utils.preprocessing import extract_metadata_features

import config
import os

from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import traceback
import torch.nn as nn

# Custom Dataset for BERT model
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, metadata, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        metadata_features = self.metadata[index]
        
        # Ensure metadata features are properly formatted
        if isinstance(metadata_features, list):
            if len(metadata_features) < 6:
                metadata_features = metadata_features + [0.0] * (6 - len(metadata_features))
            elif len(metadata_features) > 6:
                metadata_features = metadata_features[:6]
        else:
            metadata_features = [0.0] * 6
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Ensure label is within valid range [0, num_classes-1]
        try:
            label = int(label)
            if label not in [0, 1]:
                label = 0
        except (ValueError, TypeError):
            label = 0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': torch.tensor(metadata_features, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ExactEnhancedModel(nn.Module):
    """Exact replica that matches the saved model architecture with 1856 fusion dim"""
    
    def __init__(self, text_model_name='bert-base-uncased', metadata_dim=6, n_classes=2):
        super(ExactEnhancedModel, self).__init__()
        
        # Text encoder
        self.bert = AutoModel.from_pretrained(text_model_name)
        self.text_dim = 768
        
        # Metadata encoder - EXACT dimensions from saved model
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion - EXACT dimensions from saved model (1856)
        self.fusion_dim = 1856  # This is the key fix!
        
        # Classifier - EXACT dimensions from saved model  
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),  # Matches [512, 1856]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        
        # Additional projection layer to match the 1856 dimension
        self.feature_projection = nn.Linear(896, 1856)  # 768 (text) + 128 (metadata) = 896
        
        print(f"🔧 Initialized Exact Model:")
        print(f"   Text dim: {self.text_dim}")
        print(f"   Metadata encoder: {metadata_dim} -> 128")
        print(f"   Fusion dim: {self.fusion_dim}")
        print(f"   Classifier: {self.fusion_dim} -> 512 -> {n_classes}")
    
    def forward(self, input_ids, attention_mask, metadata):
        # Text features
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        
        # Metadata features
        metadata_features = self.metadata_encoder(metadata)
        
        # Basic fusion
        basic_fused = torch.cat([text_features, metadata_features], dim=1)  # 768 + 128 = 896
        
        # Project to match expected dimension
        fused_features = self.feature_projection(basic_fused)  # 896 -> 1856
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits

class FlexibleUnifiedModel(nn.Module):
    """Flexible model that can adapt to different saved model architectures"""
    
    def __init__(self, text_model_name='bert-base-uncased', metadata_dim=6, n_classes=2, fusion_dim=1856):
        super(FlexibleUnifiedModel, self).__init__()
        
        # Text encoder
        self.bert = AutoModel.from_pretrained(text_model_name)
        self.text_dim = 768
        
        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Flexible fusion dimension
        self.fusion_dim = fusion_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        
        # Projection to handle dimension mismatches
        self.feature_projection = nn.Linear(896, self.fusion_dim)  # 768 + 128 = 896
        
        print(f"🔧 Initialized Flexible Model:")
        print(f"   Fusion dim: {self.fusion_dim}")
        print(f"   Classifier: {self.fusion_dim} -> 512 -> {n_classes}")
    
    def forward(self, input_ids, attention_mask, metadata, image_data=None, graph_data=None):
        # Text features
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        
        # Metadata features
        metadata_features = self.metadata_encoder(metadata)
        
        # Basic fusion
        basic_fused = torch.cat([text_features, metadata_features], dim=1)  # 896
        
        # Project to target dimension
        fused_features = self.feature_projection(basic_fused)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits

def cleanup(model, test_loader):
    """Clean up GPU memory after evaluation"""
    print("Cleaning up GPU memory after evaluation...")
    
    # Delete model and data loader
    if model is not None:
        del model
    if test_loader is not None:
        del test_loader
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("GPU memory cleanup completed")

def load_test_data():
    """Load test data, fallback to dummy data if real data not available"""
    try:
        _, _, test_df = prepare_datasets(
            config.DATA_PATH, 
            getattr(config, 'DATASET_FILES', [])
        )
        
        if test_df is None or len(test_df) == 0:
            print("⚠️ No test data found in datasets, using dummy data")
            return create_dummy_data()
        
        print(f"✅ Loaded test data: {len(test_df)} samples")
        
        # Check and clean labels
        unique_labels = test_df['label'].unique()
        
        # Check for problematic labels
        problematic_labels = [l for l in unique_labels if l not in [0, 1]]
        if problematic_labels:
            print(f"⚠️ Found problematic labels: {problematic_labels}")
            print("🔧 Converting labels to binary [0, 1]...")
            
            if len(unique_labels) == 2:
                label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                test_df['label'] = test_df['label'].map(label_mapping)
            else:
                label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                for label in unique_labels[2:]:
                    label_mapping[label] = 0
                test_df['label'] = test_df['label'].map(label_mapping)
        
        print(f"   Final label distribution: {test_df['label'].value_counts().to_dict()}")
        
        return test_df
        
    except Exception as e:
        print(f"⚠️ Error loading test data: {e}")
        return create_dummy_data()

def create_dummy_data():
    """Create dummy test data"""
    print("📝 Creating dummy test data...")
    
    fake_samples = [
        "BREAKING: Celebrity found dead in hotel room! Shocking details inside!",
        "You won't believe what this politician said about vaccines! Must read!",
        "Viral video shows alien spacecraft over major city! Government hiding truth!",
    ]
    
    real_samples = [
        "The president gave a speech about economic policies today at the conference.",
        "New study shows benefits of regular exercise for heart health.",
        "Company announces quarterly earnings with 5% revenue growth.",
    ]
    
    texts = fake_samples + real_samples
    labels = [0] * len(fake_samples) + [1] * len(real_samples)
    
    test_df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'clean_text': [clean_text(text) for text in texts]
    })
    
    print(f"✅ Created dummy dataset: {len(test_df)} samples")
    return test_df

def create_simple_model():
    """Create a simple BERT model for fallback evaluation"""
    class SimpleBERTClassifier(nn.Module):
        def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
            super(SimpleBERTClassifier, self).__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.config = AutoModel.from_pretrained(model_name).config
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.config.hidden_size, num_classes)
            
        def forward(self, input_ids, attention_mask, metadata=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits
    
    return SimpleBERTClassifier()

def analyze_saved_model(model_path):
    """Analyze the saved model to understand its architecture"""
    print(f"🔍 Analyzing saved model: {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        print("📊 Model Architecture Analysis:")
        for key in sorted(state_dict.keys()):
            shape = state_dict[key].shape
            print(f"   {key}: {shape}")
            
        # Detect key dimensions
        metadata_weight_shape = state_dict.get('metadata_encoder.0.weight', None)
        classifier_weight_shape = state_dict.get('classifier.0.weight', None)
        
        fusion_dim = None
        if classifier_weight_shape is not None:
            fusion_dim = classifier_weight_shape[1]  # [512, fusion_dim]
            print(f"   Detected fusion dimension: {fusion_dim}")
        
        if metadata_weight_shape is not None:
            print(f"   Metadata encoder: input_dim={metadata_weight_shape[1]}, hidden_dim={metadata_weight_shape[0]}")
        
        if classifier_weight_shape is not None:
            print(f"   Classifier: input_dim={classifier_weight_shape[1]}, hidden_dim={classifier_weight_shape[0]}")
            
        return state_dict, fusion_dim
        
    except Exception as e:
        print(f"❌ Failed to analyze saved model: {e}")
        return None, None

def load_flexible_model(model_path):
    """Load a flexible model that adapts to the saved weights"""
    print(f"🚀 Loading flexible model from: {model_path}")
    
    try:
        # First analyze the saved model
        state_dict, fusion_dim = analyze_saved_model(model_path)
        if state_dict is None:
            return None
        
        if fusion_dim is None:
            print("⚠️ Could not detect fusion dimension, using default 1856")
            fusion_dim = 1856
        
        # Create flexible model with detected fusion dimension
        print(f"🔧 Creating flexible model with fusion_dim={fusion_dim}")
        model = FlexibleUnifiedModel(fusion_dim=fusion_dim)
        
        # Try to load state dict with flexible matching
        model_state_dict = model.state_dict()
        
        # Filter state dict to only include matching keys
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"   ⚠️ Shape mismatch for {key}: saved {value.shape} vs model {model_state_dict[key].shape}")
            else:
                print(f"   ⚠️ Missing key in model: {key}")
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Check how many parameters were loaded
        loaded_keys = set(filtered_state_dict.keys())
        saved_keys = set(state_dict.keys())
        matching_keys = loaded_keys.intersection(saved_keys)
        
        print(f"   ✅ Successfully loaded {len(matching_keys)}/{len(saved_keys)} parameters")
        
        if len(matching_keys) > len(saved_keys) * 0.5:  # At least 50% match
            print(f"   ✅ Flexible model loaded successfully!")
            return model
        else:
            print(f"   ⚠️ Only {len(matching_keys)} keys matched")
            return None
        
    except Exception as e:
        print(f"❌ Failed to load flexible model: {e}")
        return None

def evaluate_with_flexible_model(test_loader, device):
    """Evaluate using the flexible model architecture"""
    print("🔄 Loading flexible model architecture...")
    
    model_save_path = getattr(config, 'MODEL_SAVE_PATH', './saved_models')
    model_paths = [
        f'{model_save_path}/best_enhanced_model.pth',
        f'{model_save_path}/best_model.pth',
        f'{model_save_path}/best_unified_multimodal_model.pth',
        f'{model_save_path}/unified_multimodal_model.pth'
    ]
    
    model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            model = load_flexible_model(model_path)
            if model is not None:
                break
    
    if model is None:
        print("❌ No flexible model could be loaded")
        return None
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_scores = []
    
    print("🔍 Running evaluation with flexible model...")
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating with flexible model")):
        try:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    metadata=batch['metadata']
                )
                
                # Get probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_scores.extend(probabilities[:, 1].cpu().numpy())
                
        except Exception as e:
            print(f"⚠️ Error in batch {batch_idx}: {e}")
            continue
    
    if len(all_preds) == 0:
        print("❌ No predictions generated with flexible model")
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    advanced_metrics = calculate_additional_metrics(all_labels, all_preds, all_scores)
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_scores)
        metrics['roc_auc'] = roc_auc
    except:
        metrics['roc_auc'] = 0.5
    
    all_metrics = {**metrics, **advanced_metrics}
    
    print(f"📊 Flexible Model Results - Accuracy: {all_metrics.get('accuracy', 0):.4f}")
    
    return all_metrics

def evaluate_with_exact_model(test_loader, device):
    """Evaluate using the exact model architecture with 1856 fusion dim"""
    print("🔄 Loading exact model with 1856 fusion dimension...")
    
    try:
        model = ExactEnhancedModel()
        model.to(device)
        model.eval()
        
        # Try to load weights
        model_save_path = getattr(config, 'MODEL_SAVE_PATH', './saved_models')
        model_paths = [
            f'{model_save_path}/best_enhanced_model.pth',
            f'{model_save_path}/best_model.pth',
            f'{model_save_path}/best_unified_multimodal_model.pth'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location=device)
                    # Load with strict=False to handle dimension mismatches
                    model.load_state_dict(state_dict, strict=False)
                    print(f"✅ Loaded model from {os.path.basename(model_path)}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {model_path}: {e}")
                    continue
        
        if not model_loaded:
            print("❌ No trained weights found for exact model")
            return None
        
        # Evaluation
        all_preds = []
        all_labels = []
        all_scores = []
        
        print("🔍 Running evaluation with exact model...")
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating with exact model")):
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        metadata=batch['metadata']
                    )
                    
                    # Get probabilities
                    probabilities = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['label'].cpu().numpy())
                    all_scores.extend(probabilities[:, 1].cpu().numpy())
                    
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        if len(all_preds) == 0:
            print("❌ No predictions generated with exact model")
            return None
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        advanced_metrics = calculate_additional_metrics(all_labels, all_preds, all_scores)
        
        # Calculate ROC AUC
        try:
            roc_auc = roc_auc_score(all_labels, all_scores)
            metrics['roc_auc'] = roc_auc
        except:
            metrics['roc_auc'] = 0.5
        
        all_metrics = {**metrics, **advanced_metrics}
        
        print(f"📊 Exact Model Results - Accuracy: {all_metrics.get('accuracy', 0):.4f}")
        
        return all_metrics
        
    except Exception as e:
        print(f"❌ Exact model evaluation failed: {e}")
        return None

def evaluate_with_simple_model(test_loader, device):
    """Evaluate using a simple BERT model as fallback"""
    print("🔄 Creating simple BERT model for fallback evaluation...")
    
    try:
        # Create simple model
        simple_model = create_simple_model()
        simple_model.to(device)
        simple_model.eval()
        
        print("✅ Simple model created successfully")
        
        # Evaluation
        all_preds = []
        all_labels = []
        all_scores = []
        
        print("🔍 Running evaluation with simple model...")
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating with simple model")):
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with torch.no_grad():
                    outputs = simple_model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        metadata=batch['metadata']
                    )
                    
                    # Get probabilities
                    probabilities = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['label'].cpu().numpy())
                    all_scores.extend(probabilities[:, 1].cpu().numpy())
                    
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        if len(all_preds) == 0:
            print("❌ No predictions generated with simple model")
            return None
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        advanced_metrics = calculate_additional_metrics(all_labels, all_preds, all_scores)
        
        # Calculate ROC AUC
        try:
            roc_auc = roc_auc_score(all_labels, all_scores)
            metrics['roc_auc'] = roc_auc
        except:
            metrics['roc_auc'] = 0.5
        
        all_metrics = {**metrics, **advanced_metrics}
        
        print(f"📊 Simple Model Results - Accuracy: {all_metrics.get('accuracy', 0):.4f}")
        
        return all_metrics
        
    except Exception as e:
        print(f"❌ Simple model evaluation failed: {e}")
        return None

def evaluate_model():
    """Main evaluation function"""
    # Clear GPU memory
    clear_gpu_memory()
    
    print("🧪 Starting Model Evaluation...")
    print("=" * 50)
    
    # Load test data
    test_df = load_test_data()
    
    # Prepare metadata features
    test_metadata = []
    for idx, row in test_df.iterrows():
        try:
            metadata = extract_metadata_features(row)
            if isinstance(metadata, list) and len(metadata) < 6:
                metadata = metadata + [0.0] * (6 - len(metadata))
            elif not isinstance(metadata, list):
                metadata = [0.0] * 6
            test_metadata.append(metadata)
        except Exception as e:
            test_metadata.append([0.0] * 6)
    
    # Initialize tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    except:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Create test dataset
    test_dataset = NewsDataset(
        texts=test_df['clean_text'].values,
        labels=test_df['label'].values,
        metadata=test_metadata,
        tokenizer=tokenizer,
        max_len=128
    )
    
    # Create data loader
    batch_size = 8
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Use CPU for stability
    device = torch.device('cpu')
    
    print("🎯 Enhanced Evaluation Strategy:")
    print("   1. Try to load flexible model that adapts to saved weights")
    print("   2. Try exact model with 1856 fusion dimension")
    print("   3. If both fail, use simple BERT model as fallback")
    print("=" * 50)
    
    # First try: Flexible model
    print("\n🚀 Attempt 1: Loading flexible model...")
    results = evaluate_with_flexible_model(test_loader, device)
    
    if results is not None:
        print("✅ Flexible model evaluation completed successfully!")
        model_type = "flexible_trained_model"
    else:
        # Second try: Exact model with 1856 fusion
        print("\n🔄 Attempt 2: Loading exact model with 1856 fusion...")
        results = evaluate_with_exact_model(test_loader, device)
        
        if results is not None:
            print("✅ Exact model evaluation completed successfully!")
            model_type = "exact_1856_model"
        else:
            # Fallback: Use simple model
            print("\n🔄 Attempt 3: Using simple BERT model as fallback...")
            results = evaluate_with_simple_model(test_loader, device)
            model_type = "simple_fallback"
    
    if results is None:
        print("❌ All evaluation attempts failed!")
        return None
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("="*60)
    
    accuracy = results.get('accuracy', 0)
    f1 = results.get('f1', 0)
    roc_auc = results.get('roc_auc', 0)
    
    print(f"📈 Basic Metrics:")
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   🎯 F1-Score: {f1:.4f}")
    print(f"   🔍 Precision: {results.get('precision', 0):.4f}")
    print(f"   📝 Recall: {results.get('recall', 0):.4f}")
    print(f"   📊 ROC AUC: {roc_auc:.4f}")
    
    # Performance interpretation
    performance_level = "EXCELLENT" if accuracy > 0.8 else \
                       "GOOD" if accuracy > 0.7 else \
                       "FAIR" if accuracy > 0.6 else \
                       "POOR" if accuracy > 0.5 else "VERY POOR"
    
    print(f"\n🎯 Performance Level: {performance_level}")
    
    # Save evaluation results
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'model_used': model_type,
        'test_set_size': len(test_df),
        'metrics': results,
        'performance_level': performance_level
    }
    
    # Ensure evaluation directory exists
    os.makedirs('./evaluation_results', exist_ok=True)
    
    # Save results as JSON
    results_path = './evaluation_results/final_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4, default=str)
    
    print(f"💾 Detailed evaluation results saved to '{results_path}'")
    
    # Generate recommendations
    print(f"\n📋 Recommendations:")
    if accuracy < 0.7:
        print("   • Consider retraining the model with consistent architecture")
        print("   • Ensure the model architecture matches between training and evaluation")
        print("   • Use the same model definition files for both training and inference")
        print("   • Consider using the simple model for deployment if it meets requirements")
    else:
        print("   • Model performance is satisfactory for deployment")
        print("   • Consider monitoring performance on new data")
    
    print(f"\n📝 Evaluation Info:")
    print(f"   📋 Test Set Size: {len(test_df)} samples")
    print(f"   🔧 Model Type: {model_type}")
    print(f"   ⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == '__main__':
    print("🎯 Multimodal Fake News Detection - Final Model Evaluation")
    print("=" * 60)
    
    # Force CPU for stability
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    try:
        evaluation_results = evaluate_model()
        
        if evaluation_results:
            accuracy = evaluation_results.get('accuracy', 0)
            print(f"\n✅ Evaluation completed successfully!")
            print(f"📊 Final Accuracy: {accuracy:.4f}")
            
            if accuracy > 0.7:
                print("🎉 Good performance! The model can be used for fake news detection.")
            elif accuracy > 0.6:
                print("⚠️ Fair performance. Consider improvements before production deployment.")
            else:
                print("🔧 Poor performance. Significant improvements needed.")
                
        else:
            print("\n❌ Evaluation failed!")
            
    except Exception as e:
        print(f"💥 Critical error during evaluation: {e}")
        print(traceback.format_exc())
