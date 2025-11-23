import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def calculate_metrics(y_true, y_pred):
    """
    Calculate basic classification metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {}
    
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Class-specific metrics
        metrics['precision_fake'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['recall_fake'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['f1_fake'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        metrics['precision_real'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall_real'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_real'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'precision_fake': 0, 'recall_fake': 0, 'f1_fake': 0,
            'precision_real': 0, 'recall_real': 0, 'f1_real': 0
        }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=['Fake', 'Real'], title="Confusion Matrix"):
    """
    Plot confusion matrix
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.show()
        
        return cm
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        return None

def plot_training_history(history):
    """
    Plot training history
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Training Accuracy')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(history['train_f1'], label='Training F1')
        axes[1, 0].plot(history['val_f1'], label='Validation F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        if 'learning_rates' in history:
            axes[1, 1].plot(history['learning_rates'], label='Learning Rate', color='purple')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        return plt
    except Exception as e:
        print(f"Error plotting training history: {e}")
        return None

def save_model(model, filepath, optimizer=None, scheduler=None, epoch=None, metrics=None):
    """
    Save model checkpoint
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch if epoch is not None else 0,
            'metrics': metrics if metrics is not None else {}
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to: {filepath}")
        
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def load_model(model, filepath, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint
    """
    try:
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return None
        
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        result = {
            'model': model,
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {})
        }
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            result['optimizer'] = optimizer
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            result['scheduler'] = scheduler
        
        print(f"📥 Model loaded from: {filepath}")
        print(f"📅 Training epoch: {result['epoch']}")
        
        return result
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def save_training_results(history, test_metrics, filepath):
    """
    Save training results to JSON file
    """
    try:
        # Convert numpy values to Python types
        serializable_history = {}
        for key, values in history.items():
            serializable_history[key] = [float(v) if isinstance(v, (np.floating, float)) else v 
                                        for v in values]
        
        serializable_metrics = {}
        for key, value in test_metrics.items():
            if hasattr(value, 'item'):
                serializable_metrics[key] = value.item()
            elif isinstance(value, (np.floating, float)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                serializable_metrics[key] = int(value)
            else:
                serializable_metrics[key] = value
        
        results = {
            'training_history': serializable_history,
            'test_metrics': serializable_metrics,
            'best_epoch': np.argmax(history['val_f1']) if 'val_f1' in history else 0,
            'best_val_f1': max(history['val_f1']) if 'val_f1' in history else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"💾 Training results saved to: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving training results: {e}")
        return False

def load_training_results(filepath):
    """
    Load training results from JSON file
    """
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        print(f"📥 Training results loaded from: {filepath}")
        return results
    except Exception as e:
        print(f"❌ Error loading training results: {e}")
        return None

def print_training_summary(history, test_metrics):
    """
    Print training summary
    """
    print("=" * 60)
    print("🏁 TRAINING SUMMARY")
    print("=" * 60)
    
    if history and test_metrics:
        # Training stats
        best_epoch = np.argmax(history['val_f1'])
        best_val_f1 = max(history['val_f1'])
        
        print(f"📈 Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch + 1})")
        print(f"📊 Final Training Loss: {history['train_loss'][-1]:.4f}")
        print(f"📊 Final Validation Loss: {history['val_loss'][-1]:.4f}")
        
        # Test metrics
        print(f"🧪 Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"🎯 Test F1-Score: {test_metrics['f1']:.4f}")
        print(f"🔍 Test Precision: {test_metrics['precision']:.4f}")
        print(f"📝 Test Recall: {test_metrics['recall']:.4f}")
        
        if 'roc_auc' in test_metrics:
            print(f"📈 Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    
    print("=" * 60)

def create_prediction_analysis(y_true, y_pred, y_scores=None, texts=None):
    """
    Create detailed prediction analysis
    """
    analysis = {}
    
    # Confusion matrix details
    cm = confusion_matrix(y_true, y_pred)
    analysis['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    analysis['classification_report'] = report
    
    # Error analysis
    incorrect_indices = np.where(y_true != y_pred)[0]
    analysis['incorrect_predictions'] = len(incorrect_indices)
    analysis['error_rate'] = len(incorrect_indices) / len(y_true)
    
    # False positives and false negatives
    tn, fp, fn, tp = cm.ravel()
    analysis['false_positives'] = fp
    analysis['false_negatives'] = fn
    analysis['true_positives'] = tp
    analysis['true_negatives'] = tn
    
    # Add text samples for errors if available
    if texts is not None and len(incorrect_indices) > 0:
        error_samples = []
        for idx in incorrect_indices[:10]:  # First 10 errors
            error_samples.append({
                'text': texts[idx] if idx < len(texts) else f"Sample {idx}",
                'true_label': y_true[idx],
                'predicted_label': y_pred[idx],
                'confidence': y_scores[idx] if y_scores is not None else None
            })
        analysis['error_samples'] = error_samples
    
    return analysis

def export_predictions(y_true, y_pred, y_scores=None, texts=None, filepath='predictions.csv'):
    """
    Export predictions to CSV file
    """
    try:
        import pandas as pd
        
        data = {
            'true_label': y_true,
            'predicted_label': y_pred
        }
        
        if y_scores is not None:
            data['prediction_score'] = y_scores
        
        if texts is not None:
            data['text'] = texts
        
        df = pd.DataFrame(data)
        df['correct'] = df['true_label'] == df['predicted_label']
        
        df.to_csv(filepath, index=False)
        print(f"💾 Predictions exported to: {filepath}")
        
        return df
    except Exception as e:
        print(f"❌ Error exporting predictions: {e}")
        return None

def calculate_feature_importance(model, dataloader, device, feature_names=None):
    """
    Calculate feature importance using permutation importance
    """
    try:
        from sklearn.inspection import permutation_importance
        import numpy as np
        
        model.eval()
        all_features = []
        all_labels = []
        
        # Collect features and labels
        with torch.no_grad():
            for batch in dataloader:
                features = batch['metadata'].numpy()
                labels = batch['labels'].numpy()
                
                all_features.append(features)
                all_labels.append(labels)
        
        # Concatenate all batches
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Create a simple wrapper for the model
        def model_predict(X):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                # Assuming the model takes metadata as input
                outputs = model.forward_metadata_only(X_tensor)
                predictions = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                return predictions
        
        # Calculate permutation importance
        result = permutation_importance(
            model_predict, X, y, 
            n_repeats=10, 
            random_state=42
        )
        
        # Create feature importance dictionary
        importance_dict = {}
        for i, score in enumerate(result.importances_mean):
            feature_name = feature_names[i] if feature_names else f'feature_{i}'
            importance_dict[feature_name] = {
                'importance': score,
                'std': result.importances_std[i]
            }
        
        # Sort by importance
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda x: abs(x[1]['importance']), 
            reverse=True
        ))
        
        return sorted_importance
        
    except Exception as e:
        print(f"❌ Error calculating feature importance: {e}")
        return {}