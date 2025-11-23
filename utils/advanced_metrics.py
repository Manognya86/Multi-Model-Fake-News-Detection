import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix
import seaborn as sns

def calculate_additional_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate comprehensive evaluation metrics for fake news detection
    """
    metrics = {}
    
    try:
        # Basic classification metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
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
        
        # ROC AUC if scores are provided
        if y_scores is not None:
            try:
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_true)) > 1:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                else:
                    metrics['roc_auc'] = 0.5
            except:
                metrics['roc_auc'] = 0.5
        
        # Precision-Recall AUC
        if y_scores is not None:
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                metrics['pr_auc'] = auc(recall, precision)
            except:
                metrics['pr_auc'] = 0.5
        
        # Additional statistical metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        
        # Derived metrics
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics['true_positive_rate'] + metrics['true_negative_rate']) / 2
        
        # F2 score (emphasizes recall)
        beta = 2
        metrics['f2_score'] = (1 + beta**2) * (metrics['precision'] * metrics['recall']) / \
                             (beta**2 * metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # Matthews Correlation Coefficient
        metrics['mcc'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
        
        # Cohen's Kappa
        total = tn + fp + fn + tp
        po = (tp + tn) / total
        pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (total * total)
        metrics['cohens_kappa'] = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
        
    except Exception as e:
        print(f"Error calculating advanced metrics: {e}")
        # Return basic metrics if advanced calculation fails
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return metrics

def plot_precision_recall_curve(y_true, y_scores, title="Precision-Recall Curve"):
    """
    Plot precision-recall curve
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()
        
        return average_precision
    except Exception as e:
        print(f"Error plotting precision-recall curve: {e}")
        return 0.5

def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """
    Plot ROC curve
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        return roc_auc
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")
        return 0.5

def plot_confusion_matrix_advanced(y_true, y_pred, labels=['Fake', 'Real'], title="Confusion Matrix"):
    """
    Plot enhanced confusion matrix
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

def calculate_confidence_intervals(metrics, y_true, y_pred, n_bootstraps=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for metrics
    """
    try:
        from sklearn.utils import resample
        import scipy.stats as st
        
        bootstrapped_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        n_samples = len(y_true)
        
        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = resample(range(n_samples), replace=True, n_samples=n_samples)
            y_true_boot = [y_true[i] for i in indices]
            y_pred_boot = [y_pred[i] for i in indices]
            
            # Calculate metrics for bootstrap sample
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            bootstrapped_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            bootstrapped_metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
            bootstrapped_metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
            bootstrapped_metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        
        # Calculate confidence intervals
        confidence_intervals = {}
        alpha = (1 - confidence) / 2
        
        for metric_name, values in bootstrapped_metrics.items():
            lower = np.percentile(values, 100 * alpha)
            upper = np.percentile(values, 100 * (1 - alpha))
            confidence_intervals[metric_name] = {
                'mean': np.mean(values),
                'lower': lower,
                'upper': upper,
                'interval': f"{lower:.3f} - {upper:.3f}"
            }
        
        return confidence_intervals
    except Exception as e:
        print(f"Error calculating confidence intervals: {e}")
        return {}

def calculate_model_calibration(y_true, y_scores, n_bins=10):
    """
    Calculate model calibration metrics
    """
    try:
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_data = []
        
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            if np.sum(bin_mask) > 0:
                bin_mean_score = np.mean(y_scores[bin_mask])
                bin_actual_positive = np.mean(y_true[bin_mask])
                bin_count = np.sum(bin_mask)
                
                calibration_data.append({
                    'bin': bin_idx,
                    'mean_predicted': bin_mean_score,
                    'mean_actual': bin_actual_positive,
                    'count': bin_count,
                    'calibration_error': abs(bin_mean_score - bin_actual_positive)
                })
        
        # Calculate expected calibration error (ECE)
        ece = 0
        total_samples = len(y_scores)
        
        for data in calibration_data:
            ece += (data['count'] / total_samples) * data['calibration_error']
        
        return {
            'expected_calibration_error': ece,
            'calibration_data': calibration_data
        }
    except Exception as e:
        print(f"Error calculating model calibration: {e}")
        return {'expected_calibration_error': 0, 'calibration_data': []}

def generate_comprehensive_report(y_true, y_pred, y_scores=None, model_name="Model"):
    """
    Generate comprehensive evaluation report
    """
    report = {}
    
    # Basic metrics
    report['basic_metrics'] = calculate_additional_metrics(y_true, y_pred, y_scores)
    
    # Confidence intervals
    report['confidence_intervals'] = calculate_confidence_intervals(
        report['basic_metrics'], y_true, y_pred
    )
    
    # Model calibration if scores are available
    if y_scores is not None:
        report['calibration'] = calculate_model_calibration(y_true, y_scores)
    
    # Performance summary
    report['summary'] = {
        'model_name': model_name,
        'total_samples': len(y_true),
        'fake_ratio': np.mean(y_true == 0),
        'real_ratio': np.mean(y_true == 1),
        'best_metric': max(report['basic_metrics']['accuracy'], 
                          report['basic_metrics']['f1'],
                          report['basic_metrics'].get('roc_auc', 0))
    }
    
    return report

def print_detailed_report(report):
    """
    Print detailed evaluation report
    """
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 60)
    
    # Basic metrics
    metrics = report['basic_metrics']
    print("\n📊 BASIC METRICS:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        print(f"   PR AUC: {metrics['pr_auc']:.4f}")
    
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"   Matthews CC: {metrics['mcc']:.4f}")
    print(f"   Cohen's Kappa: {metrics['cohens_kappa']:.4f}")
    
    # Class-specific metrics
    print("\n🎯 CLASS-SPECIFIC METRICS:")
    print(f"   Fake News - Precision: {metrics['precision_fake']:.4f}, Recall: {metrics['recall_fake']:.4f}, F1: {metrics['f1_fake']:.4f}")
    print(f"   Real News - Precision: {metrics['precision_real']:.4f}, Recall: {metrics['recall_real']:.4f}, F1: {metrics['f1_real']:.4f}")
    
    # Confidence intervals
    if report['confidence_intervals']:
        print("\n📈 CONFIDENCE INTERVALS (95%):")
        for metric, ci in report['confidence_intervals'].items():
            print(f"   {metric.capitalize()}: {ci['interval']}")
    
    # Calibration
    if 'calibration' in report:
        print(f"\n⚖️  MODEL CALIBRATION:")
        print(f"   Expected Calibration Error: {report['calibration']['expected_calibration_error']:.4f}")
    
    # Summary
    summary = report['summary']
    print(f"\n📋 SUMMARY:")
    print(f"   Model: {summary['model_name']}")
    print(f"   Total Samples: {summary['total_samples']}")
    print(f"   Fake Ratio: {summary['fake_ratio']:.4f}")
    print(f"   Real Ratio: {summary['real_ratio']:.4f}")
    print(f"   Best Metric Score: {summary['best_metric']:.4f}")
    
    print("=" * 60)