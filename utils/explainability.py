import shap
import numpy as np
import torch

def explain_predictions(model, tokenizer, text, metadata):
    """
    Use SHAP to explain model predictions
    """
    # Define prediction function
    def predict_fn(texts):
        # Tokenize texts
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
        # Move to device
        device = next(model.parameters()).device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Create dummy metadata (same for all texts)
        batch_metadata = torch.tensor([metadata] * len(texts), dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, batch_metadata)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    # Create explainer
    explainer = shap.Explainer(predict_fn, tokenizer)
    
    # Calculate SHAP values
    shap_values = explainer([text])
    
    # Plot results
    shap.plots.text(shap_values)
    
    return shap_values