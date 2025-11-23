import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image

class ImageModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224', fine_tune=False):
        super(ImageModel, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(pretrained_model_name)
        self.fine_tune = fine_tune
        
        # Freeze ViT parameters if not fine-tuning
        if not fine_tune:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Classifier head
        self.classifier = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.3)
    
    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def forward(self, pixel_values):
        if pixel_values is None:
            return torch.zeros(768), torch.zeros(2)
            
        # Get ViT embeddings
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return pooled_output, logits