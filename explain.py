import torch

from transformers import BertTokenizer
from models.multimodal_bert_model import UnifiedMultimodalModel
from utils.preprocessing import extract_metadata_features

import config
import os

def explain_prediction():
    print("🔍 Initializing model for explanation...")
    
    # Initialize the CORRECT model class
    model = UnifiedMultimodalModel(
        text_model_name=config.TEXT_MODEL_NAME,
        image_dim=config.IMAGE_EMBEDDING_DIM,
        metadata_dim=config.METADATA_DIM,
        n_classes=config.NUM_CLASSES,
        fusion_type='transformer'
    )
    
    # Load trained weights
    model_path = f'{config.MODEL_SAVE_PATH}/multimodal_model.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            print("✅ Loaded trained model weights")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️ Using randomly initialized weights")
            return
    else:
        print(f"❌ No trained model found at: {model_path}")
        print("⚠️ Please train the model first using: python train.py")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Example text to explain
    example_texts = [
        "BREAKING: Scientists discover miracle cure that reverses aging completely! Clinical trials show 100% success rate with no side effects.",
        "The National Weather Service has issued a winter storm warning for the northern region, expecting 6-8 inches of snow accumulation overnight.",
        "SHOCKING: This one simple trick helped me lose 30 pounds in one week without diet or exercise! Doctors hate this secret method.",
        "According to the latest economic report, unemployment rates have decreased by 0.3% this quarter, marking the fourth consecutive month of job growth."
    ]
    
    print("\n" + "="*60)
    print("🤖 FAKE NEWS DETECTION EXPLANATIONS")
    print("="*60)
    
    for i, example_text in enumerate(example_texts):
        print(f"\n📝 Example {i+1}:")
        print("-" * 50)
        print(f"Text: {example_text}")
        
        # Extract metadata features
        metadata = extract_metadata_features({
            'tweet_count': 10,
            'text_length': len(example_text)
        })
        
        # Tokenize text
        encoding = tokenizer.encode_plus(
            example_text,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(config.device)
        attention_mask = encoding['attention_mask'].to(config.device)
        metadata_tensor = torch.tensor(metadata, dtype=torch.float).unsqueeze(0).to(config.device)
        
        # Predict
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_features=None,
                metadata_features=metadata_tensor
            )
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        print(f"🎯 Prediction: {'FAKE NEWS' if prediction == 1 else 'REAL NEWS'}")
        print(f"📊 Confidence: {confidence:.4f}")
        print(f"🔴 Fake probability: {probabilities[0][1].item():.4f}")
        print(f"🟢 Real probability: {probabilities[0][0].item():.4f}")
        
        # Simple linguistic analysis
        words = example_text.lower().split()
        suspicious_indicators = {
            'sensational_words': ['breaking', 'shocking', 'miracle', 'secret', 'trick', 'unbelievable', 'astounding'],
            'urgency_words': ['urgent', 'immediately', 'now', 'instant', 'quick'],
            'certainty_words': ['100%', 'guaranteed', 'proven', 'scientific', 'doctors hate'],
            'normal_words': ['according', 'report', 'study', 'research', 'data', 'official']
        }
        
        print("\n🔍 Linguistic Analysis:")
        for category, word_list in suspicious_indicators.items():
            count = sum(1 for word in words if any(indicator in word for indicator in word_list))
            if count > 0:
                indicator = "⚠️" if category != 'normal_words' else "✅"
                print(f"   {indicator} {category}: {count}")
        
        # Overall assessment
        fake_indicators = sum(1 for word in words if any(indicator in word for indicator in 
                            suspicious_indicators['sensational_words'] + 
                            suspicious_indicators['urgency_words'] + 
                            suspicious_indicators['certainty_words']))
        
        if fake_indicators > 3:
            print("   🚨 HIGH fake news indicators detected!")
        elif fake_indicators > 1:
            print("   ⚠️  Moderate fake news indicators detected")
        else:
            print("   ✅ Low fake news indicators")
    
    print("\n" + "="*60)
    print("✅ Explanation completed!")
    return True

if __name__ == '__main__':
    explain_prediction()
