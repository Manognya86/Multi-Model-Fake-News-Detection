import unittest
import torch
from models.multimodal_bert_model import MultimodalFakeNewsDetector
from models.advanced_fusion import TransformerFusion, CrossAttentionFusion

class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.text_dim = 768
        self.metadata_dim = 2
        self.input_ids = torch.randint(0, 1000, (self.batch_size, 128))
        self.attention_mask = torch.ones((self.batch_size, 128))
        self.metadata = torch.randn((self.batch_size, self.metadata_dim))
    
    def test_multimodal_model(self):
        model = MultimodalFakeNewsDetector('bert-base-uncased', self.metadata_dim, 2)
        output = model(self.input_ids, self.attention_mask, self.metadata)
        self.assertEqual(output.shape, (self.batch_size, 2))
    
    def test_transformer_fusion(self):
        model = TransformerFusion(self.text_dim, self.metadata_dim)
        text_features = torch.randn((self.batch_size, self.text_dim))
        metadata_features = torch.randn((self.batch_size, self.metadata_dim))
        output = model(text_features, metadata_features)
        self.assertEqual(output.shape, (self.batch_size, 2))
    
    def test_cross_attention_fusion(self):
        model = CrossAttentionFusion(self.text_dim, self.metadata_dim)
        text_features = torch.randn((self.batch_size, self.text_dim))
        metadata_features = torch.randn((self.batch_size, self.metadata_dim))
        output = model(text_features, metadata_features)
        self.assertEqual(output.shape, (self.batch_size, 2))

if __name__ == '__main__':
    unittest.main()