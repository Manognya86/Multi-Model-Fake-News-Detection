import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def create_contrastive_pairs(dataset, num_pairs=1000):
    """
    Create pairs for contrastive learning
    """
    pairs = []
    labels = []
    
    # Get indices for each class
    class_indices = {0: [], 1: []}
    for i in range(len(dataset)):
        # Get the item from the dataset
        item = dataset[i]
        # Extract the label (it's a tensor, so we need to call .item() on it)
        label = item['label'].item()  # This converts tensor to Python scalar
        class_indices[label].append(i)
    
    # Create positive pairs (same class)
    for class_id, indices in class_indices.items():
        if len(indices) < 2:
            continue
            
        for _ in range(num_pairs // 2):
            i, j = np.random.choice(indices, 2, replace=False)
            pairs.append((i, j))
            labels.append(1)  # Positive pair
    
    # Create negative pairs (different classes)
    for _ in range(num_pairs // 2):
        if len(class_indices[0]) > 0 and len(class_indices[1]) > 0:
            class_0_idx = np.random.choice(class_indices[0], 1)[0]
            class_1_idx = np.random.choice(class_indices[1], 1)[0]
            pairs.append((class_0_idx, class_1_idx))
            labels.append(0)  # Negative pair
    
    return pairs, labels

class ContrastiveDataset(Dataset):
    def __init__(self, original_dataset, pairs, labels):
        self.original_dataset = original_dataset
        self.pairs = pairs
        self.labels = labels
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        label = self.labels[idx]
        
        item1 = self.original_dataset[i]
        item2 = self.original_dataset[j]
        
        return item1, item2, torch.tensor(label, dtype=torch.float)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for similarity learning
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        """
        Compute contrastive loss
        """
        # Euclidean distance between the two outputs
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Contrastive loss
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

def create_contrastive_training_pipeline(model, dataset, num_pairs=1000, batch_size=32):
    """
    Create complete contrastive training pipeline
    """
    # Create contrastive pairs
    pairs, labels = create_contrastive_pairs(dataset, num_pairs)
    
    # Create contrastive dataset
    contrastive_dataset = ContrastiveDataset(dataset, pairs, labels)
    
    # Create data loader
    contrastive_loader = torch.utils.data.DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return contrastive_loader

def train_contrastive_epoch(model, contrastive_loader, optimizer, criterion, device):
    """
    Train one epoch with contrastive learning
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (data1, data2, labels) in enumerate(contrastive_loader):
        # Move data to device
        data1 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data1.items()}
        data2 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data2.items()}
        labels = labels.to(device)
        
        # Get embeddings for both data points
        output1 = model.get_embeddings(data1)
        output2 = model.get_embeddings(data2)
        
        # Compute contrastive loss
        loss = criterion(output1, output2, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(contrastive_loader)
    return avg_loss

def compute_similarity_matrix(embeddings):
    """
    Compute cosine similarity matrix between embeddings
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    return similarity_matrix

def evaluate_contrastive_learning(model, dataset, device):
    """
    Evaluate contrastive learning performance
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            # Extract features and move to device
            features = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v 
                       for k, v in data.items() if k != 'label'}
            label = data['label']
            
            # Get embedding
            embedding = model.get_embeddings(features)
            all_embeddings.append(embedding.cpu())
            all_labels.append(label)
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(all_embeddings)
    
    # Evaluate clustering performance
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    
    try:
        # Convert to numpy for sklearn
        embeddings_np = all_embeddings.numpy()
        labels_np = all_labels.numpy()
        
        # Compute silhouette score
        silhouette = silhouette_score(embeddings_np, labels_np)
        
        # Compute clustering metrics
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        ari = adjusted_rand_score(labels_np, cluster_labels)
        
        return {
            'silhouette_score': silhouette,
            'adjusted_rand_index': ari,
            'embeddings': all_embeddings,
            'similarity_matrix': similarity_matrix
        }
    except Exception as e:
        print(f"Error in contrastive evaluation: {e}")
        return {
            'silhouette_score': 0,
            'adjusted_rand_index': 0,
            'embeddings': all_embeddings,
            'similarity_matrix': similarity_matrix
        }