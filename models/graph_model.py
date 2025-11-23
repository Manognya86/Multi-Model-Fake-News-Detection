import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import torch_geometric
import numpy as np
from collections import defaultdict

class EnhancedGNNModel(nn.Module):
    def __init__(self, node_feature_dim=64, hidden_dim=256, num_heads=4, dropout=0.3):
        super(EnhancedGNNModel, self).__init__()
        
        # Graph attention layers with residual connections
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Temporal attention for propagation patterns
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Propagation pattern analyzer
        self.propagation_analyzer = PropagationAnalyzer()
        
        # Enhanced graph classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch, edge_attr=None, timestamps=None):
        # First graph convolutional layer with residual connection
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        
        # Second graph convolutional layer
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = self.dropout(x2)
        
        # Residual connection
        x_out = x2 + x1[:, :x2.size(1)]  # Ensure dimension matching
        
        # Global pooling (both mean and max)
        graph_embedding_mean = global_mean_pool(x_out, batch)
        graph_embedding_max = global_max_pool(x_out, batch)
        graph_embedding = torch.cat([graph_embedding_mean, graph_embedding_max], dim=1)
        
        # Apply temporal analysis if timestamps are available
        if timestamps is not None:
            temporal_features = self.analyze_propagation_patterns(x_out, batch, timestamps, edge_index)
            graph_embedding = torch.cat([graph_embedding, temporal_features], dim=1)
        
        # Classification
        logits = self.classifier(graph_embedding)
        
        return graph_embedding, logits
    
    def analyze_propagation_patterns(self, node_embeddings, batch, timestamps, edge_index):
        """Enhanced propagation pattern analysis"""
        batch_size = batch.max().item() + 1
        temporal_features = []
        
        for i in range(batch_size):
            # Get nodes for this graph
            graph_mask = (batch == i)
            graph_nodes = node_embeddings[graph_mask]
            
            if len(graph_nodes) > 1:
                # Calculate propagation metrics
                propagation_metrics = self.propagation_analyzer.analyze_graph(
                    graph_nodes, timestamps[graph_mask] if timestamps is not None else None,
                    edge_index, graph_mask
                )
                
                # Convert metrics to tensor
                metrics_tensor = torch.tensor(list(propagation_metrics.values()), 
                                            dtype=torch.float32, device=node_embeddings.device)
                temporal_features.append(metrics_tensor)
            else:
                # Default features for small graphs
                temporal_features.append(torch.zeros(6, device=node_embeddings.device))
        
        return torch.stack(temporal_features)

class TemporalGNNModel(nn.Module):
    """GNN model with enhanced temporal propagation analysis"""
    
    def __init__(self, node_feature_dim=64, hidden_dim=256, temporal_dim=32):
        super().__init__()
        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=4)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1)
        
        # Temporal propagation analysis
        self.temporal_encoder = nn.LSTM(hidden_dim, temporal_dim, batch_first=True)
        self.propagation_analyzer = EnhancedPropagationAnalyzer()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x, edge_index, batch, timestamps=None, edge_attr=None):
        # Graph processing
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        
        # Graph-level embedding
        graph_embedding = global_mean_pool(x, batch)
        
        # Temporal analysis if available
        if timestamps is not None:
            temporal_features = self.analyze_temporal_propagation(x, batch, timestamps, edge_index)
            combined_embedding = torch.cat([graph_embedding, temporal_features], dim=1)
        else:
            combined_embedding = graph_embedding
            
        logits = self.classifier(combined_embedding)
        return graph_embedding, logits
    
    def analyze_temporal_propagation(self, node_embeddings, batch, timestamps, edge_index):
        """Analyze temporal propagation patterns"""
        return self.propagation_analyzer.analyze_temporal_patterns(
            node_embeddings, batch, timestamps, edge_index
        )

class PropagationAnalyzer:
    """Analyze propagation patterns in social networks"""
    
    def analyze_graph(self, node_embeddings, timestamps, edge_index, node_mask):
        """Analyze various propagation metrics"""
        metrics = {}
        
        # Node embedding statistics
        metrics['embedding_variance'] = torch.var(node_embeddings, dim=0).mean().item()
        metrics['embedding_range'] = (node_embeddings.max() - node_embeddings.min()).item()
        
        # Graph structure metrics
        if edge_index is not None and node_mask is not None:
            # Calculate degree distribution
            degrees = self._calculate_degrees(edge_index, node_mask)
            metrics['degree_variance'] = torch.var(degrees).item() if len(degrees) > 0 else 0.0
            metrics['max_degree'] = degrees.max().item() if len(degrees) > 0 else 0.0
        
        # Temporal metrics if available
        if timestamps is not None and len(timestamps) > 1:
            temporal_metrics = self._analyze_temporal_patterns(timestamps)
            metrics.update(temporal_metrics)
        
        return metrics
    
    def _calculate_degrees(self, edge_index, node_mask):
        """Calculate node degrees for the subgraph"""
        # This is a simplified implementation
        # In practice, you'd want to extract the subgraph first
        node_indices = torch.where(node_mask)[0]
        if len(node_indices) == 0:
            return torch.tensor([])
        
        # Simple degree calculation (approximate)
        return torch.tensor([1.0] * len(node_indices))  # Placeholder
    
    def _analyze_temporal_patterns(self, timestamps):
        """Analyze temporal propagation patterns"""
        metrics = {}
        
        if len(timestamps) < 2:
            return metrics
        
        # Convert to seconds if needed
        if isinstance(timestamps[0], str):
            # This would require datetime conversion
            pass
        else:
            time_diffs = torch.diff(timestamps)
            if len(time_diffs) > 0:
                metrics['temporal_variance'] = torch.var(time_diffs).item()
                metrics['temporal_spread'] = (time_diffs.max() - time_diffs.min()).item()
                metrics['avg_time_diff'] = time_diffs.mean().item()
        
        return metrics

class EnhancedPropagationAnalyzer:
    """Enhanced propagation analysis with network science metrics"""
    
    def analyze_temporal_patterns(self, node_embeddings, batch, timestamps, edge_index):
        """Analyze temporal propagation patterns with enhanced metrics"""
        batch_size = batch.max().item() + 1
        temporal_features = []
        
        for i in range(batch_size):
            graph_mask = (batch == i)
            if torch.sum(graph_mask) < 2:
                temporal_features.append(torch.zeros(8))  # Default features
                continue
            
            metrics = self._calculate_propagation_metrics(
                node_embeddings[graph_mask],
                timestamps[graph_mask] if timestamps is not None else None,
                edge_index,
                graph_mask
            )
            
            feature_vector = torch.tensor(list(metrics.values()), dtype=torch.float32)
            temporal_features.append(feature_vector)
        
        return torch.stack(temporal_features) if temporal_features else torch.zeros(batch_size, 8)
    
    def _calculate_propagation_metrics(self, embeddings, timestamps, edge_index, mask):
        """Calculate comprehensive propagation metrics"""
        metrics = {}
        
        # Virality metrics
        metrics['virality_score'] = self._calculate_virality_score(embeddings)
        metrics['burstiness'] = self._calculate_burstiness(timestamps) if timestamps is not None else 0.5
        metrics['user_diversity'] = self._calculate_user_diversity(embeddings)
        metrics['echo_chamber_score'] = self._detect_echo_chambers(embeddings)
        
        # Structural metrics
        metrics['network_density'] = self._estimate_network_density(edge_index, mask)
        metrics['clustering_tendency'] = self._estimate_clustering(embeddings)
        metrics['influence_distribution'] = self._calculate_influence_distribution(embeddings)
        metrics['propagation_efficiency'] = self._estimate_propagation_efficiency(embeddings)
        
        return metrics
    
    def _calculate_virality_score(self, embeddings):
        """Calculate virality score based on embedding patterns"""
        if len(embeddings) < 2:
            return 0.5
        
        # Variance in embeddings can indicate diverse propagation
        variance = torch.var(embeddings, dim=0).mean()
        return min(1.0, variance.item() * 10)
    
    def _calculate_burstiness(self, timestamps):
        """Calculate temporal burstiness of propagation"""
        if len(timestamps) < 2:
            return 0.5
        
        intervals = torch.diff(timestamps)
        if intervals.std() == 0:
            return 0.5
        
        burstiness = (intervals.std() - intervals.mean()) / (intervals.std() + intervals.mean())
        return (burstiness + 1) / 2  # Normalize to [0, 1]
    
    def _calculate_user_diversity(self, embeddings):
        """Calculate diversity of users in propagation"""
        if len(embeddings) < 2:
            return 0.5
        
        # Use cosine similarity to measure diversity
        similarities = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        avg_similarity = similarities.mean()
        return 1.0 - avg_similarity  # Higher diversity = lower average similarity
    
    def _detect_echo_chambers(self, embeddings):
        """Detect echo chamber effects"""
        if len(embeddings) < 3:
            return 0.5
        
        # Calculate polarization in embedding space
        center = embeddings.mean(dim=0)
        distances = torch.norm(embeddings - center, dim=1)
        polarization = distances.std() / distances.mean() if distances.mean() > 0 else 0
        
        return min(1.0, polarization)
    
    def _estimate_network_density(self, edge_index, mask):
        """Estimate network density"""
        # Simplified estimation
        n_nodes = torch.sum(mask).item()
        if n_nodes < 2:
            return 0.1
        
        # Approximate density (this would be more accurate with actual edge data)
        return min(1.0, n_nodes / 100)
    
    def _estimate_clustering(self, embeddings):
        """Estimate clustering coefficient from embeddings"""
        if len(embeddings) < 3:
            return 0.3
        
        # Simplified clustering estimation
        similarities = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        avg_similarity = similarities.mean()
        return avg_similarity.item()
    
    def _calculate_influence_distribution(self, embeddings):
        """Calculate influence distribution inequality"""
        if len(embeddings) < 2:
            return 0.5
        
        # Use embedding norms as proxy for influence
        norms = torch.norm(embeddings, dim=1)
        gini = self._gini_coefficient(norms)
        return gini
    
    def _estimate_propagation_efficiency(self, embeddings):
        """Estimate propagation efficiency"""
        if len(embeddings) < 2:
            return 0.5
        
        # Distance from ideal propagation pattern
        center = embeddings.mean(dim=0)
        avg_distance = torch.norm(embeddings - center, dim=1).mean()
        efficiency = 1.0 / (1.0 + avg_distance.item())
        
        return efficiency
    
    def _gini_coefficient(self, values):
        """Calculate Gini coefficient for inequality measurement"""
        if len(values) == 0:
            return 0.5
        
        sorted_vals = torch.sort(values).values
        n = len(sorted_vals)
        index = torch.arange(1, n + 1, dtype=torch.float32, device=values.device)
        
        gini = ((torch.sum((2 * index - n - 1) * sorted_vals)) / 
                (n * torch.sum(sorted_vals)))
        
        return gini.item() if not torch.isnan(gini) else 0.5

class SocialGraphBuilder:
    """Build social network graphs from user interactions with enhanced features"""
    
    def __init__(self):
        self.user_features = {}
        self.propagation_analyzer = EnhancedPropagationAnalyzer()
    
    def build_propagation_graph(self, post_data):
        """
        Build a propagation graph for a news post with enhanced features
        """
        if post_data is None or 'user_interactions' not in post_data:
            return self._create_default_graph()
        
        interactions = post_data['user_interactions']
        
        if len(interactions) < 2:
            return self._create_default_graph()
        
        # Create nodes (users) and edges (interactions)
        node_features = []
        edge_index = []
        edge_attr = []
        node_timestamps = []
        
        # Add nodes for each user
        user_ids = list(set([interaction['user_id'] for interaction in interactions]))
        user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        
        for user_id in user_ids:
            features = self._get_enhanced_user_features(user_id, post_data)
            node_features.append(features)
            node_timestamps.append(self._get_user_timestamp(user_id, interactions))
        
        # Add edges based on interactions
        for interaction in interactions:
            source_user = user_map[interaction['user_id']]
            
            # If this is a retweet/reply, create edge to original poster
            if 'parent_user_id' in interaction and interaction['parent_user_id'] in user_map:
                target_user = user_map[interaction['parent_user_id']]
                edge_index.append([source_user, target_user])
                
                # Enhanced edge attributes
                edge_type = self._encode_interaction_type(interaction['type'])
                timestamp = interaction.get('timestamp', 0)
                engagement_strength = self._calculate_engagement_strength(interaction)
                
                edge_attr.append([edge_type, timestamp, engagement_strength])
        
        if len(edge_index) == 0:
            # If no edges, create a simple connected graph
            edge_index, edge_attr = self._create_connected_graph(user_ids, user_map)
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        timestamps = torch.tensor(node_timestamps, dtype=torch.float)
        
        return x, edge_index, edge_attr, timestamps
    
    def _get_enhanced_user_features(self, user_id, post_data):
        """Extract enhanced features for a user"""
        # In real implementation, this would use actual user data
        # For now, return features with some structure
        
        # Create deterministic but varied features based on user_id
        import hashlib
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        
        features = [
            (hash_val % 100) / 100.0,  # User activity level
            ((hash_val // 100) % 100) / 100.0,  # Influence score
            ((hash_val // 10000) % 100) / 100.0,  # Engagement pattern
            len(user_id) / 20.0,  # Username length (normalized)
            (hash_val % 10) / 10.0,  # Account age indicator
            ((hash_val // 10) % 10) / 10.0,  # Verification likelihood
        ]
        
        # Pad to 64 dimensions with some pattern
        while len(features) < 64:
            features.append((hash_val % 100) / 100.0)
            hash_val = hash_val // 100
        
        return features[:64]
    
    def _get_user_timestamp(self, user_id, interactions):
        """Get timestamp for user's first interaction"""
        user_interactions = [i for i in interactions if i['user_id'] == user_id]
        if user_interactions:
            return user_interactions[0].get('timestamp', 0)
        return 0
    
    def _encode_interaction_type(self, interaction_type):
        """Encode different types of social interactions with nuanced weights"""
        type_map = {
            'retweet': 1.0,      # Strong propagation signal
            'reply': 0.8,        # Engagement signal
            'quote': 0.9,        # Strong propagation with commentary
            'mention': 0.7,      # Moderate engagement
            'like': 0.3,         # Weak engagement
            'share': 0.95        # Very strong propagation
        }
        return type_map.get(interaction_type, 0.5)
    
    def _calculate_engagement_strength(self, interaction):
        """Calculate engagement strength based on interaction metadata"""
        strength = 0.5  # Base strength
        
        # Add based on interaction type
        type_strength = self._encode_interaction_type(interaction.get('type', ''))
        strength += (type_strength - 0.5) * 0.3
        
        # Consider additional engagement metrics
        if 'likes' in interaction:
            strength += min(interaction['likes'] / 100, 0.2)
        if 'replies' in interaction:
            strength += min(interaction['replies'] / 50, 0.1)
        
        return max(0, min(1, strength))
    
    def _create_connected_graph(self, user_ids, user_map):
        """Create a connected graph when no natural edges exist"""
        edge_index = []
        edge_attr = []
        
        # Create a simple path through all users
        for i in range(len(user_ids) - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])  # Undirected
            edge_attr.append([0.5, 0.0, 0.5])  # Default attributes
            edge_attr.append([0.5, 0.0, 0.5])
        
        return edge_index, edge_attr
    
    def _create_default_graph(self):
        """Create a default graph when no data is available"""
        num_nodes = 10
        x = torch.randn(num_nodes, 64)
        
        # Create a simple connected graph
        edge_index = []
        for i in range(num_nodes - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(edge_index.size(1), 3) * 0.5  # 3 attributes now
        timestamps = torch.zeros(num_nodes)
        
        return x, edge_index, edge_attr, timestamps