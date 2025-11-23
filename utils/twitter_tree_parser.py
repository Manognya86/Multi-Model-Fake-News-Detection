import os
import json
import torch
import networkx as nx
from torch_geometric.data import Data
import re
from collections import defaultdict

class TwitterTreeParser:
    """Parse Twitter15 and Twitter16 tree data to build propagation graphs"""
    
    def __init__(self):
        self.tree_cache = {}
    
    def parse_tree_file(self, tree_file_path):
        """
        Parse a tree file and extract propagation structure
        Returns: networkx graph and node features
        """
        try:
            with open(tree_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            # Parse the tree structure - format varies by dataset
            if '->' in content:
                return self._parse_arrow_format(content)
            else:
                return self._parse_default_format(content)
                
        except Exception as e:
            print(f"Error parsing tree file {tree_file_path}: {e}")
            return self._create_default_graph()
    
    def _parse_arrow_format(self, content):
        """Parse tree files with arrow notation (common in Twitter15/16)"""
        lines = content.split('\n')
        graph = nx.DiGraph()
        node_features = {}
        
        for line in lines:
            line = line.strip()
            if not line or '->' not in line:
                continue
            
            # Parse relationships like "user1 -> user2"
            parts = line.split('->')
            if len(parts) == 2:
                parent = parts[0].strip()
                child = parts[1].strip()
                
                # Add nodes and edge
                graph.add_node(parent)
                graph.add_node(child)
                graph.add_edge(parent, child)
                
                # Initialize node features if not present
                if parent not in node_features:
                    node_features[parent] = self._create_node_features(parent)
                if child not in node_features:
                    node_features[child] = self._create_node_features(child)
        
        return self._convert_to_pyg_data(graph, node_features)
    
    def _parse_default_format(self, content):
        """Parse other tree formats"""
        lines = content.split('\n')
        graph = nx.DiGraph()
        node_features = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to extract user mentions and relationships
            users = re.findall(r'@?\w+', line)
            if len(users) >= 2:
                # Assume first user is root or parent
                for i in range(1, len(users)):
                    parent = users[0]
                    child = users[i]
                    
                    graph.add_node(parent)
                    graph.add_node(child)
                    graph.add_edge(parent, child)
                    
                    if parent not in node_features:
                        node_features[parent] = self._create_node_features(parent)
                    if child not in node_features:
                        node_features[child] = self._create_node_features(child)
        
        # If no relationships found, create a simple chain
        if len(graph.nodes) == 0:
            return self._create_default_graph()
        
        return self._convert_to_pyg_data(graph, node_features)
    
    def _create_node_features(self, user_id):
        """Create feature vector for a user node"""
        # In a real system, you'd use actual user features
        # For now, create random but consistent features based on user_id hash
        import hashlib
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        
        features = [
            (hash_val % 100) / 100.0,  # User activity level
            ((hash_val // 100) % 100) / 100.0,  # Influence score
            ((hash_val // 10000) % 100) / 100.0,  # Engagement pattern
            len(user_id) / 20.0,  # Username length (normalized)
        ]
        
        # Pad to 64 dimensions
        while len(features) < 64:
            features.append((hash_val % 100) / 100.0)
            hash_val = hash_val // 100
        
        return torch.tensor(features[:64], dtype=torch.float)
    
    def _convert_to_pyg_data(self, graph, node_features):
        """Convert networkx graph to PyTorch Geometric Data object"""
        if len(graph.nodes) == 0:
            return self._create_default_graph()
        
        # Create node feature matrix
        nodes = list(graph.nodes())
        x = torch.stack([node_features[node] for node in nodes])
        
        # Create edge index
        edge_index = []
        for edge in graph.edges():
            source_idx = nodes.index(edge[0])
            target_idx = nodes.index(edge[1])
            edge_index.append([source_idx, target_idx])
        
        if not edge_index:
            # Add self-loops if no edges
            for i in range(len(nodes)):
                edge_index.append([i, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create edge attributes (propagation strength, time delay, etc.)
        edge_attr = torch.ones(edge_index.size(1), 2) * 0.5  # Default attributes
        
        # Create batch (all nodes belong to graph 0)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    def _create_default_graph(self):
        """Create default graph when tree parsing fails"""
        num_nodes = 10
        x = torch.randn(num_nodes, 64)
        
        # Create a simple chain
        edge_index = []
        for i in range(num_nodes - 1):
            edge_index.append([i, i + 1])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(edge_index.size(1), 2) * 0.5
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    def get_tree_for_sample(self, sample_id, dataset_name):
        """Get tree data for a specific sample"""
        cache_key = f"{dataset_name}_{sample_id}"
        
        if cache_key in self.tree_cache:
            return self.tree_cache[cache_key]
        
        # Determine tree file path based on dataset
        if dataset_name in ['twitter15', 'twitter16']:
            tree_file = f"sample_{sample_id}.txt"
            tree_path = os.path.join('data', dataset_name, 'tree', tree_file)
        else:
            # For other datasets, create synthetic tree
            return self._create_default_graph()
        
        if os.path.exists(tree_path):
            tree_data = self.parse_tree_file(tree_path)
            self.tree_cache[cache_key] = tree_data
            return tree_data
        else:
            default_graph = self._create_default_graph()
            self.tree_cache[cache_key] = default_graph
            return default_graph