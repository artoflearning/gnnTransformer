# Hybrid GNN-Transformer Model

import torch
import torch.nn as nn
from torch_geometric.nn import GraphSAGE
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GNNSAGETransformer(nn.Module):
    def __init__(self, in_feats, gnn_hidden, transformer_hidden, n_heads, num_layers, num_classes):
        super(GNNSAGETransformer, self).__init__()
        
        # GraphSAGE GNN for spatial embedding
        self.gnn = GraphSAGE(in_feats, gnn_hidden, num_layers=2)
        
        # Transformer for temporal sequence modeling
        encoder_layer = TransformerEncoderLayer(d_model=gnn_hidden, nhead=n_heads, dim_feedforward=transformer_hidden, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classifier
        self.fc = nn.Linear(gnn_hidden, num_classes)
    
    def forward(self, graph_list):
        # Graph_list: list of PyG graph data objects (batch_size, seq_len)
        batch_size, seq_len = len(graph_list), len(graph_list[0])

        # Extract GNN embeddings for each timestep graph
        embeddings = []
        for graphs in graph_list:
            timestep_embeds = []
            for g in graphs:
                node_embed = self.gnn(g.x, g.edge_index)
                graph_embed = torch.mean(node_embed, dim=0)  # mean pooling of node embeddings
                timestep_embeds.append(graph_embed)
            embeddings.append(torch.stack(timestep_embeds))

        embeddings = torch.stack(embeddings)  # (batch_size, seq_len, gnn_hidden)
        
        # Transformer modeling
        transformer_out = self.transformer(embeddings)  # (batch_size, seq_len, gnn_hidden)

        # Classify using embedding of last timestep (can use pooling alternatively)
        logits = self.fc(transformer_out[:, -1, :])
        return logits
