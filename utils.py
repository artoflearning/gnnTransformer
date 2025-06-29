%python
%pip install torch torchvision torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv torchmetrics

"""
project/
├── data/
│   ├── trajectories.csv
│   └── labels.csv
├── models/
│   ├── gnn_transformer.py (Hybrid Model)
│   └── utils.py (Graph processing utilities)
└── train.py (Training Script)
"""

import torch
from torch_geometric.data import Data

def create_graph(traj):
    # traj: [(lat, lon, speed, dir, time), ...]
    nodes = torch.tensor([point[:4] for point in traj], dtype=torch.float)

    # Create edges connecting sequential points
    edges = [(i, i+1) for i in range(len(traj)-1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=nodes, edge_index=edge_index)
