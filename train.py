import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.gnn_transformer import GNNSAGETransformer
from models.utils import create_graph

# Hypothetical dataset loader
class DrivingDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories, labels):
        self.trajectories = trajectories
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]  # list of (lat, lon, speed, dir, time)
        graphs = [create_graph(segment) for segment in traj]  # segmented trajectories
        label = self.labels[idx]
        return graphs, label

# Parameters
in_feats = 4  # lat, lon, speed, dir
gnn_hidden = 64
transformer_hidden = 128
n_heads = 4
num_layers = 2
num_classes = 2
batch_size = 8
epochs = 20

# Instantiate model
model = GNNSAGETransformer(in_feats, gnn_hidden, transformer_hidden, n_heads, num_layers, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example DataLoader
train_dataset = DrivingDataset(train_trajs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for graphs, labels in train_loader:
        optimizer.zero_grad()
        logits = model(graphs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

print("Training complete!")
