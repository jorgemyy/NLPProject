from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class SentimentAnalysis(nn.Module):
    def __init__(self, num_node_features, num_classes, out_dim=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch=batch)

        return self.classifier(x)