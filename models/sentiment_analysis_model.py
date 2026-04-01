from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class SentimentAnalysis(nn.Module):
    def __init__(self, num_node_features, num_classes, out_dim=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, out_dim)
        self.conv2 = GCNConv(out_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)