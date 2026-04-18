from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from kagglehub import KaggleDatasetAdapter
import kagglehub

class SentimentAnalysis(nn.Module):
    def __init__(self, num_node_features, num_classes, out_dim):
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
    

class SentimentAnalysisModel():
    def __init__(self):
        self.model = None
        self.num_classes = 3

    def build_model(self, num_node_features, out_dim=16):
        self.model = SentimentAnalysis(num_node_features, self.num_classes, out_dim)

    def get_data(self):
        full_df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "abhi8923shriv/sentiment-analysis-dataset",
            "train.csv",
            pandas_kwargs={"usecols": ["text", "sentiment"], "encoding": "latin1"}).sample(frac=1).dropna()
        
        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        full_df['label'] = full_df['sentiment'].replace(mapping)
        
        return full_df