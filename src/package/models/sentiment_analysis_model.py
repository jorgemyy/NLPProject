from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from kagglehub import KaggleDatasetAdapter
import kagglehub
import torch
from torch_geometric.loader import DataLoader

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
    def __init__(self, batch_size, epochs, out_dim):
        self.model = None
        self.num_classes = 3
        self.batch_size = batch_size
        self.epochs = epochs
        self.out_dim = out_dim

    def build_model(self, num_node_features):
        self.model = SentimentAnalysis(num_node_features, self.num_classes, self.out_dim)

    def get_data(self):
        full_df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "abhi8923shriv/sentiment-analysis-dataset",
            "train.csv",
            pandas_kwargs={"usecols": ["text", "sentiment"], "encoding": "latin1"}).sample(frac=1).dropna()
        
        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        full_df['label'] = full_df['sentiment'].replace(mapping).infer_objects(copy=False)
        
        return full_df
    
    def get_name(self):
        return "Sentiment Analysis"

    def train_model(self, train_data):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs+1):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out,batch.y.long())

                #back prop
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch + 1 == self.epochs:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")


    def eval_model(self, test_data):
        self.model.eval()
        loader = DataLoader(test_data, batch_size=self.batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                pred = self.model(batch).argmax(dim=1)
                all_preds.append(pred)
                all_labels.append(batch.y.long())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
        return accuracy