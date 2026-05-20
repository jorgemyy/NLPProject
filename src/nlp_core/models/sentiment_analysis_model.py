from torch import nn
import torch
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from kagglehub import KaggleDatasetAdapter
import kagglehub
from sklearn.metrics import f1_score, confusion_matrix

class SentimentAnalysis(nn.Module):
    def __init__(self, num_node_features, num_classes, out_dim, num_relations, compressed_embedding_dim, embedding_dim):
        super().__init__()
        num_node_features = num_node_features - embedding_dim + compressed_embedding_dim if compressed_embedding_dim != None else num_node_features
        self.conv1 = RGCNConv(num_node_features, out_dim, num_relations) 
        self.conv2 = RGCNConv(out_dim, out_dim, num_relations)
        self.conv3 = RGCNConv(out_dim, out_dim, num_relations)

        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.3)

        self.classifier = nn.Linear(out_dim * 2, num_classes)

        self.compress_embedding = nn.Linear(embedding_dim, compressed_embedding_dim) if compressed_embedding_dim != None else None
        self.embedding_dim = embedding_dim


    def forward(self, data):
        x, edge_index, batch, edge_type = data.x, data.edge_index, data.batch, data.edge_type

        if self.compress_embedding != None:
            emb_x = x[:, :self.embedding_dim]
            other_x = x[:, self.embedding_dim:]
            emb_x = self.compress_embedding(emb_x)
            x = torch.cat([emb_x, other_x], dim=-1)

        x = self.dropout(self.relu(self.bn1(self.conv1(x, edge_index, edge_type))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x, edge_index, edge_type))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x, edge_index, edge_type))))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)

        return self.classifier(x) 
    

class SentimentAnalysisModel():
    def __init__(self, batch_size, epochs, out_dim):
        self.model = None
        self.num_classes = 3
        self.batch_size = batch_size
        self.epochs = epochs
        self.out_dim = out_dim

    def build_model(self, num_node_features, num_relations, compressed_embedding_dim=None, embedding_dim=None):
        self.model = SentimentAnalysis(num_node_features, self.num_classes, self.out_dim, num_relations, compressed_embedding_dim, embedding_dim)

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
        
        accuracy = round((all_preds == all_labels).sum().item() / len(all_labels),4)
        fscore = round(f1_score(all_labels, all_preds, average='weighted'),4)

        cm = confusion_matrix(all_labels, all_preds).tolist()

        return accuracy, fscore, cm