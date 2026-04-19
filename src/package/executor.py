import stanza
import amrlib
import gensim.downloader as gd
import torch
from tqdm import tqdm
import nltk
import penman
from torch_geometric.data import Data 
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from package import featurizer, graph_initializer

class Executor():
    def __init__(self, model, graph_type, feature_extractor, embedding_model = gd.load("glove-wiki-gigaword-100"), batch_size=16, epochs=200, hidden_layer_dim = 16):
        self.nlp = stanza.Pipeline('en') if graph_type == 'ud' else None
        self.stog = amrlib.load_stog_model() if graph_type == 'amr' else None

        self.graph_type = graph_type
        self.model = model
        self.feature_extractor = feature_extractor
        self.embedding_model = embedding_model

        self.batch_size=batch_size
        self.epochs = epochs
        self.hidden_layer_dim = hidden_layer_dim
        

    def run(self, cap=None):
        df = self.model.get_data().head(cap) if cap else self.model.get_data()
        data_objects = self.create_objects_for_gnn(df)
        train_objects, test_objects = train_test_split(data_objects, test_size=0.2)

        num_node_features = data_objects[0].num_node_features
        self.model.build_model(num_node_features, self.hidden_layer_dim)

        self.train_model(train_objects)
        acc = self.eval_model(test_objects)
        return acc


    def get_ud_data_from_df(self,df):
        labels = torch.tensor(list(df["label"]), dtype=torch.float32)
        graphs = []
        for text in tqdm(list(df["text"])):
            ud_text = self.nlp(text)
            ud_sentences = ud_text.sentences
            graphs.append(graph_initializer.make_and_merge_graphs(ud_sentences))
            
        return graphs, labels


    def get_amr_data_from_df(self,df):
        labels = torch.tensor(list(df["label"]), dtype=torch.float32)
        graphs = []
        for text in tqdm(list(df["text"])):
            sentences = nltk.sent_tokenize(text)
            amr_sentences = self.stog.parse_sents(sentences)
            amr_penman_sentences = [penman.decode(graph) for graph in amr_sentences]
            graphs.append(graph_initializer.make_and_merge_graphs(amr_penman_sentences))

        return graphs, labels


    def create_objects_for_gnn(self, df):
        graphs, labels = (self.get_ud_data_from_df(df) if self.graph_type == 'ud' else self.get_amr_data_from_df(df) if self.graph_type == 'amr' else ([],[]))

        data_objects = []

        features = featurizer.get_features(graphs,self.feature_extractor,self.embedding_model)
        for i, graph in enumerate(graphs):
            edge_index = torch.tensor(graph.get_edges_arr(), dtype=torch.long)

            x = features[i]
            y = labels[i]

            data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)

            data_objects.append(data)

        return data_objects
    

    def train_model(self, train_data):
        optimizer = torch.optim.Adam(self.model.model.parameters(), lr=0.01, weight_decay=5e-4)
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.model.train()
        for epoch in range(self.epochs+1):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = self.model.model(batch)
                loss = criterion(out,batch.y.long())

                #back prop
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch + 1 == self.epochs:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")


    def eval_model(self, test_data):
        self.model.model.eval()
        loader = DataLoader(test_data, batch_size=self.batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                pred = self.model.model(batch).argmax(dim=1)
                all_preds.append(pred)
                all_labels.append(batch.y.long())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
        return accuracy