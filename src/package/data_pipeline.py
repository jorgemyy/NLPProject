import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data 
import gensim.downloader as gd

from package.features import featurizer 

class DataPipeline():
    def __init__(self, model, graph_builder, feature_extractor, embedding_model):
        self.model = model
        self.graph_builder = graph_builder
        self.feature_extractor = feature_extractor
        self.embedding_model = gd.load("glove-wiki-gigaword-100") if embedding_model == None else embedding_model
        

    def prepare(self, cap):
        df = self.model.get_data().head(cap) if cap else self.model.get_data()

        graphs, labels = self.graph_builder.build_graphs_from_df(df)
        features = featurizer.get_features(graphs,self.feature_extractor,self.embedding_model)

        data_objects = self.create_objects_for_gnn(graphs, features, labels)
        train_objects, test_objects = train_test_split(data_objects, test_size=0.2)

        return train_objects, test_objects, data_objects[0].num_node_features


    def create_objects_for_gnn(self, graphs, features, labels):
        data_objects = []
        for i, graph in enumerate(graphs):
            edge_index = torch.tensor(graph.get_edges_arr(), dtype=torch.long)

            x = features[i]
            y = labels[i]

            data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
            data_objects.append(data)
        
        return data_objects