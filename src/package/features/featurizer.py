from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np

class FeatureContext:
    def __init__(self, embedding_model, node_type_encoder):
        self.embedding_model = embedding_model
        self.node_type_encoder = node_type_encoder

    def get_word_embeddings(self,node):
        word = node.text.lower()
        dim = self.embedding_model.vector_size
        embedding = torch.zeros(dim, dtype=torch.float32)
        if word in self.embedding_model:
            embedding = torch.tensor(self.embedding_model[word], dtype=torch.float32)
        return embedding

    def one_hot_encode_node_type(self, label):
        return torch.tensor(self.node_type_encoder.transform([[label]])[0], dtype=torch.float32)

def fit_one_hot_encoding(list):
    labels_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    list_2d =  np.array(list).reshape(-1,1)
    labels_encoder.fit(list_2d)
    return labels_encoder


def get_features(graphs, feature_extractor, embedding_model): 
    node_type_encoder = None
    if "node_type" in feature_extractor.get_name():
        node_type_encoder = fit_one_hot_encoding([node.node_type for graph in graphs for node in graph.nodes])

    context = FeatureContext(embedding_model, node_type_encoder)
    return [get_features_from_graph(graph, feature_extractor, context) for graph in graphs]


def get_features_from_graph(graph, feature_extractor, context):
    node_features = []

    for node in graph.nodes:
        feature_dict = feature_extractor.featurize(node, graph, context)
        feature_list = []
        if "emb" in feature_dict:
            feature_list.append(feature_dict["emb"]) # embeddings MUST be first

        for feature in feature_dict:
            if feature != "emb":
                feature_list.append(feature_dict[feature])
        
        node_features.append(torch.cat(feature_list))

    return torch.stack(node_features)