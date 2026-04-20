from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import gensim.downloader as gd

class FeatureContext:
    def __init__(self, labels_encoder, embedding_model):
        self.labels_encoder = labels_encoder
        self.embedding_model = embedding_model

    def one_hot_encode(self, edge_labels):
        num_labels = len(self.labels_encoder.categories_[0])
        edge_vec = torch.zeros(num_labels, dtype=torch.float32)

        for label in edge_labels:
            idx = self.labels_encoder.transform([[label]])[0].argmax()
            edge_vec[idx] = 1.0

        return edge_vec

    def get_word_embeddings(self, node):
        word = node.text.lower()
        dim = self.embedding_model.vector_size
        if word in self.embedding_model:
            return torch.tensor(self.embedding_model[word], dtype=torch.float32)
        return torch.zeros(dim, dtype=torch.float32)
    


def get_features(graphs,feature_extractor,embedding_model=gd.load("glove-wiki-gigaword-100")): 
    labels_encoder = fit_one_hot_encoding(graphs)
    context = FeatureContext(labels_encoder, embedding_model)
    return [get_features_from_graph(graph, feature_extractor, context) for graph in graphs]


def get_all_edge_labels(graphs):
    labels = []
    for graph in graphs:
        for node in graph.nodes:
            for label in node.incoming_edge_labels:
                labels.append(label)
    return labels


def fit_one_hot_encoding(graphs):
    labels_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    list_of_labels = get_all_edge_labels(graphs)
    labels_list_2d =  np.array(list_of_labels).reshape(-1,1)

    labels_encoder.fit(labels_list_2d)

    return labels_encoder


def get_features_from_graph(graph, feature_extractor, context):
    node_features = []

    for node in graph.nodes:
        feature_list = feature_extractor.featurize(node, graph, context)
        node_features.append(torch.cat(feature_list))

    return torch.stack(node_features)