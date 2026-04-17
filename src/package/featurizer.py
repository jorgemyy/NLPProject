from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import gensim.downloader as gd


def get_features(graphs, embedding_model=gd.load("glove-wiki-gigaword-100")):
    labels_encoder = fit_one_hot_encoding(graphs)   
    return [get_features_from_graph(graph, labels_encoder, embedding_model) for graph in graphs]


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


def one_hot_encode(node, labels_encoder):
    num_labels = len(labels_encoder.categories_[0])
    incoming_vec = torch.zeros(num_labels, dtype=torch.float32)
    outgoing_vec = torch.zeros(num_labels, dtype=torch.float32)

    for label in node.outgoing_edge_labels:
        idx = labels_encoder.transform([[label]])[0].argmax()
        outgoing_vec[idx] = 1.0

    for label in node.incoming_edge_labels:
        idx = labels_encoder.transform([[label]])[0].argmax()
        incoming_vec[idx] = 1.0

    return (incoming_vec, outgoing_vec)


def get_word_embeddings(node,embedding_model):
    word = node.text.lower()
    embedding_dim = embedding_model.vector_size
    if word in embedding_model:
        return torch.tensor(embedding_model[word], dtype=torch.float32)
    else:
        return torch.zeros(embedding_dim, dtype=torch.float32)
    

def get_features_from_graph(graph, labels_encoder, embedding_model):
    node_features = []

    for node in graph.nodes:
        normalized_id = node.id / len(graph.nodes)
        normalized_distance_from_root = (node.id - node.root) / len(graph.nodes)
        structural_features = torch.tensor([normalized_id, normalized_distance_from_root], dtype=torch.float32)

        incoming_edge_label_vecs, outgoing_edge_label_vecs = one_hot_encode(node, labels_encoder)
        num_incoming_edge_labels = sum(incoming_edge_label_vecs)
        num_outgoing_edge_labels = sum(outgoing_edge_label_vecs)
        num_labels_in_and_out = torch.tensor([num_incoming_edge_labels, num_outgoing_edge_labels], dtype=torch.float32)

        word_embedding = get_word_embeddings(node,embedding_model)
        
        feature_vec = torch.cat([structural_features, num_labels_in_and_out, incoming_edge_label_vecs, outgoing_edge_label_vecs, word_embedding])
        node_features.append(feature_vec)

    return torch.stack(node_features)