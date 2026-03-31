from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import gensim.downloader as gd


def get_features(graphs, embedding_model=gd.load("glove-wiki-gigaword-50")):
    upos_encoder, xpos_encoder = fit_one_hot_encoding(graphs)   
    return [get_features_from_graph(graph, upos_encoder, xpos_encoder, embedding_model) for graph in graphs]


def get_pos_tags(graphs):
    upos = []
    xpos = []
    for graph in graphs:
        for node in graph.nodes:
            upos.append(node.upos)
            xpos.append(node.xpos)
    return upos, xpos 


def fit_one_hot_encoding(graphs):
    upos_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    xpos_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    upos_data, xpos_data = get_pos_tags(graphs)
    upos_data_2d, xpos_data_2d = np.array(upos_data).reshape(-1,1), np.array(xpos_data).reshape(-1,1)

    upos_encoder.fit(upos_data_2d)
    xpos_encoder.fit(xpos_data_2d)

    return upos_encoder, xpos_encoder


def one_hot_encode(node, upos_encoder, xpos_encoder):
    upos_vec = upos_encoder.transform([[node.upos]])[0]
    xpos_vec = xpos_encoder.transform([[node.xpos]])[0]

    return (torch.tensor(upos_vec, dtype=torch.float32),
    torch.tensor(xpos_vec, dtype=torch.float32))


def get_word_embeddings(node,embedding_model,dim=300):
    word = node.text.lower()
    if word in embedding_model:
        return torch.tensor(embedding_model[word], dtype=torch.float32)
    else:
        return torch.zeros(dim, dtype=torch.float32)
    

def get_features_from_graph(graph, upos_encoder, xpos_encoder, embedding_model):
    node_features = []

    for node in graph.nodes:
        normalized_id = node.id / len(graph.nodes)
        normalized_head = node.head / len(graph.nodes) if node.head != 0 else 0
        structural_features = torch.tensor([normalized_id, normalized_head], dtype=torch.float32)

        upos_vec, xpos_vec = one_hot_encode(node, upos_encoder, xpos_encoder)

        word_embedding = get_word_embeddings(node,embedding_model)
        
        feature_vec = torch.cat([structural_features, upos_vec, xpos_vec, word_embedding])
        node_features.append(feature_vec)

    return torch.stack(node_features)