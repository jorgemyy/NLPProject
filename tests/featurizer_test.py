import torch

from package import featurizer
from package import graph_initializer
from package.graph import Node, Graph

def test_get_word_embeddings(embedding_model):
    """check if a given word is embedded"""
    word = "born"
    node = Node(id = 0, text = word, root = 0)
    embedding = featurizer.get_word_embeddings(node,embedding_model)
    assert word in embedding_model
    assert type(embedding) == torch.Tensor
    assert len(embedding) == embedding_model.vector_size


def test_get_word_embeddings_word_not_in_vocab(embedding_model):
    """check if a given made up is all 0s"""
    word = "garblebleekster"
    node = Node(id = 0, text = word, root = 0)
    embedding = featurizer.get_word_embeddings(node,embedding_model)
    assert word not in embedding_model
    assert type(embedding) == torch.Tensor
    assert len(embedding) == embedding_model.vector_size
    assert all(v == 0 for v in embedding)


def test_get_features_from_ud_graph(ud_obama_graphs, embedding_model):
    """check if features are extracted from the graph"""
    assert type(ud_obama_graphs) == list
    labels_encoder = featurizer.fit_one_hot_encoding(ud_obama_graphs)
    test_graph = ud_obama_graphs[0]
    assert type(test_graph) == Graph
    features = featurizer.get_features_from_graph(test_graph,labels_encoder,embedding_model)
    assert isinstance(features, torch.Tensor)

    """check structural features"""
    first_word_id = test_graph.nodes[0].id
    num_words = len(test_graph.nodes)
    root = test_graph.nodes[0].root
    first_sentence_features = features[0]
    assert torch.isclose(first_sentence_features[0], torch.tensor(1 / num_words)) # normalized id
    assert torch.isclose(first_sentence_features[1], torch.tensor((first_word_id - root)/ num_words)) # distance from root

    """check shape of node features"""
    edge_labels = featurizer.get_all_edge_labels(ud_obama_graphs)
    num_total_edge_labels = len(set(edge_labels)) * 2 
    incoming_and_outgoing = 2
    num_structural_features = 2 # id, root
    num_node_features = num_structural_features + num_total_edge_labels + incoming_and_outgoing + embedding_model.vector_size 
    assert features.shape == (num_words,num_node_features)


def test_get_features_from_amr_graph(amr_hawaii_graph, embedding_model):
    """check if features are extracted from the graph"""
    assert type(amr_hawaii_graph) == list
    test_graph = amr_hawaii_graph[0]
    assert type(test_graph) == Graph

    labels_encoder = featurizer.fit_one_hot_encoding(amr_hawaii_graph)
    features = featurizer.get_features_from_graph(test_graph,labels_encoder,embedding_model)
    assert type(features) == torch.Tensor

    """check structural features"""
    first_word_id = test_graph.nodes[0].id
    num_words = len(test_graph.nodes)
    root = test_graph.nodes[0].root 
    first_sentence_features = features[0]
    assert torch.isclose(first_sentence_features[0], torch.tensor(1 / num_words)) # normalized id
    assert torch.isclose(first_sentence_features[1], torch.tensor((first_word_id - root) / num_words))

    """check shape of node features"""
    edge_labels = featurizer.get_all_edge_labels(amr_hawaii_graph)
    num_total_edge_labels = len(set(edge_labels)) * 2 
    incoming_and_outgoing = 2
    num_structural_features = 2 # id, root
    num_node_features = num_structural_features + num_total_edge_labels + incoming_and_outgoing + embedding_model.vector_size
    assert features.shape == (num_words,num_node_features)


def test_one_hot_encoding_on_gettysburg(amr_gettys_graphs):
    """check one hot encoding of parts of speech"""
    assert type(amr_gettys_graphs) == list
    assert type(amr_gettys_graphs[0]) == Graph
    edge_labels = featurizer.get_all_edge_labels(amr_gettys_graphs)
    num_edge_labels= len(set(edge_labels))
    labels_encoder = featurizer.fit_one_hot_encoding(amr_gettys_graphs)

    first_node_of_speech = amr_gettys_graphs[0].nodes[0]
    labels_vec = featurizer.one_hot_encode(first_node_of_speech, labels_encoder)
    incoming_edges_vec, outgoing_edges_vec = labels_vec[0], labels_vec[1]

    assert len(incoming_edges_vec) == num_edge_labels
    assert len(outgoing_edges_vec) == num_edge_labels
    assert len([v for v in incoming_edges_vec if v == 1]) == len(first_node_of_speech.incoming_edge_labels)
    assert len([v for v in outgoing_edges_vec if v == 1]) == len(first_node_of_speech.outgoing_edge_labels)


def test_featurizer_on_gettysburg_by_sentence(ud_gettys_graphs,embedding_model,ud_gettys_doc):
    """test featurizer on simple sentences, creating a graph for each sentence"""
    assert type(ud_gettys_graphs) == list
    assert type(ud_gettys_graphs[0]) == Graph
    
    features = featurizer.get_features(ud_gettys_graphs,embedding_model)

    edge_labels = featurizer.get_all_edge_labels(ud_gettys_graphs)
    num_total_edge_labels = len(set(edge_labels)) * 2 
    incoming_and_outgoing = 2
    num_normalized_features = 2 # id, root
    num_node_features = num_normalized_features + num_total_edge_labels + incoming_and_outgoing + embedding_model.vector_size
    num_nodes_in_first_graph = len(ud_gettys_doc.sentences[0].words)
    num_sentences = len(ud_gettys_doc.sentences)
    # shape = num_sentences, num_nodes, num_node_features, where num_nodes is inconstant
    assert features[0].shape == (num_nodes_in_first_graph, num_node_features)
    assert len(features) == num_sentences


def test_featurizer_on_gettysburg_using_merge(ud_gettys_doc,embedding_model):
    """test featurizer on simple sentences, then merge all graphs into a single graph"""
    gettys_merged_graph = graph_initializer.make_and_merge_graphs(ud_gettys_doc)
    assert type(gettys_merged_graph) == Graph
    features = featurizer.get_features([gettys_merged_graph],embedding_model)

    edge_labels = featurizer.get_all_edge_labels([gettys_merged_graph])
    num_total_edge_labels = len(set(edge_labels)) * 2 
    incoming_and_outgoing = 2
    num_normalized_features = 2 # id, root
    num_node_features = num_normalized_features + num_total_edge_labels + incoming_and_outgoing + embedding_model.vector_size
    num_nodes_in_doc = ud_gettys_doc.num_words
    assert len(features) == 1
    assert features[0].shape == (num_nodes_in_doc, num_node_features)

