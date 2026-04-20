import torch

from package.features.featurizer import FeatureContext
from package.features import featurizer
from package.graphs.graph import Node
from package.features.featurizer_decorator import *
from package.graphs.graph_builder_factory import GraphBuilderFactory



def test_get_word_embeddings(embedding_model):
    """check if a given word is embedded"""
    word = "born"
    node = Node(id = 0, text = word, root = 0)
    featurizer_context = FeatureContext(labels_encoder=None, embedding_model=embedding_model)
    embedding = featurizer_context.get_word_embeddings(node)
    assert word in embedding_model
    assert type(embedding) == torch.Tensor
    assert len(embedding) == embedding_model.vector_size


def test_get_word_embeddings_word_not_in_vocab(embedding_model):
    """check if a given made up is all 0s"""
    word = "garblebleekster"
    node = Node(id = 0, text = word, root = 0)
    featurizer_context = FeatureContext(labels_encoder=None, embedding_model=embedding_model)
    embedding = featurizer_context.get_word_embeddings(node)
    assert word not in embedding_model
    assert type(embedding) == torch.Tensor
    assert len(embedding) == embedding_model.vector_size
    assert all(v == 0 for v in embedding)


def test_get_features_from_ud_graph_all_features(obama_sentence, nlp, embedding_model, full_feature_extractor):
    """check if features are extracted from the graph"""
    graph_builder_factory = GraphBuilderFactory()
    ud_graph_builder = graph_builder_factory.create_UD_Builder(nlp)
    ud_obama_graphs = ud_graph_builder.make_graphs(obama_sentence)
    test_graph = ud_obama_graphs[0]

    labels_encoder = featurizer.fit_one_hot_encoding(ud_obama_graphs)
    featurizer_context = FeatureContext(labels_encoder=labels_encoder, embedding_model=embedding_model)

    features = featurizer.get_features_from_graph(test_graph,full_feature_extractor,featurizer_context)
    assert type(features) == torch.Tensor

    """check structural features"""
    first_word_id = test_graph.nodes[0].id
    num_words = len(test_graph.nodes)
    root = test_graph.root
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


def test_featurizer_decorator():
    feature_extractor = (FeatureExtractorBuilder()
                         .add_incoming_labels()
                         .add_outgoing_labels()
                         .build())

    """check get_name"""
    to_string = feature_extractor.get_name()
    assert "in_labels" in to_string
    assert "embedding" not in to_string


def test_get_features_from_amr_graph_all_features(obama_sentence, stog, embedding_model, full_feature_extractor):
    """check if features are extracted from the graph"""
    graph_builder_factory = GraphBuilderFactory()
    amr_graph_builder = graph_builder_factory.create_AMR_Builder(stog)
    amr_obama_graphs = amr_graph_builder.make_graphs(obama_sentence)
    test_graph = amr_obama_graphs[0]

    labels_encoder = featurizer.fit_one_hot_encoding(amr_obama_graphs)
    featurizer_context = FeatureContext(labels_encoder=labels_encoder, embedding_model=embedding_model)
    
    features = featurizer.get_features_from_graph(test_graph,full_feature_extractor,featurizer_context)
    assert type(features) == torch.Tensor

    """check structural features"""
    first_word_id = test_graph.nodes[0].id
    num_words = len(test_graph.nodes)
    root = test_graph.root 
    first_sentence_features = features[0]
    assert torch.isclose(first_sentence_features[0], torch.tensor(1 / num_words)) # normalized id
    assert torch.isclose(first_sentence_features[1], torch.tensor((first_word_id - root) / num_words))

    """check shape of node features"""
    edge_labels = featurizer.get_all_edge_labels(amr_obama_graphs)
    num_total_edge_labels = len(set(edge_labels)) * 2 
    incoming_and_outgoing = 2
    num_structural_features = 2 # id, root
    num_node_features = num_structural_features + num_total_edge_labels + incoming_and_outgoing + embedding_model.vector_size
    assert features.shape == (num_words,num_node_features)


def test_get_features_from_amr_graph_only_edge_features(obama_sentence, stog, embedding_model):
    graph_builder_factory = GraphBuilderFactory()
    amr_graph_builder = graph_builder_factory.create_AMR_Builder(stog)
    amr_obama_graphs = amr_graph_builder.make_graphs(obama_sentence)
    test_graph = amr_obama_graphs[0]
    
    labels_encoder = featurizer.fit_one_hot_encoding(amr_obama_graphs)
    featurizer_context = FeatureContext(labels_encoder=labels_encoder, embedding_model=embedding_model)
    feature_extractor = (FeatureExtractorBuilder()
                         .add_incoming_labels()
                         .add_outgoing_labels()
                         .build())
    
    features = featurizer.get_features_from_graph(test_graph,feature_extractor,featurizer_context)

    """check edge features"""
    num_words = len(test_graph.nodes)
    edge_labels = featurizer.get_all_edge_labels(amr_obama_graphs)
    first_sentence_features = features[0]
    num_edge_labels = len(set(edge_labels))
    incoming_edge_multi_hot = first_sentence_features[:num_edge_labels]
    incoming_edge_num = first_sentence_features[num_edge_labels]
    assert sum(incoming_edge_multi_hot) == incoming_edge_num

    """check shape of node features"""
    num_total_edge_labels = len(set(edge_labels)) * 2 
    incoming_and_outgoing = 2
    num_node_features = num_total_edge_labels + incoming_and_outgoing
    assert features.shape == (num_words,num_node_features)


def test_one_hot_encoding_on_gettysburg(gettys_text, stog):
    graph_builder_factory = GraphBuilderFactory()
    amr_graph_builder = graph_builder_factory.create_AMR_Builder(stog)
    amr_gettys_graphs = amr_graph_builder.make_graphs(gettys_text)
    test_graph = amr_gettys_graphs[0]

    """check one hot encoding of parts of speech"""
    edge_labels = featurizer.get_all_edge_labels(amr_gettys_graphs)
    num_edge_labels= len(set(edge_labels))

    labels_encoder = featurizer.fit_one_hot_encoding(amr_gettys_graphs)
    featurizer_context = FeatureContext(labels_encoder=labels_encoder, embedding_model=None)

    first_node_of_speech = test_graph.nodes[0]
    incoming_edges_vec = featurizer_context.one_hot_encode(first_node_of_speech.incoming_edge_labels)
    outgoing_edges_vec = featurizer_context.one_hot_encode(first_node_of_speech.outgoing_edge_labels)

    assert len(incoming_edges_vec) == num_edge_labels
    assert len(outgoing_edges_vec) == num_edge_labels
    assert len([v for v in incoming_edges_vec if v == 1]) == len(first_node_of_speech.incoming_edge_labels)
    assert len([v for v in outgoing_edges_vec if v == 1]) == len(first_node_of_speech.outgoing_edge_labels)


def test_featurizer_on_gettysburg_by_sentence(gettys_text,nlp,embedding_model,full_feature_extractor):
    """test featurizer on simple sentences, creating a graph for each sentence"""
    graph_builder_factory = GraphBuilderFactory()
    ud_graph_builder = graph_builder_factory.create_UD_Builder(nlp)
    ud_gettys_graphs = ud_graph_builder.make_graphs(gettys_text)
    
    features = featurizer.get_features(ud_gettys_graphs,full_feature_extractor,embedding_model)

    edge_labels = featurizer.get_all_edge_labels(ud_gettys_graphs)
    num_total_edge_labels = len(set(edge_labels)) * 2 
    incoming_and_outgoing = 2
    num_normalized_features = 2 # id, root
    num_node_features = num_normalized_features + num_total_edge_labels + incoming_and_outgoing + embedding_model.vector_size
    
    ud_gettys_doc = ud_graph_builder.parse_strategy.parse(gettys_text)
    num_nodes_in_first_graph = len(ud_gettys_doc[0].words)
    num_sentences = len(ud_gettys_doc)
    # shape = num_sentences, num_nodes, num_node_features, where num_nodes is inconstant
    assert features[0].shape == (num_nodes_in_first_graph, num_node_features)
    assert len(features) == num_sentences


def test_featurizer_on_gettysburg_using_merge(gettys_text,nlp,embedding_model,full_feature_extractor):
    """test featurizer on simple sentences, then merge all graphs into a single graph"""
    graph_builder_factory = GraphBuilderFactory()
    ud_graph_builder = graph_builder_factory.create_UD_Builder(nlp)
    gettys_merged_graph = ud_graph_builder.make_and_merge_graphs(gettys_text)

    features = featurizer.get_features([gettys_merged_graph],full_feature_extractor,embedding_model)

    edge_labels = featurizer.get_all_edge_labels([gettys_merged_graph])
    num_total_edge_labels = len(set(edge_labels)) * 2 
    incoming_and_outgoing = 2
    num_normalized_features = 2 # id, root
    num_node_features = num_normalized_features + num_total_edge_labels + incoming_and_outgoing + embedding_model.vector_size
    
    ud_gettys_doc = ud_graph_builder.parse_strategy.parse(gettys_text)
    num_nodes_in_doc = sum([len(ud_gettys_sentence.words) for ud_gettys_sentence in ud_gettys_doc])
    assert len(features) == 1
    assert features[0].shape == (num_nodes_in_doc, num_node_features)

