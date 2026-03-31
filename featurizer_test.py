import featurizer
import graph_initializer
from graph import Node
import stanza
import gensim.downloader as gd
import torch

stanza.download('en') 
nlp = stanza.Pipeline('en') 

obama_sentence = "Barack Obama was born in Hawaii"
ud_obama_sentence = nlp(obama_sentence)

filename = "gettysburg.txt"
with open(filename, 'r', encoding='utf-8') as f:
    gettys_text = f.read()
ud_gettys_doc = nlp(gettys_text)

obama_graphs = graph_initializer.make_graphs(ud_obama_sentence)
gettys_graphs = graph_initializer.make_graphs(ud_gettys_doc)

embedding_model = gd.load("glove-wiki-gigaword-50")

def test_get_word_embeddings():
    """check if a given word is embedded"""
    word = "Born"
    node = Node(id = 0, text = word, upos = ".", xpos = ".", head = ".")
    embedding = featurizer.get_word_embeddings(node,embedding_model)
    assert word in embedding_model
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding) == 300 #for the gensim 300 model


def test_get_word_embeddings_word_not_in_vocab():
    """check if a given made up is all 0s"""
    word = "garblebleekster"
    node = Node(id = 0, text = word, upos = ".", xpos = ".", head = ".")
    embedding = featurizer.get_word_embeddings(node,embedding_model)
    assert word not in embedding_model
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding) == 300 
    assert all(v == 0 for v in embedding)


def test_get_features_from_graph():
    """check if features are extracted from the graph"""
    upos_encoder, xpos_encoder = featurizer.fit_one_hot_encoding(obama_graphs)
    test_graph = obama_graphs[0]
    features = featurizer.get_features_from_graph(test_graph,upos_encoder,xpos_encoder,embedding_model)
    assert isinstance(features, torch.Tensor)

    """check normalized id and normalized head"""
    first_word_head = obama_graphs[0].nodes[0].head
    num_words = len(obama_sentence.split(' '))
    first_sentence_features = features[0]
    assert torch.isclose(first_sentence_features[0], torch.tensor(1 / num_words)) # normalized id
    assert torch.isclose(first_sentence_features[1], torch.tensor(first_word_head / num_words) if first_word_head != 0 else torch.tensor(0)) # normalized head id

    """check shape of node features"""
    upos_tags, xpos_tags = featurizer.get_pos_tags(obama_graphs)
    num_upos, num_xpos = len(set(upos_tags)), len(set(xpos_tags))
    embedding_shape = 300 #from gensim model
    num_normalized_features = 2 # id, head
    num_node_features = num_normalized_features + num_upos + num_xpos + embedding_shape
    num_nodes = len(test_graph.nodes)
    assert features.shape == (num_nodes,num_node_features)


def test_one_hot_encoding_on_gettysburg():
    """check one hot encoding of parts of speech"""
    upos_tags, xpos_tags = featurizer.get_pos_tags(gettys_graphs)
    num_upos, num_xpos = len(set(upos_tags)), len(set(xpos_tags))
    upos_encoder, xpos_encoder = featurizer.fit_one_hot_encoding(gettys_graphs)

    first_word_of_speech = gettys_graphs[0].nodes[0]
    pos_vec = featurizer.one_hot_encode(first_word_of_speech, upos_encoder, xpos_encoder)
    upos_vec, xpos_vec = pos_vec[0], pos_vec[1]

    assert len(upos_vec) == num_upos
    assert len(xpos_vec) == num_xpos
    assert len([v for v in upos_vec if v == 1]) == 1
    assert len([v for v in xpos_vec if v == 1]) == 1


def test_featurizer_on_gettysburg():
    """test featurizer on simple sentences"""
    features = featurizer.get_features(gettys_graphs,embedding_model)

    upos_tags, xpos_tags = featurizer.get_pos_tags(gettys_graphs)
    num_upos, num_xpos = len(set(upos_tags)), len(set(xpos_tags))
    embedding_shape = 300 #from gensim model
    num_normalized_features = 2 # id, head
    num_node_features = num_normalized_features + num_upos + num_xpos + embedding_shape
    num_nodes_in_first_graph = len(gettys_graphs[0].nodes)
    num_sentences = len(ud_gettys_doc.sentences)
    # shape = num_sentences, num_nodes, num_node_features, where num_nodes is inconstant
    assert features[0].shape == (num_nodes_in_first_graph, num_node_features)
    assert len(features) == num_sentences
