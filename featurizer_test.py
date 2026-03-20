from featurizer import Featurizer
from graph import Node
import stanza
import gensim.downloader as gd
import torch

stanza.download('en') 
nlp = stanza.Pipeline('en') 
word_embedding_model = gd.load("word2vec-google-news-300")

obama_sentence = "Barack Obama was born in Hawaii"
ud_sentence = nlp(obama_sentence)
obama_featurizer = Featurizer("ud", word_embedding_model)

filename = "gettysburg.txt"
with open(filename, 'r', encoding='utf-8') as f:
    gettys_text = f.read()
ud_doc = nlp(gettys_text)


def test_get_graph_from_ud():
    """test whether nodes are created"""
    sentence = ud_sentence.sentences[0]
    graph = obama_featurizer.get_graph_from_ud(sentence)
    num_words = len(obama_sentence.split(' '))
    assert len(graph.nodes) == num_words
    assert graph.nodes[0].text == "Barack"
    assert graph.nodes[3].head == 0 #this is 'born' and the root of the sentence


def test_get_word_embeddings():
    word = "Born"
    node = Node(id = 0, text = word, upos = ".", xpos = ".", head = ".")
    embedding = obama_featurizer.get_word_embeddings(node)
    assert word in obama_featurizer.embedding_model
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding) == 300 #for the gensim 300 model


def test_get_word_embeddings_word_not_in_vocab():
    word = "garblebleekster"
    node = Node(id = 0, text = word, upos = ".", xpos = ".", head = ".")
    embedding = obama_featurizer.get_word_embeddings(node)
    assert word not in obama_featurizer.embedding_model
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding) == 300 
    assert all(v == 0 for v in embedding)


def test_get_features_from_graph():
    obama_featurizer.make_graphs(ud_sentence)
    obama_featurizer.fit_one_hot_encoding()
    features = obama_featurizer.get_features_from_graph(obama_featurizer.graphs[0])
    assert isinstance(features, torch.Tensor)

    first_word_head = obama_featurizer.graphs[0].nodes[0].head
    num_words = len(obama_sentence.split(' '))
    sentence_features = features[0]
    assert torch.isclose(sentence_features[0], torch.tensor(1 / num_words)) # normalized id
    assert torch.isclose(sentence_features[1], torch.tensor(first_word_head / num_words) if first_word_head != 0 else torch.tensor(0)) # normalized head id

    upos_tags, xpos_tags = obama_featurizer.get_pos_tags()
    num_upos, num_xpos = len(set(upos_tags)), len(set(xpos_tags))
    length_of_vec = 2 + num_upos + num_xpos + 300
    assert len(sentence_features) == length_of_vec


def test_one_hot_encoding_on_gettysburg():
    gettys_featurizer = Featurizer("ud", word_embedding_model)
    gettys_featurizer.make_graphs(ud_doc)
    upos_tags, xpos_tags = gettys_featurizer.get_pos_tags()
    num_upos, num_xpos = len(set(upos_tags)), len(set(xpos_tags))
    gettys_featurizer.fit_one_hot_encoding()

    first_word_of_speech = gettys_featurizer.graphs[0].nodes[0]
    pos_vec = gettys_featurizer.one_hot_encode(first_word_of_speech)
    upos_vec, xpos_vec = pos_vec[0], pos_vec[1]

    assert len(upos_vec) == num_upos
    assert len(xpos_vec) == num_xpos
    assert len([v for v in upos_vec if v == 1]) == 1
    assert len([v for v in xpos_vec if v == 1]) == 1


def test_featurizer_on_gettysburg():
    """test featurizer on simple sentences"""
    new_gettys_featurizer = Featurizer("ud", word_embedding_model)
    features = new_gettys_featurizer.get_features(ud_doc)

    num_sentences = len(gettys_text.split('.'))
    assert len(features) == num_sentences
