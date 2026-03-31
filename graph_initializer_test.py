import graph_initializer
from featurizer_test import nlp

obama_sentence = "Barack Obama was born in Hawaii"
ud_obama_sentence = nlp(obama_sentence)

def test_get_graph_from_ud():
    """test whether nodes are created"""
    sentence = ud_obama_sentence.sentences[0]
    graph = graph_initializer.get_graph_from_ud(sentence)
    num_words = len(obama_sentence.split(' '))
    assert len(graph.nodes) == num_words
    assert graph.nodes[0].text == "Barack"
    assert graph.edges[0].source == 0 
    assert graph.edges[0].target == graph.nodes[0].head - 1
    assert graph.edges_arr[0] == (0, graph.nodes[0].head - 1)