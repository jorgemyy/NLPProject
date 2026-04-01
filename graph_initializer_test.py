import graph_initializer
from featurizer_test import nlp
from torch_geometric import data

obama_sentence = "Barack Obama was born in Hawaii"
ud_obama_sentence = nlp(obama_sentence)

filename = "gettysburg.txt"
with open(filename, 'r', encoding='utf-8') as f:
    gettys_text = f.read()
ud_gettys_doc = nlp(gettys_text)
gettys_graphs = graph_initializer.make_graphs(ud_gettys_doc)


def test_get_graph_from_ud():
    """test whether nodes are created"""
    sentence = ud_obama_sentence.sentences[0]
    graph = graph_initializer.get_graph_from_ud(sentence)
    num_words = len(obama_sentence.split(' '))
    assert len(graph.nodes) == num_words
    assert graph.nodes[0].text == "Barack"
    assert graph.edges[0].target == 0 
    assert graph.edges[0].source == graph.nodes[0].head - 1
    assert graph.edges_arr[0] == [graph.nodes[0].head - 1, 0]


def test_create_objects_for_gnn():
    data_objects = graph_initializer.create_objects_for_gnn(gettys_graphs)
    first_data_object = data_objects[0]
    assert isinstance(first_data_object, data.Data)
    assert first_data_object.num_nodes == len(gettys_graphs[0].nodes)
    assert first_data_object.num_edges == len(gettys_graphs[0].edges)