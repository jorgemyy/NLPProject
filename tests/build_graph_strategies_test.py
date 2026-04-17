import stanza
import penman
from package import graph_initializer
from package.graph import Graph
from package.build_graph_strategies import BuildUDGraphStrategy, BuildAMRGraphStrategy

def test_get_graph_from_ud(ud_obama_doc):
    """test whether nodes are created"""
    sentence = ud_obama_doc[0]
    assert type(sentence) == stanza.models.common.doc.Sentence

    build_strategy = BuildUDGraphStrategy()
    graph = Graph(sentence, build_strategy)

    num_words = len(sentence.text.split(' '))
    assert len(graph.nodes) == num_words
    assert graph.nodes[0].text == "Barack"
    assert graph.edges[0].target == 0 
    assert graph.edges[0].source == sentence.words[0].head - 1
    assert graph.get_edges_arr()[0] == [sentence.words[0].head - 1, 0]

def test_get_graph_from_amr(amr_hawaii_doc):
    """test whether nodes are created"""
    amr_penman = amr_hawaii_doc[0]
    assert type(amr_penman) == penman.graph.Graph

    build_strategy = BuildAMRGraphStrategy()
    graph = Graph(amr_penman, build_strategy)

    num_concepts = len(amr_penman.variables())
    num_edges = len(amr_penman.edges())
    num_incoming_edges = sum([len(node.incoming_edge_labels) for node in graph.nodes])
    num_outgoing_edges = sum([len(node.outgoing_edge_labels) for node in graph.nodes])

    assert len(graph.nodes) == num_concepts
    assert len(graph.edges) == num_edges
    assert num_incoming_edges == num_edges
    assert num_outgoing_edges == num_edges
    assert graph.nodes[0].root == 0

