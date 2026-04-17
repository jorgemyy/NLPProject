from package import graph_initializer
import penman
import stanza
from package.graph import Graph

def test_get_graph_from_ud(ud_obama_doc):
    """test whether nodes are created"""
    sentence = ud_obama_doc.sentences[0]
    assert type(sentence) == stanza.models.common.doc.Sentence
    graph = graph_initializer.get_graph_from_ud(sentence)
    num_words = len(ud_obama_doc.text.split(' '))
    assert len(graph.nodes) == num_words
    assert graph.nodes[0].text == "Barack"
    assert graph.edges[0].target == 0 
    assert graph.edges[0].source == sentence.words[0].head - 1
    assert graph.edges_arr[0] == [sentence.words[0].head - 1, 0]

def test_get_graph_from_amr(amr_hawaii_doc):
    """test whether nodes are created"""
    amr_penman = amr_hawaii_doc[0]
    assert type(amr_penman) == penman.graph.Graph
    graph = graph_initializer.get_graph_from_amr(amr_penman)
    num_concepts = len(amr_penman.variables())
    num_edges = len(amr_penman.edges())
    num_incoming_edges = sum([len(node.incoming_edge_labels) for node in graph.nodes])
    num_outgoing_edges = sum([len(node.outgoing_edge_labels) for node in graph.nodes])

    assert len(graph.nodes) == num_concepts
    assert len(graph.edges) == num_edges
    assert num_incoming_edges == num_edges
    assert num_outgoing_edges == num_edges
    assert graph.nodes[0].root == 0


def test_get_graphs_from_amr(amr_gettys_list_of_graphs):
    """test whether nodes are created"""
    assert type(amr_gettys_list_of_graphs) == list and type(amr_gettys_list_of_graphs[0]) == penman.graph.Graph
    graphs = graph_initializer.make_graphs(amr_gettys_list_of_graphs)
    assert type(graphs[0]) == Graph
    num_concepts = sum([len(sentence.variables()) for sentence in amr_gettys_list_of_graphs])
    num_edges = sum([len(sentence.edges()) for sentence in amr_gettys_list_of_graphs])

    num_incoming_edges = 0
    num_outgoing_edges = 0
    num_nodes = 0
    num_edges_from_graph_objects = 0
    for graph in graphs:
        num_incoming_edges +=  sum([len(node.incoming_edge_labels) for node in graph.nodes])
        num_outgoing_edges += sum([len(node.outgoing_edge_labels) for node in graph.nodes])
        num_nodes += len(graph.nodes)
        num_edges_from_graph_objects += len(graph.edges)

    assert num_nodes == num_concepts
    assert num_edges_from_graph_objects == num_edges
    assert num_outgoing_edges == num_edges
    assert num_incoming_edges == num_edges


def test_make_and_merge_graphs(ud_gettys_doc):
    """test making a single graph for gettysburg"""
    assert(type(ud_gettys_doc) == stanza.models.common.doc.Document)
    gettys_graph = graph_initializer.make_and_merge_graphs(ud_gettys_doc)
    assert(type(gettys_graph) == Graph)
    assert len(gettys_graph.nodes) == ud_gettys_doc.num_words
    
    graphs_not_merged = graph_initializer.make_graphs(ud_gettys_doc)
    assert len(gettys_graph.edges) == sum([len(graph.edges) for graph in graphs_not_merged])