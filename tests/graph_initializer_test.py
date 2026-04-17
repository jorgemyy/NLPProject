from package import graph_initializer
import penman
import stanza
from package.graph import Graph

def test_get_graphs_from_amr(amr_gettys_doc):
    """test whether nodes are created"""
    assert type(amr_gettys_doc) == list and type(amr_gettys_doc[0]) == penman.graph.Graph
    graphs = graph_initializer.make_graphs(amr_gettys_doc)
    assert type(graphs[0]) == Graph
    num_concepts = sum([len(sentence.variables()) for sentence in amr_gettys_doc])
    num_edges = sum([len(sentence.edges()) for sentence in amr_gettys_doc])

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
    assert type(ud_gettys_doc) == list
    assert type(ud_gettys_doc[0]) == stanza.models.common.doc.Sentence
    gettys_graph = graph_initializer.make_and_merge_graphs(ud_gettys_doc)
    assert type(gettys_graph) == Graph
    assert len(gettys_graph.nodes) == sum([len(doc.words) for doc in ud_gettys_doc])
    
    graphs_not_merged = graph_initializer.make_graphs(ud_gettys_doc)
    assert len(gettys_graph.edges) == sum([len(graph.edges) for graph in graphs_not_merged])