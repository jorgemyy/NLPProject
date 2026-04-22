import pandas as pd
import nltk

from package.graphs.graph import Graph
from package.graphs.graph_builder_factory import GraphBuilderFactory

def test_make_graphs_from_amr(gettys_text, stog):
    """test whether graphs are created"""
    graph_builder_factory = GraphBuilderFactory()
    amr_graph_builder = graph_builder_factory.create_AMR_Builder(stog)

    graphs = amr_graph_builder.make_graphs(gettys_text)
    assert all(type(graph) == Graph for graph in graphs)

    """test graph features"""
    amr_gettys_doc = amr_graph_builder.parse_strategy.parse(gettys_text)
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
    assert num_edges_from_graph_objects == num_edges * 2
    assert num_outgoing_edges == num_edges
    assert num_incoming_edges == num_edges 


def test_make_and_merge_graphs_from_ud(gettys_text, nlp):
    """test making a single graph for gettysburg"""
    graph_builder_factory = GraphBuilderFactory()
    ud_graph_builder = graph_builder_factory.create_UD_Builder(nlp)

    gettys_graph = ud_graph_builder.make_and_merge_graphs(gettys_text)
    assert type(gettys_graph) == Graph

    ud_gettys_doc = ud_graph_builder.parse_strategy.parse(gettys_text)
    assert len(gettys_graph.nodes) == sum([len(doc.words) for doc in ud_gettys_doc])
    
    graphs_not_merged = ud_graph_builder.make_graphs(gettys_text)
    assert len(gettys_graph.edges) == sum([len(graph.edges) for graph in graphs_not_merged])


def test_build_graphs_from_df(gettys_text, nlp):
    sentences = nltk.sent_tokenize(gettys_text)
    num_sentences = len(sentences)
    data = {
    "text": sentences,
    "label": [0] * num_sentences}
 
    df = pd.DataFrame(data)
    graph_builder_factory = GraphBuilderFactory()
    ud_graph_builder = graph_builder_factory.create_UD_Builder(nlp)

    graphs, labels = ud_graph_builder.build_graphs_from_df(df)
    assert len(graphs) == num_sentences
    assert len(labels) == num_sentences
    assert all(type(graph) == Graph for graph in graphs)