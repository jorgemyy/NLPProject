from package import graph_initializer

def test_get_graph_from_ud(ud_obama_doc):
    """test whether nodes are created"""
    sentence = ud_obama_doc.sentences[0]
    graph = graph_initializer.get_graph_from_ud(sentence)
    num_words = len(ud_obama_doc.text.split(' '))
    assert len(graph.nodes) == num_words
    assert graph.nodes[0].text == "Barack"
    assert graph.edges[0].target == 0 
    assert graph.edges[0].source == graph.nodes[0].head - 1
    assert graph.edges_arr[0] == [graph.nodes[0].head - 1, 0]

def test_get_graph_from_amr(amr_obama_doc):
    """test whether nodes are created"""
    graph = graph_initializer.get_graph_from_amr(amr_obama_doc)
    num_concepts = len(amr_obama_doc.variables())
    assert len(graph.nodes) == num_concepts
    assert graph.nodes[0].root == 1


def test_make_and_merge_graphs(ud_gettys_doc):
    """test making a single graph for gettysburg"""
    gettys_graph = graph_initializer.make_and_merge_graphs(ud_gettys_doc)
    assert len(gettys_graph.nodes) == ud_gettys_doc.num_words
    
    graphs_not_merged = graph_initializer.make_graphs(ud_gettys_doc)
    assert len(gettys_graph.edges) == sum([len(graph.edges) for graph in graphs_not_merged])