from package.graph import Graph, Node, Edge
from package.build_graph_strategies import BuildUDGraphStrategy, BuildAMRGraphStrategy, DefaultBuildGraphStrategy
import stanza
import penman

def make_graphs(doc):
    build_strategy = DefaultBuildGraphStrategy()
    if type(doc) == list and type(doc[0]) == stanza.models.common.doc.Sentence:
        build_strategy = BuildUDGraphStrategy()
        
    elif type(doc) == list and type(doc[0]) == penman.graph.Graph:
        build_strategy = BuildAMRGraphStrategy() 

    graphs = [Graph(sentence, build_strategy) for sentence in doc]
    return graphs


def make_and_merge_graphs(doc):
    graphs = make_graphs(doc)
    
    merge_graph = graphs[0]
    for graph in graphs[1:]:
        merge_graph.merge(graph)

    return merge_graph