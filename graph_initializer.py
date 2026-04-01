from graph import Graph, Node, Edge
import torch
import featurizer
from torch_geometric.data import Data

def get_graph_from_ud(ud_sentence):
    graph = Graph()
    words = ud_sentence.words
    for word in words:
        newNode = Node(id = word.id,
                        text = word.text,
                        upos = word.upos,
                        xpos = word.xpos,
                        head = word.head
                        )
        graph.add_node(newNode)

        if word.head != 0:
            newEdge = Edge(source=word.head-1,
                           target=word.id-1,
                           label=word.deprel)
            graph.add_edge(newEdge)

    return graph
    

def make_graphs(ud_doc):
    ud_sentences = ud_doc.sentences
    graphs = [get_graph_from_ud(sentence) for sentence in ud_sentences]
    return graphs


def create_objects_for_gnn(graphs):
    data_objects = []
    features = featurizer.get_features(graphs)
    for i in range(len(graphs)):
        edge_index = torch.tensor(graphs[i].edges_arr, dtype = torch.long)
        x = features[i]
        data_graph = Data(x=x, edge_index=edge_index.t().contiguous())
        data_objects.append(data_graph)
    return data_objects
