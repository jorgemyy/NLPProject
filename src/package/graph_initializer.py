from src.package.graph import Graph, Node, Edge

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


def make_and_merge_graphs(ud_doc):
    graphs = make_graphs(ud_doc)
    
    merge_graph = graphs[0]
    for graph in graphs[1:]:
        merge_graph.merge(graph)

    return merge_graph