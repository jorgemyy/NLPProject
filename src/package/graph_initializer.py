from package.graph import Graph, Node, Edge
import stanza
import penman

def find_root_from_ud(ud_sentence):
    words = ud_sentence.words
    for word in words:
        if word.deprel == 'root':
            return word.id - 1

def get_graph_from_ud(ud_sentence):
    graph = Graph()
    root = find_root_from_ud(ud_sentence)
    words = ud_sentence.words
    for word in words:
        newNode = Node(id = word.id - 1,
                        text = word.text,
                        root = root,
                        )
        newNode.add_incoming_edge_label(word.deprel)
        graph.add_node(newNode)

        if word.head != 0:
            newEdge = Edge(source = word.head-1,
                           target = word.id-1)
            graph.add_edge(newEdge)

    for edge in graph.edges:
        graph.nodes[edge.source].add_outgoing_edge_label(graph.nodes[edge.target].incoming_edge_labels[0])

    return graph


def get_graph_from_amr(amr_penman_graph):
    graph = Graph()
    variables = list(sorted(amr_penman_graph.variables()))
    var_to_index = {var: i for i, var in enumerate(variables)}

    root = var_to_index[amr_penman_graph.top]

    for label,rel,concept in amr_penman_graph.instances():
        newNode = Node(id = var_to_index[label],
                       text = "".join([char for char in concept if not char.isdigit() and char != '-']),
                       root = root
                       )
        graph.add_node(newNode)
    
    for edge in amr_penman_graph.edges():
        source = var_to_index[edge.source]
        target = var_to_index[edge.target]
        newEdge = Edge(source=source,
                       target=target)
        graph.add_edge(newEdge)
        graph.nodes[target].add_incoming_edge_label(edge.role)
        graph.nodes[source].add_outgoing_edge_label(edge.role)

    return graph


def make_graphs(doc):
    graphs = []
    if type(doc) == stanza.models.common.doc.Document:
        ud_sentences = doc.sentences
        graphs = [get_graph_from_ud(sentence) for sentence in ud_sentences]
    elif type(doc) == list and type(doc[0]) == penman.graph.Graph:
        graphs = [get_graph_from_amr(sentence) for sentence in doc]
    return graphs 


def make_and_merge_graphs(doc):
    graphs = make_graphs(doc)
    
    merge_graph = graphs[0]
    for graph in graphs[1:]:
        merge_graph.merge(graph)

    return merge_graph