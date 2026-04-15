from package.graph import Graph, Node, Edge

def find_root_from_ud(ud_sentence):
    words = ud_sentence.words
    root = next(word for word in words if word.deprel == 'root')
    return root

def get_graph_from_ud(ud_sentence):
    graph = Graph()
    root = find_root_from_ud(ud_sentence)
    words = ud_sentence.words
    for word in words:
        newNode = Node(id = word.id,
                        text = word.text,
                        root = root,
                        head = word.head, 
                        incoming_edge_label=word.deprel
                        )
        graph.add_node(newNode)

    for node in graph.nodes:
        if node.head != 0:
            newEdge = Edge(source=node.head-1,
                           target=node.id-1,)
            graph.add_edge(newEdge)
            graph.nodes[word.head-1].set_outgoing_edge_label(node.incoming_edge_label)

    return graph


def get_graph_from_amr(amr_penman_graph):
    graph = Graph()
    variables = list(sorted(amr_penman_graph.variables()))
    var_to_index = {var: i+1 for i, var in enumerate(variables)}

    root = var_to_index[amr_penman_graph.top]

    for label,rel,concept in amr_penman_graph.instances():
        newNode = Node(id = var_to_index[label],
                       text = "".join([char for char in concept if not char.isdigit() and char != '-']),
                       root = root,
                       )
        graph.add_node(newNode)
        
    for edge in amr_penman_graph.edges():
        source = var_to_index[edge.source]
        target = var_to_index[edge.target]
        newEdge = Edge(source=source,
                       target=target)
        graph.add_edge(newEdge)
        graph.nodes[target-1].set_head(source)
        graph.nodes[target-1].set_incoming_edge_label(edge.role)
        graph.nodes[source-1].set_outgoing_edge_label(edge.role)

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