class Graph():
    def __init__(self, doc, build_strategy):
        self.nodes = []
        self.edges = []
        self.root = None
        self.doc = doc
        build_strategy.build_graph(self)

    def set_root(self, root):
        self.root = root

    def print_graph(self):
        pass 

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_edges_arr(self):
        return [[edge.source,edge.target] for edge in self.edges]
    
    def get_edge_labels(self):
        return [edge.label for edge in self.edges]

    def merge(self, other):
        offset = len(self.nodes)  

        for node in other.nodes:
            new_node = Node(
                id=node.id + offset,
                text=node.text,
                root=node.root,
                incoming_edge_labels=node.incoming_edge_labels,  
                outgoing_edge_labels=node.outgoing_edge_labels
            )
            self.add_node(new_node)

        for edge in other.edges:
            new_edge = Edge(
                source=edge.source + offset,
                target=edge.target + offset,
                label=edge.label
            )
            self.add_edge(new_edge)


class Edge:
    def __init__(self, source, target, label):
        self.source = source
        self.target = target
        self.label = label


class Node:
    def __init__(self, text, id, root, outgoing_edge_labels=None, incoming_edge_labels=None):
        self.id = id
        self.text = text
        self.root = root
        self.outgoing_edge_labels = outgoing_edge_labels if outgoing_edge_labels is not None else []
        self.incoming_edge_labels = incoming_edge_labels if incoming_edge_labels is not None else []

    def add_outgoing_edge_label(self, label):
        self.outgoing_edge_labels.append(label)

    def add_incoming_edge_label(self, label):
        self.incoming_edge_labels.append(label)