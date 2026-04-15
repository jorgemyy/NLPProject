class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.edges_arr = []
    
    def print_graph(self):
        pass 

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)
        self.edges_arr.append([edge.source,edge.target])

    def merge(self, other):
        offset = len(self.nodes)  

        for node in other.nodes:
            new_node = Node(
                id=node.id + offset,
                text=node.text,
                root=node.root,
                head = node.head,
                incoming_edge_label=node.incoming_edge_label,  
                outgoing_edge_label=node.outgoing_edge_label
            )
            self.add_node(new_node)

        for edge in other.edges:
            new_edge = Edge(
                source=edge.source + offset,
                target=edge.target + offset,
            )
            self.add_edge(new_edge)


class Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target


class Node:
    def __init__(self, text, id, root, head=None, outgoing_edge_label=None, incoming_edge_label=None):
        self.id = id
        self.text = text
        self.root = root
        self.head = head
        self.outgoing_edge_label = outgoing_edge_label
        self.incoming_edge_label = incoming_edge_label

    def set_head(self, head):
        self.head = head

    def set_outgoing_edge_label(self,label):
        self.outgoing_edge_label = label

    def set_incoming_edge_label(self, label):
        self.incoming_edge_label = label