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
        self.edges_tuple.append([edge.source,edge.target])


class Edge:
    def __init__(self, source, target, label):
        self.source = source
        self.target = target
        self.label = label


class Node:
    def __init__(self, id, text, upos, xpos, head):
        self.id = id
        self.text = text
        self.upos = upos
        self.xpos = xpos
        self.head = head