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
                upos=node.upos,
                xpos=node.xpos,
                head=node.head  
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
    def __init__(self, id, text, upos, xpos, head):
        self.id = id
        self.text = text
        self.upos = upos
        self.xpos = xpos
        self.head = head