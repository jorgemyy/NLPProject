from abc import ABC, abstractmethod 
from package.graphs.graph import Graph, Node, Edge

class BuildGraphStrategy(ABC):
    @abstractmethod
    def build_graph(graph) -> Graph:
        pass


class DefaultBuildGraphStrategy(BuildGraphStrategy):
    def build_graph(self, graph):
        return graph


class BuildUDGraphStrategy(BuildGraphStrategy):
    def find_root_from_ud(self, ud_sentence):
        words = ud_sentence.words
        for word in words:
            if word.deprel == 'root':
                return word.id - 1
        
    def build_graph(self, graph):
        ud_sentence = graph.doc
        root = self.find_root_from_ud(ud_sentence)
        graph.set_root(root)

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
    

class BuildAMRGraphStrategy(BuildGraphStrategy):
    def build_graph(self, graph):
        amr_penman_graph = graph.doc
        variables = list(sorted(amr_penman_graph.variables()))
        var_to_index = {var: i for i, var in enumerate(variables)}

        root = var_to_index[amr_penman_graph.top]
        graph.set_root(root)

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