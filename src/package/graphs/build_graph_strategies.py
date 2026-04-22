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
                            node_type = word.upos,
                            negated = False
                            )
            newNode.add_incoming_edge_label(word.deprel)
            graph.add_node(newNode)

            if word.head != 0:
                newEdge = Edge(source = word.head-1,
                            target = word.id-1, 
                            label = word.deprel)
                graph.add_edge(newEdge)

                newRevEdge = Edge(source=word.id-1,
                        target=word.head-1,
                        label=word.deprel + "_rev")
                graph.add_edge(newRevEdge)
            

        for edge in graph.edges:
            graph.nodes[edge.source].add_outgoing_edge_label(graph.nodes[edge.target].incoming_edge_labels[0])

            if words[edge.source].feats and "=Neg" in words[edge.source].feats:
                graph.nodes[edge.target].negated = True
        
        return graph
    

class BuildAMRGraphStrategy(BuildGraphStrategy):
    def get_concept_type(self, concept):
        concept = concept.split("-")[-1] if "-" in concept else concept
        if concept.isdigit():
            return "predicate"
        elif concept in {"possible", "likely", "necessary", "obligate", "desire"}:
            return "modal"
        else:
            return "entity"


    def build_graph(self, graph):
        amr_penman_graph = graph.doc
        variables = list(sorted(amr_penman_graph.variables()))
        var_to_index = {var: i for i, var in enumerate(variables)}

        root = var_to_index[amr_penman_graph.top]
        graph.set_root(root)

        neg_nodes = [a.source for a in amr_penman_graph.attributes() if (a.role==':polarity' and a.target=='-')]

        for label,rel,concept in amr_penman_graph.instances():
            newNode = Node(id = var_to_index[label],
                        text = "".join([char for char in concept if not char.isdigit() and char != '-']),
                        root = root,
                        negated = label in neg_nodes,
                        node_type = self.get_concept_type(concept)
                        )
            graph.add_node(newNode)
        
        for edge in amr_penman_graph.edges():
            source = var_to_index[edge.source]
            target = var_to_index[edge.target]
            newEdge = Edge(source=source,
                        target=target,
                        label=edge.role)
            graph.add_edge(newEdge)

            newRevEdge = Edge(source=target,
                        target=source,
                        label=edge.role + "_rev")
            graph.add_edge(newRevEdge)
            
            graph.nodes[target].add_incoming_edge_label(edge.role)
            graph.nodes[source].add_outgoing_edge_label(edge.role)

        return graph