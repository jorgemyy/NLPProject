from package.graphs.parse_text_strategies import*
from package.graphs.graph import Graph
from package.graphs.build_graph_strategies import BuildUDGraphStrategy, BuildAMRGraphStrategy

def test_get_graph_from_ud(obama_sentence, nlp):
    """test whether nodes are created"""
    parser = UDParseStrategy(nlp)
    sentences = parser.parse(obama_sentence)
    first_sentence = sentences[0]

    build_strategy = BuildUDGraphStrategy()
    graph = Graph(first_sentence, build_strategy)

    num_words = len(first_sentence.text.split(' '))
    assert len(graph.nodes) == num_words
    assert graph.nodes[0].text == "Barack"
    assert graph.edges[0].target == 0 
    assert graph.edges[0].source == first_sentence.words[0].head - 1
    assert graph.get_edges_arr()[0] == [first_sentence.words[0].head - 1, 0]
    assert graph.nodes[0].node_type == first_sentence.words[0].upos
    assert all(not node.negated for node in graph.nodes)


def test_get_graph_from_amr(obama_sentence, stog):
    """test whether nodes are created"""
    parser = AMRParseStrategy(stog)
    sentences = parser.parse(obama_sentence)
    first_sentence = sentences[0]

    build_strategy = BuildAMRGraphStrategy()
    graph = Graph(first_sentence, build_strategy)

    num_concepts = len(first_sentence.variables())
    num_edges = len(first_sentence.edges())
    num_incoming_edges = sum([len(node.incoming_edge_labels) for node in graph.nodes])
    num_outgoing_edges = sum([len(node.outgoing_edge_labels) for node in graph.nodes])

    assert len(graph.nodes) == num_concepts
    assert len(graph.edges) == num_edges * 2
    assert num_incoming_edges == num_edges
    assert num_outgoing_edges == num_edges 
    assert graph.nodes[0].root == 0
    assert all(node.node_type in ["predicate", "modal", "entity"] for node in graph.nodes)
    assert all(not node.negated for node in graph.nodes)


def test_create_negative_ud_sentence(nlp):
    sentence = "That was not good"
    parser = UDParseStrategy(nlp)
    sentences = parser.parse(sentence)
    first_sentence = sentences[0]

    build_strategy = BuildUDGraphStrategy()
    graph = Graph(first_sentence, build_strategy)

    assert graph.nodes[-1].negated  == True


def test_create_negative_amr_sentence(stog):
    sentence = "That was not good"
    parser = AMRParseStrategy(stog)
    sentences = parser.parse(sentence)
    first_sentence = sentences[0]

    build_strategy = BuildAMRGraphStrategy()
    graph = Graph(first_sentence, build_strategy)

    assert any(node.negated for node in graph.nodes) 