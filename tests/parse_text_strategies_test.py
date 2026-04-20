import stanza

from package.graphs.parse_text_strategies import*

def test_parse_ud(obama_sentence, nlp):
    """test if it is a list of UD sentences"""
    parser = UDParseStrategy(nlp)
    sentences = parser.parse(obama_sentence)

    assert type(sentences) == list
    assert all(type(sentence) == stanza.models.common.doc.Sentence for sentence in sentences)


def test_parse_amr(obama_sentence, stog):
    """test if it is a list of penman grpahs"""
    parser = AMRParseStrategy(stog)
    sentences = parser.parse(obama_sentence)

    assert type(sentences) == list
    assert all(type(sentence) == penman.graph.Graph for sentence in sentences)