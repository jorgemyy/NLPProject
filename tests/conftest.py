import pytest
import stanza
import gensim.downloader as gd
from package import graph_initializer
import amrlib
import penman
import nltk

def pytest_configure(config):
    stanza.download('en')

@pytest.fixture(scope="session")
def embedding_model():
    return gd.load("glove-wiki-gigaword-100")

@pytest.fixture(scope="session")
def nlp():
    return stanza.Pipeline('en')

@pytest.fixture(scope="session")
def embedding_model():
    return gd.load("glove-wiki-gigaword-100")

@pytest.fixture(scope="session")
def stog():
    return amrlib.load_stog_model()

@pytest.fixture(scope="session")
def ud_obama_doc(nlp):
    return nlp("Barack Obama was born in Hawaii")

@pytest.fixture(scope="session")
def amr_hawaii_doc(stog):
    graph = stog.parse_sents(['He flies to Hawaii'])[0]
    return [penman.decode(graph)]

@pytest.fixture(scope="session")
def ud_gettys_doc(nlp):
    with open("gettysburg.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    return nlp(text)

@pytest.fixture(scope="session")
def amr_gettys_list_of_graphs(stog):
    with open("gettysburg.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)
    graphs = stog.parse_sents(sentences)
    return [penman.decode(graph) for graph in graphs]

@pytest.fixture(scope="session")
def ud_obama_graphs(ud_obama_doc):
    return graph_initializer.make_graphs(ud_obama_doc)

@pytest.fixture(scope="session")
def ud_gettys_graphs(ud_gettys_doc):
    return graph_initializer.make_graphs(ud_gettys_doc)

@pytest.fixture(scope="session")
def amr_hawaii_graph(amr_hawaii_doc):
    return graph_initializer.make_graphs(amr_hawaii_doc)

@pytest.fixture(scope="session")
def amr_gettys_graphs(amr_gettys_list_of_graphs):
    return graph_initializer.make_graphs(amr_gettys_list_of_graphs)