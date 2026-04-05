import pytest
import stanza
import gensim.downloader as gd
from package import graph_initializer

def pytest_configure(config):
    """Runs once before any tests — good place for downloads."""
    stanza.download('en')

@pytest.fixture(scope="session")
def embedding_model():
    return gd.load("glove-wiki-gigaword-100")

@pytest.fixture(scope="session")
def nlp():
    print("CREATING NLP PIPELINE")
    return stanza.Pipeline('en')

@pytest.fixture(scope="session")
def embedding_model():
    return gd.load("glove-wiki-gigaword-100")

@pytest.fixture(scope="session")
def ud_obama_doc(nlp):
    return nlp("Barack Obama was born in Hawaii")

@pytest.fixture(scope="session")
def ud_gettys_doc(nlp):
    with open("gettysburg.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    return nlp(text)

@pytest.fixture(scope="session")
def obama_graphs(ud_obama_doc):
    return graph_initializer.make_graphs(ud_obama_doc)

@pytest.fixture(scope="session")
def gettys_graphs(ud_gettys_doc):
    return graph_initializer.make_graphs(ud_gettys_doc)

