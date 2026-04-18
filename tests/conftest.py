import pytest
import stanza
import gensim.downloader as gd
import amrlib
import penman
import nltk

from package import graph_initializer
from package.featurizer_decorator import FeatureExtractorBuilder

def pytest_configure(config):
    stanza.download('en')

@pytest.fixture(scope="session")
def embedding_model():
    return gd.load("glove-wiki-gigaword-100")

@pytest.fixture(scope="session")
def nlp():
    return stanza.Pipeline('en')

@pytest.fixture(scope="session")
def stog():
    return amrlib.load_stog_model()

@pytest.fixture(scope="session")
def full_feature_extractor():
    feature_extractor = (FeatureExtractorBuilder()
                         .add_id()
                         .add_root_distance()
                         .add_incoming_labels()
                         .add_outgoing_labels()
                         .add_embedding()
                         .build())
    return feature_extractor

@pytest.fixture(scope="session")
def ud_obama_doc(nlp):
    ud_document = nlp("Barack Obama was born in Hawaii")
    return ud_document.sentences

@pytest.fixture(scope="session")
def amr_hawaii_doc(stog):
    amr_graph = stog.parse_sents(['He flies to Hawaii'])[0]
    return [penman.decode(amr_graph)]

@pytest.fixture(scope="session")
def ud_gettys_doc(nlp):
    with open("gettysburg.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    ud_document = nlp(text)
    return ud_document.sentences

@pytest.fixture(scope="session")
def amr_gettys_doc(stog):
    with open("gettysburg.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)
    amr_graphs = stog.parse_sents(sentences)
    return [penman.decode(graph) for graph in amr_graphs]

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
def amr_gettys_graphs(amr_gettys_doc):
    return graph_initializer.make_graphs(amr_gettys_doc)