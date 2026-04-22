import pytest
import stanza
import gensim.downloader as gd
import amrlib

from package.features.featurizer_decorator import FeatureExtractorBuilder

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
def obama_sentence():
    return "Barack Obama was born in Hawaii"

@pytest.fixture(scope="session")
def gettys_text():
    with open("gettysburg.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    return text


@pytest.fixture(scope="session")
def full_feature_extractor():
    feature_extractor = (FeatureExtractorBuilder()
                         .add_id()
                         .add_root()
                         .add_type()
                         .add_neg()
                         .add_embedding()
                         .build())
    return feature_extractor