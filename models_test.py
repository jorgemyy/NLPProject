from model_factory import ModelFactory
from featurizer import Featurizer
import stanza
import gensim.downloader as gd

stanza.download('en') 
nlp = stanza.Pipeline('en') 
word_embedding_model = gd.load("word2vec-google-news-300")

model_factory = ModelFactory()
featurizer = Featurizer("ud",word_embedding_model)

def test_sentiment_analysis_for_ud():
    model = model_factory.createModel("sem")
