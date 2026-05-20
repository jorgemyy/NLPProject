from abc import ABC, abstractmethod 
import nltk
import penman

class ParseTextStrategy(ABC):
    @abstractmethod
    def parse(text):
        pass


class DefaultParseTextStrategy(ParseTextStrategy):
    def parse(self, text):
        return text
    

class UDParseStrategy(ParseTextStrategy):
    def __init__(self, nlp):
        self.nlp = nlp

    def parse(self,text):
        ud_text = self.nlp(text)
        return ud_text.sentences
    

class AMRParseStrategy(ParseTextStrategy):
    def __init__(self, stog):
        self.stog = stog

    def parse(self,text):
        sentences = nltk.sent_tokenize(text)
        amr_sentences = self.stog.parse_sents(sentences)
        return [penman.decode(graph) for graph in amr_sentences]