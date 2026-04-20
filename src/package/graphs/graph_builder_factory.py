import stanza
import amrlib

from package.graphs.graph_builder import GraphBuilder
from package.graphs.build_graph_strategies import *
from package.graphs.parse_text_strategies import *

class GraphBuilderFactory:
    def __init__(self):
        pass

    def create_UD_Builder(self, nlp=None):
        nlp = stanza.Pipeline('en') if nlp == None else nlp
        return GraphBuilder(BuildUDGraphStrategy(), UDParseStrategy(nlp))
    
    def create_AMR_Builder(self, stog=None):
        stog = amrlib.load_stog_model() if stog == None else stog
        return GraphBuilder(BuildAMRGraphStrategy(), AMRParseStrategy(stog))