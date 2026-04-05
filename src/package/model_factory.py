from src.package.sentiment_analysis import SentimentAnalysis

class ModelFactory():
    def __init__(self, num_node_features):
        self.num_node_features = num_node_features

    def createSemModel(self):
        return SentimentAnalysis()