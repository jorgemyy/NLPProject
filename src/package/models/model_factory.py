from package.models.sentiment_analysis_model import SentimentAnalysis

class ModelFactory():
    def __init__(self, num_node_features):
        self.num_node_features = num_node_features

    def createSemModel(self, num_classes, out_dim=16):
        return SentimentAnalysis(self.num_node_features, num_classes, out_dim)