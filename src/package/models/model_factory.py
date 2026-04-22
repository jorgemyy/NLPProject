from package.models.sentiment_analysis_model import SentimentAnalysisModel

class ModelFactory():
    def createSemModel(self, batch_size=16, epochs=100, out_dim=128):
        return SentimentAnalysisModel(batch_size, epochs, out_dim)