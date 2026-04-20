from package.models.sentiment_analysis_model import SentimentAnalysisModel

class ModelFactory():
    def createSemModel(self, batch_size=16, epochs=200, out_dim=16):
        return SentimentAnalysisModel(batch_size, epochs, out_dim)