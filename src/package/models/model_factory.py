from package.models.sentiment_analysis_model import SentimentAnalysisModel

class ModelFactory():
    def createSemModel(self):
        return SentimentAnalysisModel()