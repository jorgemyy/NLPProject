from sentiment_analysis import SentimentAnalysis

class ModelFactory():
    def createModel(model_type):
        if model_type == "sem":
            model = SentimentAnalysis()
        return model