from package.models.model_factory import ModelFactory
from package.graphs.graph_builder_factory import GraphBuilderFactory
from package.data_pipeline import DataPipeline
from package.executor import Executor


def test_ud_model(nlp, full_feature_extractor, embedding_model):
    '''test train and eval'''
    model_factory = ModelFactory()
    sentiment_analysis_model = model_factory.createSemModel()
    
    graph_builder_factory = GraphBuilderFactory()
    ud_builder = graph_builder_factory.create_UD_Builder(nlp)

    pipeline = DataPipeline(sentiment_analysis_model, ud_builder, full_feature_extractor, embedding_model)
    
    executor = Executor(pipeline)
    acc = executor.run(cap=10) 

    assert acc <= 1
    assert acc >= 0


def test_amr_model(stog, full_feature_extractor, embedding_model):
    '''test train and eval'''
    model_factory = ModelFactory()
    sentiment_analysis_model = model_factory.createSemModel()
    
    graph_builder_factory = GraphBuilderFactory()
    amr_builder = graph_builder_factory.create_AMR_Builder(stog)

    pipeline = DataPipeline(sentiment_analysis_model, amr_builder, full_feature_extractor, embedding_model)
    
    executor = Executor(pipeline)
    acc = executor.run(cap=10) 

    assert acc <= 1
    assert acc >= 0