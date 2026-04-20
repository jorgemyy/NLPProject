from torch_geometric.data import Data

from package.models.model_factory import ModelFactory
from package.graphs.graph_builder_factory import GraphBuilderFactory
from package.data_pipeline import DataPipeline

def test_prepare_ud(nlp, embedding_model, full_feature_extractor):
    '''test if the correct data objects are being created'''

    model_factory = ModelFactory()
    sentiment_analysis_model = model_factory.createSemModel()

    graph_builder_factory = GraphBuilderFactory()
    ud_builder = graph_builder_factory.create_UD_Builder(nlp)

    pipeline = DataPipeline(sentiment_analysis_model, ud_builder, full_feature_extractor, embedding_model)

    train_objects, test_objects, num_node_features = pipeline.prepare(cap=10)

    assert(all(type(train_object) == Data for train_object in train_objects))
    assert(all(type(test_object) == Data for test_object in test_objects))

    assert len(train_objects) == 8
    assert len(test_objects) == 2


def test_prepare_amr(stog, embedding_model, full_feature_extractor):
    '''test if the correct data objects are being created'''

    model_factory = ModelFactory()
    sentiment_analysis_model = model_factory.createSemModel()

    graph_builder_factory = GraphBuilderFactory()
    amr_builder = graph_builder_factory.create_AMR_Builder(stog)

    pipeline = DataPipeline(sentiment_analysis_model, amr_builder, full_feature_extractor, embedding_model)

    train_objects, test_objects, num_node_features = pipeline.prepare(cap=10)

    assert(all(type(train_object) == Data for train_object in train_objects))
    assert(all(type(test_object) == Data for test_object in test_objects))

    assert len(train_objects) == 8
    assert len(test_objects) == 2