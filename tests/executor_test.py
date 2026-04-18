import pytest
import torch
from torch_geometric.data import Data 
from sklearn.model_selection import train_test_split

from package.models.model_factory import ModelFactory
from package.executor import Executor
from package import graph
from package import featurizer

@pytest.fixture(scope="session")
def model_factory():
    return ModelFactory()

@pytest.fixture(scope="session")
def sentiment_analysis_model(model_factory):
    return model_factory.createSemModel()

@pytest.fixture(scope="session")
def amr_executor(sentiment_analysis_model,full_feature_extractor, embedding_model):
    return Executor(sentiment_analysis_model, 'amr', full_feature_extractor, embedding_model)

@pytest.fixture(scope="session")
def ud_executor(sentiment_analysis_model,full_feature_extractor, embedding_model):
    return Executor(sentiment_analysis_model, 'ud', full_feature_extractor, embedding_model)

@pytest.fixture(scope="session")
def full_df(sentiment_analysis_model):
    return sentiment_analysis_model.get_data()


def test_get_graphs_from_ud_df(full_df, ud_executor):
    '''test if graphs and labels are correctly extracted from a data frame using ud'''
    num_to_look_at = 5
    graphs, labels = ud_executor.get_ud_data_from_df(full_df.head(num_to_look_at))
    assert len(graphs) == len(labels)
    assert len(graphs) == num_to_look_at
    assert type(labels) == torch.Tensor
    assert type(graphs[0]) == graph.Graph


def test_get_graphs_from_amr_df(full_df, amr_executor):
    '''test if graphs and labels are correctly extracted from a data frame using ud'''
    num_to_look_at = 5
    graphs, labels = amr_executor.get_amr_data_from_df(full_df.head(num_to_look_at))
    assert len(graphs) == len(labels)
    assert len(graphs) == num_to_look_at
    assert type(labels) == torch.Tensor
    assert type(graphs[0]) == graph.Graph


def test_create_objects_for_gnn(full_df, ud_executor, amr_executor, embedding_model, full_feature_extractor):
    '''test if the correct data objects are being created'''
    ud_graphs, ud_labels = ud_executor.get_ud_data_from_df(full_df.head(1))
    amr_graphs, amr_labels = amr_executor.get_amr_data_from_df(full_df.head(1))

    assert type(ud_graphs[0]) == graph.Graph
    assert type(amr_graphs[0]) == graph.Graph

    first_ud_graph = ud_graphs[0]
    first_ud_label = ud_labels[0]
    first_amr_graph = amr_graphs[0]
    first_amr_label = amr_labels[0]

    ud_features = featurizer.get_features(ud_graphs,full_feature_extractor,embedding_model)
    amr_features = featurizer.get_features(amr_graphs,full_feature_extractor,embedding_model)

    ud_num_features = len(ud_features[0][0])
    amr_num_features = len(amr_features[0][0])

    ud_first_data_object = ud_executor.create_objects_for_gnn(full_df.head(1))[0]
    amr_first_data_object = amr_executor.create_objects_for_gnn(full_df.head(1))[0]

    assert type(ud_first_data_object) == Data
    assert ud_first_data_object.num_node_features == ud_num_features
    assert ud_first_data_object.num_nodes == len(first_ud_graph.nodes)
    assert ud_first_data_object.num_edges == len(first_ud_graph.edges)
    assert ud_first_data_object.y == first_ud_label

    assert type(amr_first_data_object) == Data
    assert amr_first_data_object.num_node_features == amr_num_features
    assert amr_first_data_object.num_nodes == len(first_amr_graph.nodes)
    assert amr_first_data_object.num_edges == len(first_amr_graph.edges)
    assert amr_first_data_object.y == first_amr_label


def test_ud_model(ud_executor, full_df):
    '''test train and eval'''
    ud_data_objects_small = ud_executor.create_objects_for_gnn(full_df.head(10))
    train_objects, test_objects = train_test_split(ud_data_objects_small, test_size=0.2)
    num_node_features = train_objects[0].num_node_features
    ud_executor.model.build_model(num_node_features)

    ud_executor.train_model(train_objects)
    acc = ud_executor.eval_model(test_objects) # right now, test and train would have shape mismatch - CHANGE THIS

    assert acc <= 1
    assert acc >= 0


def test_amr_model(amr_executor, full_df):
    '''test train and eval'''
    amr_data_objects_small = amr_executor.create_objects_for_gnn(full_df.head(10))
    train_objects, test_objects = train_test_split(amr_data_objects_small, test_size=0.2)
    num_node_features = train_objects[0].num_node_features
    amr_executor.model.build_model(num_node_features)

    amr_executor.train_model(train_objects)
    acc = amr_executor.eval_model(test_objects) # right now, test and train would have shape mismatch - CHANGE THIS

    assert acc <= 1
    assert acc >= 0


def test_run(ud_executor, amr_executor):
    ud_acc = ud_executor.run(cap=10)
    amr_acc = amr_executor.run(cap=10)

    assert ud_acc <= 1
    assert ud_acc >= 0
    assert amr_acc <= 1
    assert amr_acc >= 0