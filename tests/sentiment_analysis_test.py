import pandas as pd
import torch
from torch_geometric.data import Data
import pytest
from sklearn.model_selection import train_test_split

from package import graph
from package import sentiment_analysis
from package import featurizer
from package.models.model_factory import ModelFactory

@pytest.fixture(scope="session")
def full_df():
    return sentiment_analysis.get_data()

@pytest.fixture(scope="session")
def first_data_object(full_df, nlp):
    return sentiment_analysis.create_objects_for_gnn(full_df.head(1), nlp)[0]

@pytest.fixture(scope="session")
def data_objects_small(full_df, nlp):
    return sentiment_analysis.create_objects_for_gnn(full_df.head(20), nlp)

@pytest.fixture(scope="session")
def data_objects_full(full_df, nlp):
    return sentiment_analysis.create_objects_for_gnn(full_df.head(5000), nlp)


def test_get_data(full_df):
    '''test if the data is successfully loaded from kaggles with the correct labels'''
    assert isinstance(full_df,pd.DataFrame)
    assert not full_df.isna().values.any()

    assert all(x <= 2 for x in set(full_df["sentiment"]))


def test_get_graphs_from_ud_df(full_df, nlp):
    '''test if graphs and labels are correctly extracted from a data frame using ud'''
    num_to_look_at = 5
    graphs, labels = sentiment_analysis.get_ud_data_from_df(full_df.head(num_to_look_at),nlp)
    assert len(graphs) == len(labels)
    assert len(graphs) == num_to_look_at
    assert isinstance(labels,torch.Tensor)
    assert isinstance(graphs[0],graph.Graph)


def test_create_objects_for_gnn(first_data_object, full_df, nlp):
    '''test if the correct data objects are being created'''
    graphs, labels = sentiment_analysis.get_ud_data_from_df(full_df.head(1),nlp)

    first_ud_graph = graphs[0]
    first_ud_label = labels[0]
    num_features = featurizer.get_features(graphs)[0].shape[1]

    assert isinstance(first_data_object,Data)
    assert first_data_object.num_node_features == num_features
    assert first_data_object.num_nodes == len(first_ud_graph.nodes)
    assert first_data_object.num_edges == len(first_ud_graph.edges)
    assert first_data_object.y == first_ud_label


def test_model(data_objects_full):
    '''test train and eval'''
    train_objects, test_objects = train_test_split(data_objects_full, test_size=0.2)

    model_factory = ModelFactory(num_node_features=train_objects[0].num_node_features)
    sem_model = model_factory.createSemModel(num_classes=3)

    sentiment_analysis.train_model(sem_model, train_objects)
    acc = sentiment_analysis.eval_model(sem_model, test_objects) # right now, test and train would have shape mismatch - CHANGE THIS

    assert acc <= 1
    assert acc > 0
