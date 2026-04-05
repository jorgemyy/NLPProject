import pandas as pd
import torch
from torch_geometric.data import Data
import pytest

from package import graph
from package import sentiment_analysis
from package import featurizer

@pytest.fixture(scope="session")
def train_test_df():
    return sentiment_analysis.get_data()

@pytest.fixture(scope="session")
def train_df(train_test_df):
    return train_test_df[0].head(10)

@pytest.fixture(scope="session")
def test_df(train_test_df):
    return train_test_df[1].head(5)

@pytest.fixture(scope="session")
def train_graphs_labels(train_df,nlp):
    return sentiment_analysis.get_ud_data_from_df(train_df,nlp)

@pytest.fixture(scope="session")
def train_graphs(train_graphs_labels):
    return train_graphs_labels[0]

@pytest.fixture(scope="session")
def train_labels(train_graphs_labels):
    return train_graphs_labels[1]


def test_get_data(train_df, test_df):
    '''test if the data is successfully loaded from kaggles with the correct labels'''
    assert isinstance(train_df,pd.DataFrame)
    assert isinstance(test_df,pd.DataFrame)
    assert not train_df.isna().values.any()
    assert not test_df.isna().values.any()

    assert all(x <= 2 for x in set(train_df["sentiment"]))
    assert all(x <= 2 for x in set(test_df["sentiment"]))


def test_get_graphs_from_ud_df(train_graphs,train_labels):
    '''test if graphs and labels are correctly extracted from a data frame using ud'''
    assert len(train_graphs) == len(train_labels)
    assert isinstance(train_labels,torch.Tensor)
    assert isinstance(train_graphs[0],graph.Graph)


def test_create_objects_for_gnn(train_df,test_df,train_graphs,train_labels, nlp):
    '''test if the correct data objects are being created'''
    train_objects, _ = sentiment_analysis.create_objects_for_gnn(train_df, test_df, nlp)
    first_train_object = train_objects[0]

    first_ud_graph = train_graphs[0]
    first_ud_label = train_labels[0]
    num_features = featurizer.get_features(train_graphs)[0].shape[1]

    assert isinstance(first_train_object,Data)
    assert first_train_object.num_node_features == num_features
    assert first_train_object.num_nodes == len(first_ud_graph.nodes)
    assert first_train_object.num_edges == len(first_ud_graph.edges)
    assert first_train_object.y == first_ud_label
