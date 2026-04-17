import pandas as pd
import torch
from torch_geometric.data import Data
import pytest
from sklearn.model_selection import train_test_split

from package import graph
from package.models import sentiment_analysis
from package import featurizer
from package.models.model_factory import ModelFactory
 
@pytest.fixture(scope="session")
def full_df():
    return sentiment_analysis.get_data()

@pytest.fixture(scope="session")
def ud_data_objects_small(full_df, nlp):
    return sentiment_analysis.create_objects_for_gnn(full_df.head(20), 'ud', nlp=nlp)

@pytest.fixture(scope="session")
def ud_data_objects_large(full_df, nlp):
    return sentiment_analysis.create_objects_for_gnn(full_df.head(200), 'ud', nlp=nlp)

@pytest.fixture(scope="session")
def amr_data_objects_small(full_df, stog):
    return sentiment_analysis.create_objects_for_gnn(full_df.head(20), 'amr', stog=stog)

@pytest.fixture(scope="session")
def amr_data_objects_large(full_df, stog):
    return sentiment_analysis.create_objects_for_gnn(full_df.head(200), 'amr', stog=stog)


def test_get_data(full_df):
    '''test if the data is successfully loaded from kaggles with the correct labels'''
    assert type(full_df) == pd.DataFrame
    assert not full_df.isna().values.any()

    assert all(x <= 2 for x in set(full_df["sentiment"]))
 

def test_get_graphs_from_ud_df(full_df, nlp):
    '''test if graphs and labels are correctly extracted from a data frame using ud'''
    num_to_look_at = 5
    graphs, labels = sentiment_analysis.get_ud_data_from_df(full_df.head(num_to_look_at),nlp)
    assert len(graphs) == len(labels)
    assert len(graphs) == num_to_look_at
    assert type(labels) == torch.Tensor
    assert type(graphs[0]) == graph.Graph


def test_get_graphs_from_amr_df(full_df, stog):
    '''test if graphs and labels are correctly extracted from a data frame using ud'''
    num_to_look_at = 5
    graphs, labels = sentiment_analysis.get_amr_data_from_df(full_df.head(num_to_look_at),stog)
    assert len(graphs) == len(labels)
    assert len(graphs) == num_to_look_at
    assert type(labels) == torch.Tensor
    assert type(graphs[0]) == graph.Graph


def test_create_objects_for_gnn(full_df, nlp, stog):
    '''test if the correct data objects are being created'''
    ud_graphs, ud_labels = sentiment_analysis.get_ud_data_from_df(full_df.head(1),nlp)
    amr_graphs, amr_labels = sentiment_analysis.get_amr_data_from_df(full_df.head(1),stog)

    assert type(ud_graphs[0]) == graph.Graph
    assert type(amr_graphs[0]) == graph.Graph

    first_ud_graph = ud_graphs[0]
    first_ud_label = ud_labels[0]
    first_amr_graph = amr_graphs[0]
    first_amr_label = amr_labels[0]
    ud_num_features = featurizer.get_features(ud_graphs)[0].shape[1]
    amr_num_features = featurizer.get_features(amr_graphs)[0].shape[1]

    ud_first_data_object = sentiment_analysis.create_objects_for_gnn(full_df.head(1), "ud", nlp=nlp)[0]
    amr_first_data_object = sentiment_analysis.create_objects_for_gnn(full_df.head(1), "amr", stog=stog)[0]

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


def test_ud_model(ud_data_objects_small):
    '''test train and eval'''
    train_objects, test_objects = train_test_split(ud_data_objects_small, test_size=0.2)

    model_factory = ModelFactory(num_node_features=train_objects[0].num_node_features)
    sem_model = model_factory.createSemModel(num_classes=3)

    sentiment_analysis.train_model(sem_model, train_objects)
    acc = sentiment_analysis.eval_model(sem_model, test_objects) # right now, test and train would have shape mismatch - CHANGE THIS

    assert acc <= 1
    assert acc >= 0


def test_amr_model(amr_data_objects_small):
    '''test train and eval'''
    train_objects, test_objects = train_test_split(amr_data_objects_small, test_size=0.2)

    model_factory = ModelFactory(num_node_features=train_objects[0].num_node_features)
    sem_model = model_factory.createSemModel(num_classes=3)

    sentiment_analysis.train_model(sem_model, train_objects)
    acc = sentiment_analysis.eval_model(sem_model, test_objects) # right now, test and train would have shape mismatch - CHANGE THIS

    assert acc <= 1
    assert acc >= 0
