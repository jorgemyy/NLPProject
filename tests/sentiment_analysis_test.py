import pandas as pd
import torch

import src.package.graph as graph
import src.package.sentiment_analysis as sentiment_analysis

def test_load_data():
    '''test if the data is successfully loaded from kaggles with the correct labels'''
    train_df, test_df = sentiment_analysis.load_data()
    assert isinstance(train_df,pd.DataFrame)
    assert isinstance(test_df,pd.DataFrame)

    assert set(train_df["sentiment"]) == {0,1,2}
    assert set(test_df["sentiment"]) == {0,1,2}


def test_get_graphs_from_ud_df():
    '''test if graphs and labels are correctly extracted from a data frame using ud'''
    train_df, _ = sentiment_analysis.load_data()
    train_graphs, train_labels = sentiment_analysis.get_ud_data_from_df(train_df)
    assert len(train_graphs) == len(train_labels)
    assert isinstance(train_labels,torch.Tensor)
    assert isinstance(train_graphs[0],graph.Graph)


def test_create_objects_for_gnn():
    pass
