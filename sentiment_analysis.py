import kagglehub
from kagglehub import KaggleDatasetAdapter
import graph_initializer
from featurizer_test import nlp
import torch
from sentiment_analysis_model import SentimentAnalysis
from torch_geometric.data import Data
import featurizer

def get_data():
    train_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "abhi8923shriv/sentiment-analysis-dataset",
        "train.csv",
        pandas_kwargs={"usecols": ["text", "sentiment"], "encoding": "latin1"})
    
    test_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "abhi8923shriv/sentiment-analysis-dataset",
        "test.csv",
        pandas_kwargs={"usecols": ["text", "sentiment"], "encoding": "latin1"})
    
    mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df['sentiment'] = train_df['sentiment'].replace(mapping)
    test_df['sentiment'] = test_df['sentiment'].replace(mapping)
    
    return train_df, test_df


def get_ud_data_from_df(df):
    ud_texts = [nlp(text) for text in df["text"]]
    labels = torch.tensor(df["sentiment"], dtype=torch.float32)
    graphs = [graph_initializer.make_and_merge_graphs(text) for text in ud_texts]
    return graphs, labels



def create_objects_for_gnn(train_df, test_df):
    train_graphs, train_labels = get_ud_data_from_df(train_df)
    test_graphs, test_labels = get_ud_data_from_df(test_df)

    def build(graphs, labels):
        data_objects = []

        features = featurizer.get_features(graphs)
        for i, graph in enumerate(graphs):
            edge_index = torch.tensor(graph.edges_arr, dtype=torch.float32)

            x = torch.tensor(features[i], dtype=torch.float32)
            y = torch.tensor(labels[i], dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)

            data_objects.append(data)

        return data_objects

    train_objects = build(train_graphs, train_labels)
    test_objects = build(test_graphs, test_labels)
    return train_objects, test_objects



def train_model(epochs=200):
    pass


def eval_model():
    pass