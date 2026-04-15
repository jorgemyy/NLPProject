import kagglehub
from kagglehub import KaggleDatasetAdapter
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

import package.featurizer as featurizer
import package.graph_initializer as graph_initializer

def get_data():
    full_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "abhi8923shriv/sentiment-analysis-dataset",
        "train.csv",
        pandas_kwargs={"usecols": ["text", "sentiment"], "encoding": "latin1"}).sample(frac=1).dropna()
    
    mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    full_df['sentiment'] = full_df['sentiment'].replace(mapping)
    
    return full_df


def get_ud_data_from_df(df,nlp):
    ud_texts = []
    for text in tqdm(list(df["text"])):
        ud_texts.append(nlp(text))

    labels = torch.tensor(list(df["sentiment"]), dtype=torch.float32)
    graphs = [graph_initializer.make_and_merge_graphs(text) for text in ud_texts]
    return graphs, labels


def create_objects_for_gnn(df, nlp):
    graphs, labels = get_ud_data_from_df(df, nlp)

    data_objects = []

    features = featurizer.get_features(graphs)
    for i, graph in enumerate(graphs):
        edge_index = torch.tensor(graph.edges_arr, dtype=torch.long)

        x = features[i]
        y = labels[i]

        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)

        data_objects.append(data)

    return data_objects


def train_model(model, train_data, epochs = 200, batch_size = 16):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out,batch.y)

            #back prop
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(loader):.4f}")


def eval_model(model, test_data, batch_size = 16):
    model.eval()
    loader = DataLoader(test_data, batch_size=batch_size)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            pred = model(batch).argmax(dim=1)
            all_preds.append(pred)
            all_labels.append(batch.y.long())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy