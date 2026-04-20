import torch
from tqdm import tqdm 
from package.graphs.graph import Graph

class GraphBuilder:
    def __init__(self, build_strategy, parse_strategy):
        self.build_strategy = build_strategy
        self.parse_strategy = parse_strategy


    def build_graphs_from_df(self, df):
        labels = torch.tensor(list(df["label"]), dtype=torch.float32)
        graphs = []
        for text in tqdm(list(df["text"])):
            graphs.append(self.make_and_merge_graphs(text))
            
        return graphs, labels
    

    def make_graphs(self, text):
        doc = self.parse_strategy.parse(text)
        graphs = [Graph(sentence, self.build_strategy) for sentence in doc]
        return graphs


    def make_and_merge_graphs(self, text):
        graphs = self.make_graphs(text)
        
        merge_graph = graphs[0]
        for graph in graphs[1:]:
            merge_graph.merge(graph)

        return merge_graph