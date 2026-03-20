from graph import Node, Edge
from graph import Graph
import gensim.downloader as gd
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np

class Featurizer:
    def __init__(self, graph_type):
        self.embedding_model = gd.load("word2vec-google-news-300")
        self.graph_type = graph_type


    def get_features(self, ud_document):
        if self.graph_type == "ud":
            return self.get_features_from_ud(ud_document)


    def get_features_from_ud(self, stanza_doc):
        self.make_graphs(stanza_doc)
        self.fit_one_hot_encoding()
        return self.get_all_features()
    

    def make_graphs(self, ud_doc):
        ud_sentences = ud_doc.sentences
        self.graphs = [self.get_graph_from_ud(sentence) for sentence in ud_sentences]


    def get_graph_from_ud(self, ud_sentence):
        graph = Graph()
        for word in ud_sentence.words:
            newNode = Node(id = word.id,
                           text = word.text,
                           upos = word.upos,
                           xpos = word.xpos,
                           head = word.head
                           )
            graph.add_Node(newNode)
        # edges need to be implemented
        return graph


    def get_features_from_amr(amr_graph):
        pass #not implemented


    def get_pos_tags(self):
        upos = []
        xpos = []
        for graph in self.graphs:
            for node in graph.nodes:
                upos.append(node.upos)
                xpos.append(node.xpos)
        return upos, xpos 
    

    def fit_one_hot_encoding(self):
        self.upos_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.xpos_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        upos_data, xpos_data = self.get_pos_tags()
        upos_data_2d, xpos_data_2d = np.array(upos_data).reshape(-1,1), np.array(xpos_data).reshape(-1,1)

        self.upos_encoder.fit(upos_data_2d)
        self.xpos_encoder.fit(xpos_data_2d)


    def one_hot_encode(self, node):
        upos_vec = self.upos_encoder.transform([[node.upos]])[0]
        xpos_vec = self.xpos_encoder.transform([[node.xpos]])[0]

        return (torch.tensor(upos_vec, dtype=torch.float32),
        torch.tensor(xpos_vec, dtype=torch.float32))
    

    def get_word_embeddings(self,node,dim=300):
        word = node.text.lower()
        if word in self.embedding_model:
            return torch.tensor(self.embedding_model[word], dtype=torch.float32)
        else:
            return torch.zeros(dim, dtype=torch.float32)
        

    def get_features_from_graph(self,graph):
        node_features = []

        for node in graph.nodes:
            normalized_id = node.id / len(graph.nodes)
            normalized_head = node.head / len(graph.nodes) if node.head != 0 else 0
            structural_features = torch.tensor([normalized_id, normalized_head], dtype=torch.float32)

            upos_vec, xpos_vec = self.one_hot_encode(node)

            word_embedding = self.get_word_embeddings(node)
            
            feature_vec = torch.cat([structural_features, upos_vec, xpos_vec, word_embedding])
            node_features.append(feature_vec)

        return torch.stack(node_features)
    

    def get_all_features(self):
        features = [self.get_features_from_graph(graph) for graph in self.graphs]
        return features