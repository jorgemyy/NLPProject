import torch
from package.features import featurizer

class FeatureExtractor():
    def featurize(self, node, graph, context):
        return []
    
    def get_name(self):
        return ''


class FeatureExtractorBuilder():
    def __init__(self):
        self.extractor = FeatureExtractor()

    def add_id(self):
        self.extractor = IDDecorator(self.extractor)
        return self
    
    def add_root(self):
        self.extractor = RootDecorator(self.extractor)
        return self
    
    def add_embedding(self):
        self.extractor = EmbeddingDecorator(self.extractor)
        return self
    
    def add_type(self):
        self.extractor = NodeTypeDecorator(self.extractor)
        return self
    
    def add_neg(self):
        self.extractor = NegationDecorator(self.extractor)
        return self
    
    def build(self):
        return self.extractor


class FeatureDecorator(FeatureExtractor):
    def __init__(self,wrapped):
        self.wrapped = wrapped

    def featurize(self, node, graph, context):
        return self.wrapped.featurize(node, graph, context)
    
    def get_name(self):
        return self.wrapped.get_name()


class IDDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        normalized_id = torch.tensor([(node.id+1) / len(graph.nodes)], dtype=torch.float32)
        features = super().featurize(node, graph, context)
        features.append(normalized_id)
        return features
    
    def get_name(self):
        return super().get_name() + '/ID'
    

class RootDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        normalized_distance_from_root = torch.tensor([(node.root+1) / len(graph.nodes)], dtype=torch.float32)
        features = super().featurize(node, graph, context)
        features.append(normalized_distance_from_root)
        return features
    
    def get_name(self):
        return super().get_name() + '/root'
    

class EmbeddingDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        word_embedding = context.get_word_embeddings(node)
        features = super().featurize(node, graph, context)
        features.append(word_embedding)
        return features
    
    def get_name(self):
        return super().get_name() + '/embedding'
    

class NodeTypeDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        encoding = context.one_hot_encode_node_type(node.node_type)
        features = super().featurize(node, graph, context)
        features.append(encoding)
        return features
    
    def get_name(self):
        return super().get_name() + '/node_type'
    
    
class NegationDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        features = super().featurize(node, graph, context)
        features.append(torch.tensor([node.negated], dtype=torch.float32))
        return features 

    def get_name(self):
        return super().get_name() + '/negation'


'''
class IncomingEdgeLabelDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        incoming_edge_label_vecs = context.one_hot_encode(node.incoming_edge_labels)
        num_incoming_edges = torch.tensor([sum(incoming_edge_label_vecs)], dtype=torch.float32)
        features = super().featurize(node,graph,context)
        features.extend([incoming_edge_label_vecs, num_incoming_edges])
        return features
    
    def get_name(self):
        return super().get_name() + '/in_labels'
    

class OutgoingEdgeLabelDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        outgoing_edge_label_vecs = context.one_hot_encode(node.outgoing_edge_labels)
        num_outgoing_edges = torch.tensor([sum(outgoing_edge_label_vecs)], dtype=torch.float32) 
        features = super().featurize(node,graph,context)
        features.extend([outgoing_edge_label_vecs, num_outgoing_edges])
        return features
    
    def get_name(self):
        return super().get_name() + '/out_labels'

'''