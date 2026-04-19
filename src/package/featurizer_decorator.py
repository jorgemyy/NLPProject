import torch

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
    
    def add_root_distance(self):
        self.extractor = RootDistanceDecorator(self.extractor)
        return self
    
    def add_incoming_labels(self):
        self.extractor = IncomingEdgeLabelDecorator(self.extractor)
        return self
    
    def add_outgoing_labels(self):
        self.extractor = OutgoingEdgeLabelDecorator(self.extractor)
        return self
    
    def add_embedding(self):
        self.extractor = EmbeddingDecorator(self.extractor)
        return self
    
    def build(self):
        return self.extractor


class FeatureDecorator(FeatureExtractor):
    def __init__(self,wrapped):
        self.wrapped = wrapped

    def featurize(self, node, graph, context):
        return self.wrapped.featurize(node,graph,context)
    
    def get_name(self):
        return self.wrapped.get_name()


class IDDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        normalized_id = torch.tensor([(node.id+1) / len(graph.nodes)], dtype=torch.float32)
        features = super().featurize(node,graph,context)
        features.append(normalized_id)
        return features
    
    def get_name(self):
        return super().get_name() + '/ID'
    

class RootDistanceDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        normalized_distance_from_root = torch.tensor([(node.id - node.root) / len(graph.nodes)], dtype=torch.float32)
        features = super().featurize(node,graph,context)
        features.append(normalized_distance_from_root)
        return features
    
    def get_name(self):
        return super().get_name() + '/root'
    

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
    

class EmbeddingDecorator(FeatureDecorator):
    def featurize(self, node, graph, context):
        word_embedding = context.get_word_embeddings(node)
        features = super().featurize(node,graph,context)
        features.append(word_embedding)
        return features
    
    def get_name(self):
        return super().get_name() + '/embedding'