import gensim.downloader as gd

class Executor():
    def __init__(self, data_pipeline):
        self.data_pipeline = data_pipeline
        self.embedding_model = data_pipeline.embedding_model

    def run(self,cap=None,compressed_embedding_size=10):
        train_objects, test_objects, num_node_features, num_relations = self.data_pipeline.prepare(cap)
        self.data_pipeline.model.build_model(num_node_features, num_relations, compressed_embedding_size, self.embedding_model.vector_size)
        self.data_pipeline.model.train_model(train_objects)
        results = self.data_pipeline.model.eval_model(test_objects)
        return results