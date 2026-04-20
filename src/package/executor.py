class Executor():
    def __init__(self, data_pipeline):
        self.data_pipeline = data_pipeline

    def run(self,cap=None):
        train_objects, test_objects, num_node_features = self.data_pipeline.prepare(cap)
        self.data_pipeline.model.build_model(num_node_features)
        self.data_pipeline.model.train_model(train_objects)
        return self.data_pipeline.model.eval_model(test_objects)