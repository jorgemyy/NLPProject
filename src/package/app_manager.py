from package.models.model_factory import ModelFactory
from package.featurizer_decorator import FeatureExtractorBuilder
from package.executor import Executor

class AppManager():
    def __init__(self):
        self.main_loop()

    def main_loop(self):
        print("\n\n~~~~~~~~~~~NLP PARSE MAIN MENU~~~~~~~~~~~")
        print("1. Load prior results")
        print("2. Run the model")

        possible_options = ['1','2']
        choice = ''
        while choice not in possible_options:
            choice = input("Choice: ")

        if choice == '1':
            self.load_prior_results()
        elif choice == '2':
            self.model_menu()


    def load_prior_results(self):
        print("\n~~~~~~~~~~~PRIOR RESULTS~~~~~~~~~~~")
        print("1. Back")
        choice = input("Choice: ")
        self.main_loop()


    def model_menu(self):
        print("\n\n~~~~~~~~~~MODEL MENU~~~~~~~~~~~")

        model = self.get_model()
        graph_type = self.get_graph_type()
        feature_extractor = self.get_feature_extractor()
        batch_size, epochs, hidden_layer_dim, cap = self.get_parameters()

        executor = Executor(model=model,
                               graph_type=graph_type,
                               feature_extractor=feature_extractor,
                               batch_size=batch_size,
                               epochs=epochs,
                               hidden_layer_dim=hidden_layer_dim)
        
        accuracy = executor.run(cap=cap)
        results = [model.get_name(), graph_type, feature_extractor.get_name()[1:], batch_size, epochs, hidden_layer_dim, accuracy]
        print(results)

    
    def get_model(self):
        print("\nChoose NLP Task")
        print("1. Semantic Analysis")
        print("2. Back")

        possible_nlp_tasks = ['1','2']
        choice = ''
        while choice not in possible_nlp_tasks:
            choice = input("Choice: ")

        model_factory = ModelFactory()
        if choice == '1':
            model = model_factory.createSemModel()
        elif choice == '2':
            self.model_menu()
        
        return model
    

    def get_graph_type(self):
        print("\nChoose Graph Type")
        print("1. Universal Dependency (UD)")
        print("2. Abstract Meaning Representation (AMR)")
        print("3. Back")

        possible_graph_types = ['1','2','3']
        choice = ''
        while choice not in possible_graph_types:
            choice = input("Choice: ")

        if choice == '1':
            graph_type = 'ud'
        elif choice == '2':
            graph_type = 'amr'
        elif choice == '3':
            self.model_menu()
        
        return graph_type


    def get_feature_extractor(self):
        print("\nCustomize Features")
        print("Choose features, separated by a space")
        print("1. Normalized Word ID (position in the sentence)")
        print("2. Normalized Distance from Root")
        print("3. Multi-hot Encoded Incoming Edge Labels")
        print("4. Multi-hot Encoded Outgoing Edge Labels")
        print("5. Word / Concept Embedding")
        print("6. Back")

        feature_choices_options = {'1': FeatureExtractorBuilder.add_id,
                                   '2': FeatureExtractorBuilder.add_root_distance,
                                   '3': FeatureExtractorBuilder.add_incoming_labels,
                                   '4': FeatureExtractorBuilder.add_outgoing_labels,
                                   '5': FeatureExtractorBuilder.add_embedding}

        feature_choices_list = []
        invalid = True
        while invalid:
            feature_choices = input("Choice: ")
            if feature_choices == '6':
                self.model_menu()
                return
            
            feature_choices_list = feature_choices.split()
            invalid = [c for c in feature_choices_list if c not in feature_choices_options]

        feature_extractor = FeatureExtractorBuilder()
        for feature_choice in feature_choices_list:
            add_feature = feature_choices_options[feature_choice]
            add_feature(feature_extractor)
        feature_extractor = feature_extractor.build()

        return feature_extractor
        

    def get_parameters(self):
        print("\nChoose parameters (hit enter for default)")

        while True:
            batch_size_input = input("Batch size (default 16): ")
            if batch_size_input == '':
                batch_size = 16
                break
            try:
                batch_size = int(batch_size_input)
                break
            except ValueError:
                continue

        while True:
            epochs_input = input("Number of epochs (default 200): ")
            if epochs_input == '':
                epochs = 200
                break
            try:
                epochs = int(epochs_input)
                break
            except ValueError:
                continue

        while True:
            hidden_dim_input = input("Hidden layer dimension (default 16): ")
            if hidden_dim_input == '':
                hidden_layer_dim = 16
                break
            try:
                hidden_layer_dim = int(hidden_dim_input)
                break
            except ValueError:
                continue

        while True:
            cap_input = input("Cap the dataset (default use the entire dataset): ")
            if cap_input == '':
                cap = None
                break
            try:
                cap = int(cap_input)
                break
            except ValueError:
                continue

        return batch_size, epochs, hidden_layer_dim, cap