import csv
import pandas as pd 
import time

from package.models.model_factory import ModelFactory
from package.features.featurizer_decorator import FeatureExtractorBuilder
from package.executor import Executor
from package.data_pipeline import DataPipeline
from package.graphs.graph_builder_factory import GraphBuilderFactory

class AppManager():
    def __init__(self, saved_state_manager, summary_file = 'output.csv', summary_file_dir = 'src/'):
        self.saved_state_manager = saved_state_manager
        self.summary_file = summary_file_dir+summary_file

    def main_loop(self):
        while True:
            print("\n\n~~~~~~~~~~~NLP PARSE MAIN MENU~~~~~~~~~~~")
            print("1. Load prior results")
            print("2. Run the model")
            print("3. Exit")

            possible_options = ['1','2','3']
            choice = ''
            while choice not in possible_options:
                choice = input("Choice: ")

            if choice == '1':
                self.load_prior_results()
            elif choice == '2':
                self.model_menu()
            elif choice == '3':
                return


    def load_prior_results(self):
        print("\n~~~~~~~~~~~PRIOR RESULTS~~~~~~~~~~~")
        print("1. Show sorted results")
        print("2. Show all results")
        print("3. Show most recent results")
        print("4. Back")

        menu_options = ['1','2','3','4']
        choice = ''
        while choice not in menu_options:
            choice = input("Choice: ")

        df = pd.read_csv(self.summary_file)
        printed_df = None
        if choice == '1':
            possible_sort_options = []
            num_columns = len(df.columns)

            print("\nSort by any of the following?")
            print(f"You can choose multiple options, press {num_columns+2} when finished\n")

            for (x, column) in enumerate(df):
                print(str(x+1) + ". " + column)
                possible_sort_options.append(str(x+1))
    
            print(str(num_columns+1) + ". " + "Done Selecting")
            print(str(num_columns+2) + ". " + "Back")

            choices_list = []
            mode_list = []
            while True:
                choice = ''
                while choice not in possible_sort_options and choice not in [str(num_columns+1), str(num_columns+2)]:
                    choice = input("Choice: ")

                if choice == str(num_columns+1):
                    if len(choices_list) > 0:
                        break
                    print("Must add choices")

                elif choice == str(num_columns+2):
                    choices_list = []
                    return
                
                else:
                    choices_list.append(df.columns[int(choice)-1])

                    print("\n1. Ascending")
                    print("2. Descending")

                    possible_mode_options = ['1','2']
                    mode = ''
                    while mode not in possible_mode_options:
                        mode = input("Choice: ")

                    mode_list.append(True if mode == '1' else False) 
                    print("Added sort parameter\n")

            sorted_df = df.sort_values(by=choices_list,ascending=mode_list)
            printed_df = sorted_df.head(10)

        elif choice == '2':
            printed_df = df
        elif choice == '3':
            printed_df = df.head(10)
        elif choice == '4':
            return
        
        print("---------------------------")
        print(printed_df)
        
        print("\nSelect any row to look deeper? Press anything else to go back")
        row = -1
        while row < 0 or row >= len(printed_df['ID']):
            row_choice = input("Choice: ")
            try: 
                row = int(row_choice)
            except: 
                return

        run_id = printed_df['ID'].iloc[row]
        self.saved_state_manager.show_run_details(run_id)
        
        return


    def model_menu(self):
        print("\n\n~~~~~~~~~~MODEL MENU~~~~~~~~~~~")

        model, batch_size, epochs, hidden_layer_dim, cap = self.get_model() 
        if model is None:
            return 
        
        graph_type = self.get_graph_type()
        if graph_type is None:
            return 
        
        feature_extractor = self.get_feature_extractor()
        if feature_extractor is None:
            return

        graph_builder_factory = GraphBuilderFactory()
        graph_builder = None
        if graph_type == 'amr':
            graph_builder = graph_builder_factory.create_AMR_Builder()
            
        elif graph_type == 'ud':
            graph_builder = graph_builder_factory.create_UD_Builder()

        if graph_builder is None:
            return

        pipeline = DataPipeline(model=model,
                                graph_builder=graph_builder,
                                feature_extractor=feature_extractor,
                                embedding_model=None)

        executor = Executor(pipeline)
        
        start = time.time()
        accuracy, fscore, cm = executor.run(cap=cap)
        end = time.time()

        cap = "None" if cap==None else cap
        df = pd.read_csv(self.summary_file)
        run_id = df['ID'].tail(1) + 1 if not df.empty else 0
        summary_results_dict = {
            "run_id": int(run_id),
            "model_name": model.get_name(),
            "graph_type": graph_type,
            "cap": cap,
            "accuracy": accuracy,
            "f1": fscore
        }
        summary_results = summary_results_dict.values()

        detailed_results = {
            "features": feature_extractor.get_name()[1:],
            "batch_size": batch_size,
            "epochs": epochs,
            "hidden_layer_dim": hidden_layer_dim,
            "confusion_matrix": cm,
            "run_time": end-start
        }

        self.saved_state_manager.save(summary_results_dict, detailed_results)

        with open('src/output.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(summary_results)
        print(f"Results: {summary_results}")
        print("Successfully wrote results to file")
        return

    
    def get_model(self):
        print("\nChoose NLP Task")
        print("1. Semantic Analysis")
        print("2. Back")

        possible_nlp_tasks = ['1','2']
        choice = ''
        while choice not in possible_nlp_tasks:
            choice = input("Choice: ")

        model_factory = ModelFactory()
        model = None
        batch_size = None
        epochs = None
        hidden_layer_dim = None
        cap = None
        if choice == '1':
            batch_size, epochs, hidden_layer_dim, cap = self.get_parameters()
            model = model_factory.createSemModel(batch_size, epochs, hidden_layer_dim)

        return model, batch_size, epochs, hidden_layer_dim, cap
    

    def get_graph_type(self):
        print("\nChoose Graph Type")
        print("1. Universal Dependency (UD)")
        print("2. Abstract Meaning Representation (AMR)")
        print("3. Back")

        possible_graph_types = ['1','2','3']
        choice = ''
        while choice not in possible_graph_types:
            choice = input("Choice: ")

        graph_type = None
        if choice == '1':
            graph_type = 'ud'
        elif choice == '2':
            graph_type = 'amr'
        
        return graph_type


    def get_feature_extractor(self):
        print("\nCustomize Features")
        print("You can choose multiple options, press 6 when finished")
        print("1. Normalized Word ID (position in the sentence)")
        print("2. Normalized Distance from Root")
        print("3. Multi-hot Encoded Incoming Edge Labels")
        print("4. Multi-hot Encoded Outgoing Edge Labels")
        print("5. Word / Concept Embedding")
        print("6. Done Selecting")
        print("7. Back")

        feature_choices_options = {'1': FeatureExtractorBuilder.add_id,
                                   '2': FeatureExtractorBuilder.add_root_distance,
                                   '3': FeatureExtractorBuilder.add_incoming_labels,
                                   '4': FeatureExtractorBuilder.add_outgoing_labels,
                                   '5': FeatureExtractorBuilder.add_embedding}

        feature_choices_list = []
        while True:
            feature_choice = ''
            while feature_choice not in feature_choices_options and feature_choice not in ['6','7']:
                feature_choice = input("Choice: ")

            if feature_choice == '6':
                if len(feature_choices_list) > 0:
                    break
                print("Must add features")

            elif feature_choice == '7':
                feature_choices_list = []
                return None
        
            else:
                print("Added feature")
                feature_choices_list.append(feature_choice)

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