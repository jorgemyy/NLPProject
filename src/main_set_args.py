import time
import pandas as pd
import csv

from nlp_core.executor import Executor
from nlp_core.data_pipeline import DataPipeline
from nlp_core.models.model_factory import ModelFactory
from nlp_core.graphs.graph_builder_factory import GraphBuilderFactory
from nlp_core.features.featurizer_decorator import FeatureExtractorBuilder
from nlp_core.saved_state_manager import SavedStateManager
import argparse

def main(args):
    graph_builder_factory = GraphBuilderFactory()
    if args.graph_type == 'amr':
        graph_builder = graph_builder_factory.create_AMR_Builder()
    elif args.graph_type == 'ud':
        graph_builder = graph_builder_factory.create_UD_Builder()

    model_factory = ModelFactory()
    model = model_factory.createSemModel(args.batch_size, args.epochs, args.hidden_layer_dim)

    feature_choices_options = {'id': FeatureExtractorBuilder.add_id,
                                'root': FeatureExtractorBuilder.add_root,
                                'type': FeatureExtractorBuilder.add_type,
                                'neg': FeatureExtractorBuilder.add_neg,
                                'emb': FeatureExtractorBuilder.add_embedding}

    features = args.feat
    if 'all' in features:
        features = feature_choices_options.keys()
        
    feature_extractor = FeatureExtractorBuilder()
    for feature_choice in features:
        add_feature = feature_choices_options[feature_choice]
        add_feature(feature_extractor)
    feature_extractor = feature_extractor.build()

    pipeline = DataPipeline(model=model,
                                graph_builder=graph_builder,
                                feature_extractor=feature_extractor,
                                embedding_model=None)

    executor = Executor(pipeline)
    start = time.time()
    accuracy, fscore, cm = executor.run(args.cap,args.compressed_embedding_size)
    end = time.time()

    cap = "None" if args.cap==None else args.cap
    summary_file = 'results/output.csv'

    df = pd.read_csv(summary_file)
    run_id = df['ID'].tail(1) + 1 if not df.empty else 0
    summary_results_dict = {
        "run_id": int(run_id),
        "model_name": model.get_name(),
        "graph_type": args.graph_type,
        "cap": cap,
        "accuracy": accuracy,
        "f1": fscore
    }
    summary_results = summary_results_dict.values()

    detailed_results = {
        "features": feature_extractor.get_name()[1:],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "hidden_layer_dim": args.hidden_layer_dim,
        "embedding_feature_size": args.compressed_embedding_size,
        "confusion_matrix": cm,
        "run_time": end-start
    }

    saved_state_manager = SavedStateManager()
    saved_state_manager.save(summary_results_dict, detailed_results)

    with open(summary_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(summary_results)
    print(f"Results: {summary_results}")
    print("Successfully wrote results to file")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_type", choices=['ud','amr'],type=str, default='ud')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_layer_dim", type=int, default=128)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--compressed_embedding_size", type=int, default=10)
    parser.add_argument('--feat', nargs='+', choices=['id','root','type','neg','emb','all'], required=True)
    args = parser.parse_args()
    main(args)