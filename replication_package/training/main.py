import pandas as pd
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
from trainer import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
    parser.add_argument("--no_dsp", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument("--unbalanced", action='store_true')
    parser.add_argument("--clean", action='store_true')
    parser.add_argument("--noisy", action='store_true')
    args = parser.parse_args()
    data_name = args.data.split("/")[1].split(".")[0]

    if args.model == "xgb":
        params = {
            "learning_rate": [0.05, 0.01, 0.2, 0.3],
            "max_depth": [3, 4, 5, 6],
            "gamma": [0, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        model = XGBClassifier()
    elif args.model == "mlp":
        params = {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50, 25)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "max_iter": [200, 300, 500],
        }
        model = MLPClassifier(activation="relu", solver="adam")
    else:
        params = {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 50, 100, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
        model = RandomForestClassifier()

    data = pd.read_csv(args.data, index_col=[0, 1])

    print(f"Training {args.model} on {data_name} data")
    train_model(model, data, params, data_name, args.model, "statistical_parity", args.no_dsp, args.balanced, args.unbalanced, args.clean, args.noisy)
    print(f"Training {args.model} on {data_name} data for equal opportunity")
    train_model(model, data, params, data_name, args.model, "equal_opportunity", args.no_dsp, args.balanced, args.unbalanced, args.clean, args.noisy)
    print(f"Training {args.model} on {data_name} data for average odds")
    train_model(model, data, params, data_name, args.model, "average_odds", args.no_dsp, args.balanced, args.unbalanced, args.clean, args.noisy)
