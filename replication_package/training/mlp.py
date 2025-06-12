import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from trainer import *
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
    parser.add_argument("--no_dsp", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(args.data, index_col=[0, 1])
    data_name = args.data.split("/")[1].split(".")[0]

    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50, 25)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter": [200, 300, 500],
    }

    print(f"Training MLP on {data_name} data")
    train_model(
        MLPClassifier(activation="relu", solver="adam"),
        data,
        param_grid,
        data_name,
        'mlp', 
        'statistical_parity',
        args.no_dsp
    )
    train_model(
        MLPClassifier(activation="relu", solver="adam"),
        data,
        param_grid,
        data_name,
        'mlp',
        'equal_opportunity',
        args.no_dsp

    )
    train_model(
        MLPClassifier(activation="relu", solver="adam"),
        data,
        param_grid,
        data_name,
        'mlp',
        'average_odds',
        args.no_dsp
    )
