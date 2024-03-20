import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from trainer import train_model
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
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
    if "sp" in data_name:
        label = "statistical_parity"
    elif "eo" in data_name:
        label = "equal_opportunity"
    else:
        label = "average_odds"

    train_model(
        MLPClassifier(activation="relu", solver="adam"),
        data,
        param_grid,
        data_name,
        "mlp",
    )
