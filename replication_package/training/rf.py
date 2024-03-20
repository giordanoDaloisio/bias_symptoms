import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from trainer import train_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
    args = parser.parse_args()

    data_name = args.data.split("/")[1].split(".")[0]
    params = {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 50, 100, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    }
    data = pd.read_csv(args.data, index_col=[0, 1])

    train_model(RandomForestClassifier(), data, params, data_name, "rf")
