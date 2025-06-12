import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from trainer import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
    parser.add_argument("--no_dsp", action="store_true")
    args = parser.parse_args()

    data_name = args.data.split("/")[1].split(".")[0]
    params = {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 50, 100, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    data = pd.read_csv(args.data, index_col=[0, 1])

    print(f"Training Random Forest on {data_name} sp")
    train_model(RandomForestClassifier(), data, params, data_name, "rf", "statistical_parity", args.no_dsp)
    print(f"Training Random Forest on {data_name} eo")
    train_model(RandomForestClassifier(), data, params, data_name, "rf", "equal_opportunity", args.no_dsp)
    print(f"Training Random Forest on {data_name} ao")
    train_model(RandomForestClassifier(), data, params, data_name, "rf", "average_odds", args.no_dsp)
