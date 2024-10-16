import pandas as pd
from xgboost import XGBClassifier
from argparse import ArgumentParser
from trainer import train_model
from sklearn.model_selection import train_test_split
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
    args = parser.parse_args()
    data_name = args.data.split("/")[1].split(".")[0]
    params = {
        "learning_rate": [0.05, 0.01, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    data = pd.read_csv(args.data, index_col=[0, 1])

    train_model(XGBClassifier(), data, params, data_name, "xgb")
