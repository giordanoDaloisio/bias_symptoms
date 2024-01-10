import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from argparse import ArgumentParser
from trainer import train_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
    args = parser.parse_args()
    data_name = args.data.split("/")[1].split(".")[0]
    params = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "n_estimators": [100, 200, 300],
    }

    data = pd.read_csv(args.data, index_col=[0, 1])

    train, val = train_test_split(data, test_size=0.2)
    r2_scores, mape_scores = train_model(
        XGBRegressor(), train, val, params, data_name, "xgb"
    )
    r2_scores.to_csv(f"scores/xgb_r2_scores_{data_name}.csv")
    mape_scores.to_csv(f"scores/xgb_rmse_scores_{data_name}.csv")
