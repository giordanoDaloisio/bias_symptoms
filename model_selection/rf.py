import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from trainer import train_model



if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("-d", "--data")
    # args = parser.parse_args()

    data = "data/all_features.csv"
    data_name = data.split("/")[1].split(".")[0]
    params = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 50, 100, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['log2', 'sqrt'],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error']
    }
    data = pd.read_csv(data, index_col=[0, 1])

    train, val = train_test_split(data, test_size=0.2)
    r2_scores, mape_scores = train_model(
        RandomForestRegressor(), train, val, params, data_name, "rf"
    )
    r2_scores.to_csv(f"scores/xgb_r2_scores_{data_name}.csv")
    mape_scores.to_csv(f"scores/xgb_rmse_scores_{data_name}.csv")