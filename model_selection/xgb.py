import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from argparse import ArgumentParser


def train_model(model, train, test):
    # Grid Search hyperparam selection
    grid = GridSearchCV(model, params, n_jobs=-1, cv=10)
    grid.fit(
        train.drop(
            columns=["statistical_parity", "equal_opportunity", "average_odds"]
        ).values,
        train[["statistical_parity", "equal_opportunity", "average_odds"]].values,
    )
    dump(grid.best_estimator_, f"{data_name}_xgb.joblib")
    with open(f"best_params.txt", "+a") as f:
        f.write(str(grid.best_params_) + "\n")

    # Validation of the best estimator
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    sp_scores = []
    eo_scores = []
    ao_scores = []
    for itrain, itest in kfold.split(test):
        train = test.iloc[itrain]
        validation = test.iloc[itest]
        grid.best_estimator_.fit(
            train.drop(
                columns=["statistical_parity", "equal_opportunity", "average_odds"]
            ).values,
            train[["statistical_parity", "equal_opportunity", "average_odds"]].values,
        )
        predictions = grid.best_estimator_.predict(
            validation.drop(
                columns=["statistical_parity", "equal_opportunity", "average_odds"]
            ).values,
        )
        sp = [pred[0] for pred in predictions]
        eo = [pred[1] for pred in predictions]
        ao = [pred[2] for pred in predictions]
        sp_scores.append(r2_score(validation["statistical_parity"], sp))
        eo_scores.append(r2_score(validation["equal_opportunity"], eo))
        ao_scores.append(r2_score(validation["average_odds"], ao))
    return sp_scores, eo_scores, ao_scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
    args = parser.parse_args()
    data_name = args.data.split("/")[1].split(".")[0]
    params = {
        "gamma": range(5),
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.2],
        "reg_lambda": [0, 0.1, 0.2],
    }

    data = pd.read_csv(args.data, index_col=[0, 1])

    train, val = train_test_split(data, test_size=0.2)
    sp_scores, eo_scores, ao_scores = train_model(XGBRegressor(), train, val)
    pd.DataFrame(
        {
            "Statistical Parity": sp_scores,
            "Equal Opportunity": eo_scores,
            "Average Odds": ao_scores,
        }
    ).to_csv(f"scores/xgb_scores_{data_name}.csv")
