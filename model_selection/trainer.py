import pandas as pd
from joblib import dump
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.base import BaseEstimator


def train_model(
    model: BaseEstimator,
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict,
    data_name: str,
    model_name: str,
):
    # Grid Search hyperparam selection
    grid = GridSearchCV(model, params, n_jobs=-1, cv=10)
    grid.fit(
        train.drop(
            columns=["statistical_parity", "equal_opportunity", "average_odds"]
        ).values,
        train[["statistical_parity", "equal_opportunity", "average_odds"]].values,
    )
    dump(grid.best_estimator_, f"{data_name}_{model_name}.joblib")
    with open(f"best_params_{model_name}.txt", "+a") as f:
        f.write(str(grid.best_params_) + "\n")

    # Validation of the best estimator
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    sp_scores_r2 = []
    eo_scores_r2 = []
    ao_scores_r2 = []
    sp_scores_mape = []
    eo_scores_mape = []
    ao_scores_mape = []
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
        sp_scores_r2.append(r2_score(validation["statistical_parity"], sp))
        eo_scores_r2.append(r2_score(validation["equal_opportunity"], eo))
        ao_scores_r2.append(r2_score(validation["average_odds"], ao))
        sp_scores_mape.append(
            mean_absolute_percentage_error(validation["statistical_parity"], sp)
        )
        eo_scores_mape.append(
            mean_absolute_percentage_error(validation["equal_opportunity"], eo)
        )
        ao_scores_mape.append(
            mean_absolute_percentage_error(validation["average_odds"], ao)
        )
    r2_scores = pd.DataFrame(
        {
            "Statistical Parity": sp_scores_r2,
            "Equal Opportunity": eo_scores_r2,
            "Average Odds": ao_scores_r2,
        }
    )
    mape_scores = pd.DataFrame(
        {
            "Statistical Parity": sp_scores_mape,
            "Equal Opportunity": eo_scores_mape,
            "Average Odds": ao_scores_mape,
        }
    )
    return r2_scores, mape_scores
