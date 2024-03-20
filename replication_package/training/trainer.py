import pandas as pd
from joblib import dump
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


def train_model(
    model: BaseEstimator,
    data: pd.DataFrame,
    params: dict,
    data_name: str,
    model_name: str,
):
    # results = []
    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # for itrain, itest in kfold.split(data.index.unique().values):
        # train_index = data.index.unique()[itrain]
        # test_index = data.index.unique()[itest]
        # train = data.loc[train_index]
        # test = data.loc[test_index]
        # Grid Search hyperparam selection
    grid = GridSearchCV(model, params, cv=5, scoring='roc_auc')
    grid.fit(
            data.drop(
                columns=["statistical_parity", "equal_opportunity", "average_odds"]
            ).values,
            data[["statistical_parity", "equal_opportunity", "average_odds"]].values,
        )
    dump(grid.best_estimator_, f"{data_name}_{model_name}.joblib")
    with open(f"best_params_{model_name}.txt", "+a") as f:
        f.write(str(grid.best_params_) + "\n")

        # Validation of the best estimator
        # kfold = KFold(n_splits=5)
        # datasets = test.index.get_level_values(1).unique()
        # for dataset in datasets:
        #     # train = test.iloc[itrain]
        #     validation = test.loc[test.index.get_level_values(1) == dataset]
        # grid.best_estimator_.fit(
        #     train.drop(
        #         columns=["statistical_parity", "equal_opportunity", "average_odds"]
        #     ).values,
        #     train[["statistical_parity", "equal_opportunity", "average_odds"]].values,
        # )
        # predictions = grid.best_estimator_.predict(
        #     test.drop(
        #         columns=["statistical_parity", "equal_opportunity", "average_odds"]
        #     ).values
        # )
        # sp_true = test["statistical_parity"].values
        # eo_true = test["equal_opportunity"].values
        # ao_true = test["average_odds"].values
        # sp_pred = [pred[0] for pred in predictions]
        # eo_pred = [pred[1] for pred in predictions]
        # ao_pred = [pred[2] for pred in predictions]
        # results.append(
        #     pd.DataFrame(
        #         {
        #             "sp_true": sp_true,
        #             "eo_true": eo_true,
        #             "ao_true": ao_true,
        #             "sp_pred": sp_pred,
        #             "eo_pred": eo_pred,
        #             "ao_pred": ao_pred,
        #         }
        #     )
        # )
        # sp_scores_r2.append(accuracy_score(validation["statistical_parity"], sp))
        # eo_scores_r2.append(accuracy_score(validation["equal_opportunity"], eo))
        # ao_scores_r2.append(accuracy_score(validation["average_odds"], ao))
        # sp_scores_mape.append(f1_score(validation["statistical_parity"], sp))
        # eo_scores_mape.append(f1_score(validation["equal_opportunity"], eo))
        # ao_scores_mape.append(f1_score(validation["average_odds"], ao))
        # sp_scores_prec.append(precision_score(validation["statistical_parity"], sp))
        # eo_scores_prec.append(precision_score(validation["equal_opportunity"], eo))
        # ao_scores_prec.append(precision_score(validation["average_odds"], ao))
        # sp_scores_rec.append(recall_score(validation["statistical_parity"], sp))
        # eo_scores_rec.append(recall_score(validation["equal_opportunity"], eo))
        # ao_scores_rec.append(recall_score(validation["average_odds"], ao))
    # r2_scores = pd.DataFrame(
    #     {
    #         "Statistical Parity": sp_scores_r2,
    #         "Equal Opportunity": eo_scores_r2,
    #         "Average Odds": ao_scores_r2,
    #     }
    # )
    # mape_scores = pd.DataFrame(
    #     {
    #         "Statistical Parity": sp_scores_mape,
    #         "Equal Opportunity": eo_scores_mape,
    #         "Average Odds": ao_scores_mape,
    #     }
    # )
    # prec_scores = pd.DataFrame(
    #     {
    #         "Statistical Parity": sp_scores_prec,
    #         "Equal Opportunity": eo_scores_prec,
    #         "Average Odds": ao_scores_prec,
    #     }
    # )
    # rec_scores = pd.DataFrame(
    #     {
    #         "Statistical Parity": sp_scores_rec,
    #         "Equal Opportunity": eo_scores_rec,
    #         "Average Odds": ao_scores_rec,
    #     }
    # )
    # return results
