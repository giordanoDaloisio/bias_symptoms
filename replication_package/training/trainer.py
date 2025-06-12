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
    label: str,
    no_dsp: bool = False,
    balanced: bool = False,
    unbalanced: bool = False,
    clean: bool = False,
    noisy: bool = False,
):
    grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
    cols_to_drop = ["statistical_parity", "equal_opportunity", "average_odds"]
    if no_dsp:
        cols_to_drop.append("pos_prob")
    if balanced:
        bal_data = pd.read_csv("stats/balanced.csv")
        bal_data['File'] = bal_data['File'].str.replace('.csv', '')
        print(bal_data)
        data = data[data.index.get_level_values(0).isin(bal_data['File'])]
        print(len(data.index))
    elif unbalanced:
        unbal_data = pd.read_csv("stats/unbalanced.csv")
        unbal_data['File'] = unbal_data['File'].str.replace('.csv', '')
        data = data[data.index.get_level_values(0).isin(unbal_data['File'])]
    elif clean:
        clean_data = pd.read_csv("stats/clean.csv")
        clean_data['File'] = clean_data['File'].str.replace('.csv', '')
        data = data[data.index.get_level_values(0).isin(clean_data['File'])]
    elif noisy:
        noisy_data = pd.read_csv("stats/noisy.csv")
        noisy_data['File'] = noisy_data['File'].str.replace('.csv', '')
        data = data[data.index.get_level_values(0).isin(noisy_data['File'])]
    grid.fit(
            data.drop(
                columns=cols_to_drop
            ).values,
            data[label].values,
        )
    if no_dsp:
        dump(grid.best_estimator_, f"{data_name}_{model_name}_{label}_nodsp.joblib")
    elif balanced:
        dump(grid.best_estimator_, f"{data_name}_{model_name}_{label}_balanced.joblib")
    elif unbalanced:
        dump(grid.best_estimator_, f"{data_name}_{model_name}_{label}_unbalanced.joblib")
    elif clean:
        dump(grid.best_estimator_, f"{data_name}_{model_name}_{label}_clean.joblib")
    elif noisy:
        dump(grid.best_estimator_, f"{data_name}_{model_name}_{label}_noisy.joblib")
    else:
        dump(grid.best_estimator_, f"{data_name}_{model_name}_{label}.joblib")
    with open(f"best_params_{model_name}.txt", "+a") as f:
        f.write(str(grid.best_params_) + "\n")

