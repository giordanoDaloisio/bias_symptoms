import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mutual_info_score
from sklearn.inspection import permutation_importance
from metrics import Metrics
from utils import get_label
from argparse import ArgumentParser


def get_probs(data, label):
    probs = []
    for v in data[label].unique():
        val = len(data[data[label] == v]) / data.shape[0]
        probs.append(val)
    return probs


def get_symbols(data):
    symb = []
    for c in data.select_dtypes(exclude=["number"]).columns:
        symb.append(len(data[c].unique()))
    return symb


def gini_fun(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (
            cumxw[-1] * cumw[-1]
        )
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def simposon_index(x):
    counts = x.value_counts()
    f = np.sum(counts**2)
    return (1 / f) - 1


def shannon_index(x):
    counts = x.value_counts()
    logs = np.log(counts)
    return -(1 / np.log(len(counts))) * np.sum(counts * logs)


def analysis(
    model,
    test: pd.DataFrame,
    predicted_label,
    true_label,
    positive_value,
    correlation,
):
    symptoms = pd.DataFrame()
    binary_variables = [
        c
        for c in test.drop(columns=[true_label, predicted_label]).columns
        if test[c].nunique() == 2
    ]
    metrics = Metrics(test, predicted_label, true_label, positive_value)

    coeff = []
    sp = []
    eo = []
    aod = []
    unpriv_unbalance = []
    priv_unbalance = []
    corr_true = []
    variable = []
    unpriv_prob = []
    priv_prob = []
    mutual_info = []
    kurtosis_var = []
    skew_var = []
    gini = []
    simpson = []
    shannon = []
    ir = []
    for i in binary_variables:
        corr_true.append(test[[true_label, i]].corr(correlation)[true_label][i])
        unpriv_unbalance.append(metrics.group_ratio({i: 0})[0])
        priv_unbalance.append(metrics.group_ratio({i: 0})[1])
        unpriv_prob.append(metrics.compute_probs({i: 0}, False)[0])
        priv_prob.append(metrics.compute_probs({i: 0}, False)[1])
        sp.append(metrics.statistical_parity({i: 0}))
        eo.append(metrics.equal_accuracy({i: 0}))
        aod.append(metrics.average_odds({i: 0}))
        variable.append(i)
        mutual_info.append(mutual_info_score(test[i], test[true_label]))

        kurt = test.kurt()

        kurtosis_var.append(kurt[i])

        skew = test.skew()

        skew_var.append(skew[i])
        gini.append(gini_fun(test[i]))
        simpson.append(simposon_index(test[i]))
        shannon.append(shannon_index(test[i]))
        ir.append(test[i].value_counts().min() / test[i].value_counts().max())
    symptoms["variable"] = variable
    symptoms["correlation_true"] = corr_true
    symptoms["mutual_info"] = mutual_info
    symptoms["unpriv_prob_pos"] = unpriv_prob
    symptoms["priv_prob_pos"] = priv_prob
    symptoms["unpriv_unbalance"] = unpriv_unbalance
    symptoms["priv_unbalance"] = priv_unbalance
    symptoms["statistical_parity"] = sp
    symptoms["equal_opportunity"] = eo
    symptoms["average_odds"] = aod
    symptoms["variable"] = variable
    symptoms["kurtosis_var"] = kurtosis_var
    symptoms["skew_var"] = skew_var
    symptoms["gini"] = gini
    symptoms["simpson"] = simpson
    symptoms["shannon"] = shannon
    symptoms["ir"] = ir
    symptoms["pos_prob"] = symptoms["unpriv_prob_pos"] - symptoms["priv_prob_pos"]

    return symptoms


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument(
        "-c",
        "--correlation",
        type=str,
        choices=["pearson", "spearman", "kendall"],
        default="kendall",
    )
    parser.add_argument("-s", "--splits", default=10, type=int)
    parser.add_argument("-m", "--model", choices=["logreg", "rf", "xgb", "mlp"])

    args = parser.parse_args()
    label, pos_val = get_label(args.data)
    data = pd.read_csv(args.data, index_col=0)
    n_folds = args.splits

    kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = pd.DataFrame()

    for train_index, test_index in kfolds.split(data):

        train = data.iloc[train_index]
        test = data.iloc[test_index]
        if args.model == "logreg":
            model = LogisticRegression()
        if args.model == "rf":
            model = RandomForestClassifier()
        if args.model == "mlp":
            model = MLPClassifier()
        if args.model == "xgb":
            model = XGBClassifier()
        model = model.fit(
            train.drop(columns=label, axis=1).values, train[label].values.ravel()
        )

        test["prediction"] = model.predict(test.drop(columns=label))

        symptoms = analysis(
            model, test, "prediction", label, pos_val, correlation=args.correlation
        )
        results = pd.concat([results, symptoms])
    name = os.path.basename(args.data)
    name = name.split(".")[0]
    os.makedirs(f"symptoms_{args.correlation}_{args.model}", exist_ok=True)
    results.to_csv(
        os.path.join(
            f"symptoms_{args.correlation}_{args.model}", f"{name}_symptoms.csv"
        )
    )
