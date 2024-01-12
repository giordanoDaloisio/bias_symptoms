import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mutual_info_score
from sklearn.inspection import permutation_importance
from metrics import Metrics
from utils import get_label
from argparse import ArgumentParser
import os


def analysis(
    model,
    test: pd.DataFrame,
    predicted_label,
    true_label,
    positive_value,
    correlation,
):
    symptoms = pd.DataFrame()
    # symptoms = pd.DataFrame(
    #     importance.importances_mean,
    #     index=test.drop(columns=[true_label, predicted_label]).columns,
    #     columns=["perm_importance"],
    # )

    binary_variables = [
        c
        for c in test.drop(columns=[true_label, predicted_label]).columns
        if test[c].nunique() == 2
    ]
    # symptoms = symptoms[symptoms.index.isin(binary_variables)]
    # symptoms.reset_index(inplace=True)
    # symptoms.rename(columns={"index": "variable"}, inplace=True)
    metrics = Metrics(test, predicted_label, true_label, positive_value)
    # importance_coeff = model.coef_[0]
    # importance = permutation_importance(
    #     model,
    #     test.drop(columns=[predicted_label, true_label], axis=1),
    #     test[predicted_label],
    #     scoring="accuracy",
    # ).importances_mean

    coeff = []
    sp = []
    eo = []
    aod = []
    unbalance = []
    # sensitive_value = []
    correlation_pred = []
    corr_true = []
    perm_importance = []
    variable = []
    performance_score = []
    unpriv_prob = []
    priv_prob = []
    mutual_info = []
    for i in binary_variables:
        # for v in test[i].unique():
        correlation_pred.append(test.corr(correlation)[predicted_label][i])
        corr_true.append(test.corr(correlation)[true_label][i])
        # coeff.append(
        #     importance_coeff[
        #         test.drop(columns=[true_label, predicted_label]).columns.get_loc(i)
        #     ]
        # )
        unbalance.append(metrics.group_ratio({i: 0}))

        unpriv_prob.append(metrics.compute_probs({i: 0}, False)[0])
        priv_prob.append(metrics.compute_probs({i: 0}, False)[1])
        sp.append(metrics.statistical_parity({i: 0}))
        eo.append(metrics.equal_accuracy({i: 0}))
        aod.append(metrics.average_odds({i: 0}))
        # sensitive_value.append(v)
        variable.append(i)
        # perm_importance.append(
        #     importance[
        #         test.drop(columns=[true_label, predicted_label]).columns.get_loc(i)
        #     ]
        # )
        # performance_score.append(metrics.accuracy(test))
        mutual_info.append(mutual_info_score(test[i], test[true_label]))
    symptoms["variable"] = variable
    # symptoms["perm_importance"] = perm_importance
    symptoms["correlation_true"] = corr_true
    symptoms["correlation_pred"] = correlation_pred
    # symptoms["importance_coeff"] = coeff
    symptoms["mutual_info"] = mutual_info
    # symptoms["variable_value"] = sensitive_value
    symptoms["unpriv_prob"] = unpriv_prob
    symptoms["priv_prob"] = priv_prob
    symptoms["unbalance"] = unbalance
    symptoms["statistical_parity"] = sp
    symptoms["equal_opportunity"] = eo
    symptoms["average_odds"] = aod
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
    parser.add_argument("-m", "--model", choices=["logreg", "svm", "xgb", "mlp"])

    args = parser.parse_args()
    label, pos_val = get_label(args.data)
    data = pd.read_csv(args.data, index_col=0)
    n_folds = args.splits

    kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = pd.DataFrame()

    for train_index, test_index in kfolds.split(data):
        # train, test = train_test_split(data, test_size=0.2, random_state=0)

        train = data.iloc[train_index]
        test = data.iloc[test_index]
        if args.model == "logreg":
            model = LogisticRegression(n_jobs=-1)
        if args.model == "svm":
            model = SVC()
        if args.model == "mlp":
            model = MLPClassifier()
        if args.model == "xgb":
            model = XGBClassifier()
        model = model.fit(
            train.drop(columns=label, axis=1).values, train[label].values.ravel()
        )

        test["prediction"] = model.predict(test.drop(columns=label))

        os.makedirs(f"symptoms_{args.correlation}", exist_ok=True)
        # if train[label].nunique() == 2:
        #     symptoms_nobias = analysis(
        #         model, test, "prediction", label, 1, correlation=args.correlation
        #     )
        #     symptoms_nobias.sort_values(by="statistical_parity", ascending=False)
        #     name = args.data.split(".")[0]
        #     symptoms_nobias.to_csv(f"symptoms_{args.correlation}/symptoms_{name}.csv")
        # else:

        symptoms = analysis(
            model, test, "prediction", label, pos_val, correlation=args.correlation
        )
        results = pd.concat([results, symptoms])
        # symptoms_nobias.sort_values(by="statistical_parity", ascending=False)
    name = os.path.basename(args.data)
    name = name.split(".")[0]
    os.makedirs(f"symptoms_{args.correlation}_{args.model}", exist_ok=True)
    results.to_csv(
        os.path.join(
            f"symptoms_{args.correlation}_{args.model}", f"{name}_symptoms.csv"
        )
    )
