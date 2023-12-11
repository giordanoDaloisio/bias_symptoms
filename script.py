import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from metrics import Metrics
from argparse import ArgumentParser
from icecream import ic
import os


def analysis(model, test: pd.DataFrame, predicted_label, true_label, positive_value):
    importance = permutation_importance(
        model,
        test.drop(columns=[predicted_label, true_label], axis=1),
        test[predicted_label],
        scoring="accuracy",
    )
    symptoms = pd.DataFrame(
        importance.importances_mean,
        index=test.drop(columns=[true_label, predicted_label]).columns,
        columns=["importance"],
    )
    correlation = test.corr()["prediction"].drop(columns=[true_label, predicted_label])
    symptoms["correlation"] = correlation
    # csymptoms = symptoms[symptoms["importance"] > 0.01]
    binary_variables = [c for c in test.columns if test[c].nunique() == 2]
    symptoms = symptoms[symptoms.index.isin(binary_variables)]
    symptoms.reset_index(inplace=True)
    symptoms.rename(columns={"index": "variable"}, inplace=True)
    metrics = Metrics(test, predicted_label, true_label, positive_value)
    sp = []
    eo = []
    unbalance = []
    for i in symptoms["variable"]:
        unbalance.append(metrics.group_ratio({i: 0}))
        sp.append(metrics.statistical_parity({i: 0}))
        eo.append(metrics.equalized_odds({i: 0}))
    symptoms["unbalance"] = unbalance
    symptoms["statistical_parity"] = sp
    symptoms["equalized_odds"] = eo
    return symptoms


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-l", "--label", type=str)
    parser.add_argument("--has_index", type=bool, default=False)

    args = parser.parse_args()
    label = args.label
    if args.has_index:
        data = pd.read_csv(os.path.join("data", args.data), index_col=0)
    else:
        data = pd.read_csv(os.path.join("data", args.data))
    train, test = train_test_split(data, test_size=0.2, random_state=0)

    model = LogisticRegression()
    model = model.fit(train.drop(columns=label, axis=1), train[label])

    test["prediction"] = model.predict(test.drop(columns=label))

    symptoms_nobias = analysis(model, test, "prediction", label, 1)

    symptoms_nobias.sort_values(by="statistical_parity", ascending=False)

    name = args.data.split(".")[0]
    symptoms_nobias.to_csv(f"symptoms_{name}.csv")
