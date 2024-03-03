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
from scipy.stats import entropy
import os


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

    sp = []
    eo = []
    aod = []
    unbalance = []
    corr_true = []
    variable = []
    unpriv_prob = []
    priv_prob = []
    mutual_info = []
    kurtosis_var = []
    skew_var = []
    for i in binary_variables:
        corr_true.append(test[[true_label, i]].corr(correlation)[true_label][i])
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

        instance_num.append(test.shape[0])
        log_inst_num.append(np.log(test.shape[0]))
        class_num.append(len(test[label].unique()))
        feat_num.append(test.shape[1])
        log_feat_num.append(np.log(test.shape[1]))
        inst_missing_vals.append(0)
        perc_inst_missing_val.append(0)
        feat_missing_val.append(0)
        perc_feat_missing_val.append(0)
        missing_vals.append(0)
        perc_miss_vals.append(0)
        numeric_features.append(len(test.select_dtypes(include=["number"]).columns))
        cat_features.append(len(test.select_dtypes(exclude=["number"]).columns))
        ratio_num_cat.append(
            len(test.select_dtypes(include=["number"]).columns)
            / len(test.select_dtypes(exclude=["number"]).columns)
            if len(test.select_dtypes(exclude=["number"]).columns) > 0
            else 1
        )
        ratio_cat_num.append(
            len(test.select_dtypes(exclude=["number"]).columns)
            / len(test.select_dtypes(include=["number"]).columns)
            if len(test.select_dtypes(include=["number"]).columns) > 0
            else 1
        )
        dataset_ratio.append(test.shape[1] / test.shape[0])
        log_dataset_ratio.append(np.log(test.shape[1] / test.shape[0]))
        inverse_ratio.append(test.shape[0] / test.shape[1])
        log_inverse_ratio.append(np.log(test.shape[0] / test.shape[1]))
        probs = get_probs(test, label)
        # ic(probs)
        class_prob_min.append(min(probs))
        class_prob_max.append(max(probs))
        class_prob_mean.append(np.mean(probs))
        class_prob_std.append(np.std(probs))
        symbols.append(2)
        symbs = get_symbols(test)
        if len(symbs) != 0:
            symbols_min.append(min(symbs))
            symbols_max.append(max(symbs))
            symbols_mean.append(np.mean(symbs))
            symbols_std.append(np.std(symbs))
            symbols_sum.append(np.sum(symbs))
        else:
            symbols_min.append(0)
            symbols_max.append(0)
            symbols_mean.append(0)
            symbols_std.append(0)
            symbols_sum.append(0)

        kurt = test.kurt()
        kurtosis_min.append(min(kurt))
        kurtosis_max.append(max(kurt))
        kurtosis_mean.append(np.mean(kurt))
        kurtosis_std.append(np.std(kurt))
        kurtosis_var.append(kurt[i])

        skew = test.skew()
        skew_min.append(min(skew))
        skew_max.append(max(skew))
        skew_std.append(np.std(skew))
        skew_mean.append(np.mean(skew))
        skew_var.append(skew[i])
        class_entropy.append(entropy(test[label].values))

    symptoms["variable"] = variable
    # symptoms["perm_importance"] = perm_importance
    symptoms["correlation_true"] = corr_true
    # symptoms["spearman_correlation"] = spearman_true
    # symptoms["correlation_pred"] = correlation_pred
    # symptoms["importance_coeff"] = coeff
    symptoms["mutual_info"] = mutual_info
    # symptoms["variable_value"] = sensitive_value
    symptoms["unpriv_prob_pos"] = unpriv_prob
    symptoms["priv_prob_pos"] = priv_prob
    symptoms["unbalance"] = unbalance
    symptoms["statistical_parity"] = sp
    symptoms["equal_opportunity"] = eo
    symptoms["average_odds"] = aod
    symptoms["variable"] = variable
    symptoms["kurtosis_var"] = kurtosis_var
    symptoms["skew_var"] = skew_var
    symptoms["unpriv_prob_neg"] = 1 - symptoms["unpriv_prob_pos"]
    symptoms["priv_prob_neg"] = 1 - symptoms["priv_prob_pos"]
    symptoms["pos_prob"] = symptoms["unpriv_prob_pos"] - symptoms["priv_prob_pos"]
    symptoms["neg_prob"] = symptoms["unpriv_prob_neg"] - symptoms["priv_prob_neg"]

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
