import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mutual_info_score
from sklearn.inspection import permutation_importance
from metrics import Metrics
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
    test: pd.DataFrame,
    true_label,
    positive_value,
    correlation,
):
    symptoms = pd.DataFrame()

    binary_variables = [
        c for c in test.drop(columns=[true_label]).columns if test[c].nunique() == 2
    ]
    # symptoms = symptoms[symptoms.index.isin(binary_variables)]
    # symptoms.reset_index(inplace=True)
    # symptoms.rename(columns={"index": "variable"}, inplace=True)
    metrics = Metrics(test, None, true_label, positive_value)
    # importance_coeff = model.coef_[0]
    # importance = permutation_importance(
    #     model,
    #     test.drop(columns=[predicted_label, true_label], axis=1),
    #     test[predicted_label],
    #     scoring="accuracy",
    # ).importances_mean

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
    instance_num = []
    log_inst_num = []
    class_num = []
    feat_num = []
    log_feat_num = []
    inst_missing_vals = []
    perc_inst_missing_val = []
    feat_missing_val = []
    perc_feat_missing_val = []
    missing_vals = []
    perc_miss_vals = []
    numeric_features = []
    cat_features = []
    ratio_num_cat = []
    ratio_cat_num = []
    dataset_ratio = []
    log_dataset_ratio = []
    inverse_ratio = []
    log_inverse_ratio = []
    class_prob_min = []
    class_prob_max = []
    class_prob_mean = []
    class_prob_std = []
    symbols = []
    symbols_min = []
    symbols_max = []
    symbols_mean = []
    symbols_std = []
    symbols_sum = []
    kurtosis_min = []
    kurtosis_max = []
    kurtosis_mean = []
    kurtosis_std = []
    kurtosis_var = []
    skew_min = []
    skew_max = []
    skew_mean = []
    skew_std = []
    skew_var = []
    class_entropy = []
    for i in binary_variables:
        # for v in test[i].unique():
        corr_true.append(test.corr(correlation)[true_label][i])
        # coeff.append(
        #     importance_coeff[
        #         test.drop(columns=[true_label, predicted_label]).columns.get_loc(i)
        #     ]
        # )
        unbalance.append(metrics.group_ratio({i: 0}))

        unpriv_prob.append(metrics.compute_probs({i: 0}, False)[0])
        priv_prob.append(metrics.compute_probs({i: 0}, False)[1])
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
    symptoms["instance_num"] = instance_num
    symptoms["log_inst_num"] = log_inst_num
    symptoms["class_num"] = class_num
    symptoms["feat_num"] = feat_num
    symptoms["log_feat_num"] = log_feat_num
    symptoms["inst_missing_vals"] = inst_missing_vals
    symptoms["perc_inst_missing_val"] = perc_inst_missing_val
    symptoms["feat_missing_val"] = feat_missing_val
    symptoms["perc_feat_missing_val"] = perc_feat_missing_val
    symptoms["missing_vals"] = missing_vals
    symptoms["perc_miss_vals"] = perc_miss_vals
    symptoms["numeric_features"] = numeric_features
    symptoms["cat_features"] = cat_features
    symptoms["ratio_num_cat"] = ratio_num_cat
    symptoms["ratio_cat_num"] = ratio_cat_num
    symptoms["dataset_ratio"] = dataset_ratio
    symptoms["log_dataset_ratio"] = log_dataset_ratio
    symptoms["inverse_ratio"] = inverse_ratio
    symptoms["log_inverse_ratio"] = log_inverse_ratio
    symptoms["class_prob_min"] = class_prob_min
    symptoms["class_prob_max"] = class_prob_max
    symptoms["class_prob_mean"] = class_prob_mean
    symptoms["class_prob_std"] = class_prob_std
    symptoms["symbols"] = symbols
    symptoms["symbols_min"] = symbols_min
    symptoms["symbols_max"] = symbols_max
    symptoms["symbols_mean"] = symbols_mean
    symptoms["symbols_std"] = symbols_std
    symptoms["symbols_sum"] = symbols_sum
    symptoms["kurtosis_min"] = kurtosis_min
    symptoms["kurtosis_max"] = kurtosis_max
    symptoms["kurtosis_mean"] = kurtosis_mean
    symptoms["kurtosis_std"] = kurtosis_std
    symptoms["kurtosis_var"] = kurtosis_var
    symptoms["skew_min"] = skew_min
    symptoms["skew_max"] = skew_max
    symptoms["skew_mean"] = skew_mean
    symptoms["skew_std"] = skew_std
    symptoms["skew_var"] = skew_var
    symptoms["class_entropy"] = class_entropy
    symptoms["unpriv_prob_neg"] = 1 - symptoms["unpriv_prob_pos"]
    symptoms["priv_prob_neg"] = 1 - symptoms["priv_prob_pos"]
    symptoms["pos_prob"] = symptoms["unpriv_prob_pos"] - symptoms["priv_prob_pos"]
    symptoms["neg_prob"] = symptoms["unpriv_prob_neg"] - symptoms["priv_prob_neg"]

    return symptoms
