import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference as average_odds

from sklearn.metrics import accuracy_score


def _get_groups(data, label_name, positive_label, group_condition):
    query = "&".join([str(k) + "==" + str(v) for k, v in group_condition.items()])
    label_query = label_name + "==" + str(positive_label)
    unpriv_group = data.query(query)
    unpriv_group_pos = data.query(query + "&" + label_query)
    priv_group = data.query("~(" + query + ")")
    priv_group_pos = data.query("~(" + query + ")&" + label_query)
    return unpriv_group, unpriv_group_pos, priv_group, priv_group_pos


def _compute_probs(data_pred, label_name, positive_label, group_condition):
    unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = _get_groups(
        data_pred, label_name, positive_label, group_condition
    )
    unpriv_group_prob = len(unpriv_group_pos) / len(unpriv_group)
    priv_group_prob = len(priv_group_pos) / len(priv_group)
    return unpriv_group_prob, priv_group_prob


def _compute_tpr_fpr(y_true, y_pred, positive_label):
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == positive_label:
            if y_true[i] == y_pred[i]:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred[i] == positive_label:
                FP += 1
            else:
                TN += 1
    if TP + FN == 0:
        TPR = 0
    else:
        TPR = TP / (TP + FN)
    if FP + TN == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)
    return FPR, TPR


def _compute_tpr_fpr_groups(data_pred, label, group_condition, positive_label):
    query = "&".join([f"{k}=={v}" for k, v in group_condition.items()])
    unpriv_group = data_pred.query(query)
    priv_group = data_pred.drop(unpriv_group.index)

    y_true_unpriv = unpriv_group["y_true"].values.ravel()
    y_pred_unpric = unpriv_group[label].values.ravel()
    y_true_priv = priv_group["y_true"].values.ravel()
    y_pred_priv = priv_group[label].values.ravel()

    fpr_unpriv, tpr_unpriv = _compute_tpr_fpr(
        y_true_unpriv, y_pred_unpric, positive_label
    )
    fpr_priv, tpr_priv = _compute_tpr_fpr(y_true_priv, y_pred_priv, positive_label)
    return fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv


def statistical_parity(
    data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str
):
    # unpriv_group_prob, priv_group_prob = _compute_probs(
    #     data_pred, label_name, positive_label, group_condition
    # )
    sensitive_attr = list(group_condition.keys())[0]
    data_pred_c = data_pred.copy()
    data_pred_c.set_index(sensitive_attr, inplace=True)
    return statistical_parity_difference(data_pred_c["y_true"], data_pred_c[label_name])
    # return unpriv_group_prob - priv_group_prob


def average_odds_difference(
    data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str
):
    data_pred_c = data_pred.copy()
    sensitive_attr = list(group_condition.keys())[0]
    data_pred_c.set_index(sensitive_attr, inplace=True)
    return average_odds(data_pred_c["y_true"], data_pred_c[label_name])
    # fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = _compute_tpr_fpr_groups(
    #     data_pred, label_name, group_condition, positive_label
    # )
    # return (tpr_priv - tpr_unpriv) + (fpr_priv - fpr_unpriv)


def equalized_odds(
    data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str
):
    sensitive_attr = list(group_condition.keys())[0]
    data_pred_c = data_pred.copy()
    data_pred_c.set_index(sensitive_attr, inplace=True)
    return equal_opportunity_difference(data_pred_c["y_true"], data_pred_c[label_name])
    # _, tpr_unpriv, _, tpr_priv = _compute_tpr_fpr_groups(
    #     data_pred, label_name, group_condition, positive_label
    # )
    # return tpr_priv - tpr_unpriv


def accuracy(df_pred: pd.DataFrame, label: str):
    return accuracy_score(df_pred["y_true"].values, df_pred[label].values)


def norm_data(data):
    return abs(1 - abs(data))
