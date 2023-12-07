import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)


class Metrics:
    def __init__(
        self, data: pd.DataFrame, label_name: str, true_label: str, positive_label: int
    ):
        self.data_pred = data
        self.label_name = label_name
        self.positive_label = positive_label
        self.true_label = true_label

    def __get_groups(self, group_condition):
        query = "&".join(
            [f"`{str(k)}`" + "==" + str(v) for k, v in group_condition.items()]
        )
        label_query = f"`{self.label_name}`" + "==" + str(self.positive_label)
        unpriv_group = self.data_pred.query(query)
        unpriv_group_pos = self.data_pred.query(query + "&" + label_query)
        priv_group = self.data_pred.query("~(" + query + ")")
        priv_group_pos = self.data_pred.query("~(" + query + ")&" + label_query)
        return unpriv_group, unpriv_group_pos, priv_group, priv_group_pos

    def __compute_probs(self, group_condition):
        unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = self.__get_groups(
            group_condition
        )
        unpriv_group_prob = len(unpriv_group_pos) / len(unpriv_group)
        priv_group_prob = len(priv_group_pos) / len(priv_group)
        return unpriv_group_prob, priv_group_prob

    def __compute_tpr_fpr(self, y_true, y_pred):
        TN = 0
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(y_true)):
            if y_true[i] == self.positive_label:
                if y_true[i] == y_pred[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if y_pred[i] == self.positive_label:
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

    def __compute_tpr_fpr_groups(self, group_condition):
        query = "&".join([f"`{k}`=={v}" for k, v in group_condition.items()])
        unpriv_group = self.data_pred.query(query)
        priv_group = self.data_pred.drop(unpriv_group.index)

        y_true_unpriv = unpriv_group[self.true_label].values.ravel()
        y_pred_unpric = unpriv_group[self.label_name].values.ravel()
        y_true_priv = priv_group[self.true_label].values.ravel()
        y_pred_priv = priv_group[self.label_name].values.ravel()

        fpr_unpriv, tpr_unpriv = self.__compute_tpr_fpr(y_true_unpriv, y_pred_unpric)
        fpr_priv, tpr_priv = self.__compute_tpr_fpr(y_true_priv, y_pred_priv)
        return fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv

    def group_ratio(self, group_condition):
        unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = self.__get_groups(
            group_condition
        )
        if len(unpriv_group_pos) > 0:
            w_exp = (len(unpriv_group) / len(self.data_pred)) * (
                len(
                    self.data_pred[
                        self.data_pred[self.label_name] == self.positive_label
                    ]
                )
                / len(self.data_pred)
            )
            w_obs = len(unpriv_group_pos) / len(self.data_pred)
            return w_obs / w_exp
        # else:
        #     w_exp = (len(priv_group) / len(self.data_pred)) * (
        #         len(
        #             self.data_pred[
        #                 self.data_pred[self.label_name] == self.positive_label
        #             ]
        #         )
        #         / len(self.data_pred)
        #     )
        #     w_obs = len(priv_group_pos) / len(self.data_pred)
        #     return w_obs / w_exp

    def disparate_impact(self, group_condition):
        unpriv_group_prob, priv_group_prob = self.__compute_probs(group_condition)
        return (
            min(
                unpriv_group_prob / priv_group_prob, priv_group_prob / unpriv_group_prob
            )
            if unpriv_group_prob != 0 and priv_group_prob != 0
            else 0
        )

    def statistical_parity(self, group_condition: dict):
        unpriv_group_prob, priv_group_prob = self.__compute_probs(group_condition)
        return unpriv_group_prob - priv_group_prob

    def equalized_odds(self, group_condition: dict):
        fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = self.__compute_tpr_fpr_groups(
            group_condition
        )
        return (tpr_priv - tpr_unpriv) + (fpr_priv - fpr_unpriv)

    def average_odds_difference(self, group_condition: str):
        unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = self.__get_groups(
            group_condition
        )
        accuracy_priv = accuracy(priv_group, self.label_name)
        accuracy_unpriv = accuracy(unpriv_group, self.label_name)
        return accuracy_priv - accuracy_unpriv

    def true_pos_diff(self, group_condition: str):
        fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = self.__compute_tpr_fpr_groups(
            group_condition
        )
        return tpr_unpriv - tpr_priv

    def false_pos_diff(self, group_condition: str):
        fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = self.__compute_tpr_fpr_groups(
            group_condition
        )
        return fpr_unpriv - fpr_priv


# def zero_one_loss_diff(
#     y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: list
# ):
#     mf = MetricFrame(
#         metrics=zero_one_loss,
#         y_true=y_true,
#         y_pred=y_pred,
#         sensitive_features=sensitive_features,
#     )
#     return mf.difference()


def accuracy(df_pred: pd.DataFrame, label: str):
    return accuracy_score(df_pred["y_true"].values, df_pred[label].values)


# def precision(df_pred: pd.DataFrame, label: str):
#     return precision_score(
#         df_pred["y_true"].values, df_pred[label].values, average="weighted"
#     )


# def recall(df_pred: pd.DataFrame, label: str):
#     return recall_score(
#         df_pred["y_true"].values, df_pred[label].values, average="weighted"
#     )


# def f1(df_pred: pd.DataFrame, label: str):
#     return f1_score(df_pred["y_true"].values, df_pred[label].values, average="weighted")


# def auc(df_pred: pd.DataFrame, label: str):
#     return roc_auc_score(df_pred["y_true"].values, df_pred[label].values)


# def euclidean_distance(df_pred, label):
#     return np.linalg.norm(df_pred["y_true"].values - df_pred[label].values) / len(
#         df_pred
#     )


# def manhattan_distance(df_pred, label):
#     return np.sum(np.abs(df_pred["y_true"].values - df_pred[label].values)) / len(
#         df_pred
#     )


# def mahalanobis_distance(df_pred, label):
#     return np.sqrt(
#         np.sum(np.square(df_pred["y_true"].values - df_pred[label].values))
#     ) / len(df_pred)


def norm_data(data):
    return abs(1 - abs(data))
