import pandas as pd
from sklearn.metrics import mutual_info_score

from metrics import Metrics


def symp_extractor(test: pd.DataFrame, true_label, positive_value, sensitive_var):
    symptoms = pd.DataFrame()
    metrics = Metrics(test, None, true_label, positive_value)

    # for v in test[i].unique():
    symptoms["correlation_true"] = test[[true_label, sensitive_var]].corr("kendall")[
        true_label
    ][sensitive_var]
    symptoms["unbalance"] = metrics.group_ratio({sensitive_var: 0})
    symptoms["unpriv_prob_pos"] = metrics.compute_probs({sensitive_var: 0}, False)[0]
    symptoms["priv_prob_pos"] = metrics.compute_probs({sensitive_var: 0}, False)[1]
    symptoms["mutual_info"] = mutual_info_score(test[sensitive_var], test[true_label])

    kurt = test.kurt()
    symptoms["kurtosis_var"] = kurt[sensitive_var]

    skew = test.skew()
    symptoms["skew_var"] = skew[sensitive_var]
    symptoms["unpriv_prob_neg"] = 1 - symptoms["unpriv_prob_pos"]
    symptoms["priv_prob_neg"] = 1 - symptoms["priv_prob_pos"]
    symptoms["pos_prob"] = symptoms["unpriv_prob_pos"] - symptoms["priv_prob_pos"]
    symptoms["neg_prob"] = symptoms["unpriv_prob_neg"] - symptoms["priv_prob_neg"]

    return symptoms
