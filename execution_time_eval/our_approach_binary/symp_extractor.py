import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

from metrics import Metrics


def symp_extractor(test: pd.DataFrame, true_label, positive_value, sensitive_var):
    symptoms = []
    metrics = Metrics(test, None, true_label, positive_value)
    unpriv_prob_pos = metrics.compute_probs({sensitive_var: 0}, False)[0]
    priv_prob_pos = metrics.compute_probs({sensitive_var: 0}, False)[1]
    symptoms.append(
        test[[true_label, sensitive_var]].corr("kendall")[true_label][sensitive_var]
    )
    symptoms.append(
        test[[true_label, sensitive_var]].corr("spearman")[true_label][sensitive_var]
    )
    symptoms.append(metrics.group_ratio({sensitive_var: 0}))
    symptoms.append(metrics.compute_probs({sensitive_var: 0}, False)[0])
    symptoms.append(metrics.compute_probs({sensitive_var: 0}, False)[1])
    symptoms.append(mutual_info_score(test[sensitive_var], test[true_label]))
    symptoms.append(test[sensitive_var].kurt())
    symptoms.append(test[sensitive_var].skew())
    symptoms.append(1 - unpriv_prob_pos)
    symptoms.append(1 - priv_prob_pos)
    symptoms.append(unpriv_prob_pos - priv_prob_pos)
    symptoms.append((1 - unpriv_prob_pos) - (1 - priv_prob_pos))
    return np.array(symptoms).reshape(1, -1)
