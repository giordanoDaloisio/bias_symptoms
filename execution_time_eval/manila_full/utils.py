import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from aif360.datasets import BinaryLabelDataset
from copy import deepcopy
from scipy import stats
import metrics


from metrics import *
from methods import FairnessMethods

np.random.seed(2)

# TRAINING FUNCTIONS

def cross_val(classifier, data, label, unpriv_group, priv_group, sensitive_features, positive_label, metrics, n_splits=2, preprocessor=None, inprocessor=None, postprocessor=None):
    n_splits = 2
    data_start = data.copy()
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    for train, test in fold.split(data_start):
        weights = None
        data = data_start.copy()
        df_train = data.iloc[train]
        df_test = data.iloc[test]
        model = deepcopy(classifier)
        exp = bool(inprocessor == FairnessMethods.EG or inprocessor == FairnessMethods.GRID)
        #adv = bool(inprocessor == FairnessMethods.AD)
        pred, model = _model_train(df_train, df_test, label, model, sensitive_features, exp=exp, weights=weights)
        if postprocessor:
            df_train = df_train.set_index(sensitive_features[0])
            df_test = df_test.set_index(sensitive_features[0])
        compute_metrics(pred, unpriv_group, label, positive_label, metrics, sensitive_features)
    return model, metrics


def _train_test_split(df_train, df_test, label):
    x_train = df_train.drop(label, axis=1).values
    y_train = df_train[label].values.ravel()
    x_test = df_test.drop(label, axis=1).values
    y_test = df_test[label].values.ravel()
    return x_train, x_test, y_train, y_test


def _model_train(df_train, df_test, label, classifier, sensitive_features, exp=False, weights=None, adv=False):
    x_train, x_test, y_train, y_test = _train_test_split(
        df_train, df_test, label)
    model = deepcopy(classifier)
    if adv:
        model.fit(x_train, y_train)
    else:
        if exp:
            model.fit(x_train, y_train,
                    sensitive_features=df_train[sensitive_features]) 
        else:
            model.fit(x_train, y_train, classifier__sample_weight=weights)
  
    df_pred = _predict_data(model, df_test, label, x_test)
    if adv:
        model.sess_.close()
    return df_pred, model



def _predict_data(model, df_test, label, x_test, aif_data=False):
    pred = model.predict(x_test)
    df_pred = df_test.copy()
    df_pred['y_true'] = df_pred[label]
    if aif_data:
        df_pred[label] = pred.labels
    else:
        df_pred[label] = pred
    return df_pred


##### METRICS FUNCTIONS #####

def compute_metrics(df_pred, unpriv_group, label, positive_label, metrics, sensitive_features):
    df_pred = df_pred.reset_index()
    stat_par = statistical_parity(
        df_pred, unpriv_group, label, positive_label)
    metrics['stat_par'].append(stat_par)
    eo = equalized_odds(
        df_pred, unpriv_group, label, positive_label)
    metrics['eq_odds'].append(eo)
    ao = average_odds_difference(df_pred, unpriv_group, label, positive_label)
    metrics['ao'] = ao
    accuracy_score = accuracy(df_pred, label)
    metrics['acc'].append(accuracy_score)
    metrics['hmean'].append(
        stats.hmean([
            accuracy_score,
 
 
            norm_data(eo), 
            norm_data(stat_par), 
            norm_data(ao),
        ])
    )
    return metrics
