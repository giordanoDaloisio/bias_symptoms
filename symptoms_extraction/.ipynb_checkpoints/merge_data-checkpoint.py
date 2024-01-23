import os
import pandas as pd

base_folder = "symptoms_kendall_logreg"

full_data = pd.DataFrame()

for file in os.listdir(base_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(base_folder, file), index_col=0)
        name = file.split("_")[0]
        df["data"] = name

        full_data = pd.concat([full_data, df])
full_data.dropna(inplace=True)
# full_data.rename(
#     columns={"unpriv_prob": "unpriv_prob_pos", "priv_prob": "priv_prob_pos"},
#     inplace=True,
# )
# full_data["unpriv_prob_neg"] = 1 - full_data["unpriv_prob_pos"]
# full_data["priv_prob_neg"] = 1 - full_data["priv_prob_pos"]
# full_data["pos_prob"] = full_data["unpriv_prob_pos"] - full_data["priv_prob_pos"]
# full_data["neg_prob"] = full_data["unpriv_prob_neg"] - full_data["priv_prob_neg"]
full_data.set_index(["variable", "data"], inplace=True)
full_data.to_csv(os.path.join("..", "model_selection", "data", "all_features.csv"))

bias_symp = [
    "correlation_true",
    "correlation_pred",
    "mutual_info",
    "unpriv_prob_pos",
    "priv_prob_pos",
    "unbalance",
    "statistical_parity",
    "equal_opportunity",
    "average_odds",
    "unpriv_prob_neg",
    "priv_prob_neg",
    "pos_prob",
    "neg_prob",
]

symp_data = full_data[bias_symp]
symp_data.to_csv(os.path.join("..", "model_selection", "data", "bias_symptoms.csv"))

metafeatures = [
    "statistical_parity",
    "equal_opportunity",
    "average_odds",
    "instance_num",
    "log_inst_num",
    "class_num",
    "feat_num",
    "log_feat_num",
    "inst_missing_vals",
    "perc_inst_missing_val",
    "feat_missing_val",
    "perc_feat_missing_val",
    "missing_vals",
    "perc_miss_vals",
    "numeric_features",
    "cat_features",
    "ratio_num_cat",
    "ratio_cat_num",
    "dataset_ratio",
    "log_dataset_ratio",
    "inverse_ratio",
    "log_inverse_ratio",
    "class_prob_min",
    "class_prob_max",
    "class_prob_mean",
    "class_prob_std",
    "symbols",
    "symbols_min",
    "symbols_max",
    "symbols_mean",
    "symbols_std",
    "symbols_sum",
    "kurtosis_min",
    "kurtosis_max",
    "kurtosis_mean",
    "kurtosis_std",
    "kurtosis_var",
    "skew_min",
    "skew_max",
    "skew_mean",
    "skew_std",
    "skew_var",
    "class_entropy",
]

meta_data = full_data[metafeatures]
meta_data.to_csv(os.path.join("..", "model_selection", "data", "metafeatures.csv"))
