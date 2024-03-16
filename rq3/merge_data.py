import os
import pandas as pd
from argparse import ArgumentParser


def data_proc(all):
    metrics = ["statistical_parity", "equal_opportunity", "average_odds"]
    all.loc[:, metrics] = all[metrics].abs()
    all.loc[all["statistical_parity"] > 0.2, "statistical_parity"] = 1
    all.loc[all["statistical_parity"] != 1, "statistical_parity"] = 0
    all.loc[all["equal_opportunity"] > 0.15, "equal_opportunity"] = 1
    all.loc[all["equal_opportunity"] != 1, "equal_opportunity"] = 0
    all.loc[all["average_odds"] > 0.15, "average_odds"] = 1
    all.loc[all["average_odds"] != 1, "average_odds"] = 0
    return all


parser = ArgumentParser()
parser.add_argument("-f", "--folder", help="Folder with the results")
args = parser.parse_args()

base_folder = f"symptoms_kendall_{args.folder}"

full_data = pd.DataFrame()

for file in os.listdir(base_folder):
    df = pd.read_csv(os.path.join(base_folder, file), index_col=0)
    name = file.split("_")[0]
    df["data"] = name
    full_data = pd.concat([full_data, df])

full_data.set_index(["variable", "data"], inplace=True)
full_data["pos_prob"] = full_data["pos_prob"].abs()
full_data["gini"] = (full_data["gini"] - full_data["gini"].min()) / (
    full_data["gini"].max() - full_data["gini"].min()
)
full_data["simpson"] = (full_data["simpson"] - full_data["simpson"].min()) / (
    full_data["simpson"].max() - full_data["simpson"].min()
)
full_data["shannon"] = (full_data["shannon"] - full_data["shannon"].min()) / (
    full_data["shannon"].max() - full_data["shannon"].min()
)
full_data["ir"] = (full_data["ir"] - full_data["ir"].min()) / (
    full_data["ir"].max() - full_data["ir"].min()
)

# bias_symp = [
#     "correlation_true",
#     "mutual_info",
#     "unpriv_prob_pos",
#     "priv_prob_pos",
#     "unpriv_unbalance",
#     "priv_unbalance",
#     "statistical_parity",
#     "equal_opportunity",
#     "average_odds",
#     "pos_prob",
#     "kurtosis_var",
#     "skew_var",
# ]

# symp_data = full_data[bias_symp]

# metafeatures = [
#     "statistical_parity",
#     "equal_opportunity",
#     "average_odds",
#     "instance_num",
#     "log_inst_num",
#     "class_num",
#     "feat_num",
#     "log_feat_num",
#     "inst_missing_vals",
#     "perc_inst_missing_val",
#     "feat_missing_val",
#     "perc_feat_missing_val",
#     "missing_vals",
#     "perc_miss_vals",
#     "numeric_features",
#     "cat_features",
#     "ratio_num_cat",
#     "ratio_cat_num",
#     "dataset_ratio",
#     "log_dataset_ratio",
#     "inverse_ratio",
#     "log_inverse_ratio",
#     "class_prob_min",
#     "class_prob_max",
#     "class_prob_mean",
#     "class_prob_std",
#     "symbols",
#     "symbols_min",
#     "symbols_max",
#     "symbols_mean",
#     "symbols_std",
#     "symbols_sum",
#     "kurtosis_min",
#     "kurtosis_max",
#     "kurtosis_mean",
#     "kurtosis_std",
#     "skew_min",
#     "skew_max",
#     "skew_mean",
#     "skew_std",
#     "class_entropy",
# ]

# meta_data = full_data[metafeatures]

# data_proc(full_data).to_csv(
#     os.path.join(f"result_class_{args.folder}", "all_features.csv")
# )
data_proc(full_data).to_csv(
    os.path.join(f"result_class_{args.folder}", f"bias_symptoms_{args.folder}.csv")
)
# data_proc(meta_data).to_csv(
#     os.path.join(f"result_class_{args.folder}", "metafeatures.csv")
# )
