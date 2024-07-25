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

base_folder = f"symptoms_{args.folder}"

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

full_data.to_csv(os.path.join("..", "data", f"bias_symptoms_raw_{args.folder}.csv"))

data_proc(full_data).to_csv(
    os.path.join("..", "data", f"bias_symptoms_{args.folder}.csv")
)
