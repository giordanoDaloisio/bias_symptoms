import os
import pandas as pd

base_folder = "symptoms_with_probs"

full_data = pd.DataFrame()

# for folder in os.listdir(base_folder):
for file in os.listdir(base_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(base_folder, file), index_col=0)
        name = file.split("_")[0]
        df["data"] = name

        full_data = pd.concat([full_data, df])

full_data.to_csv("full_data_probs.csv", index=False)
