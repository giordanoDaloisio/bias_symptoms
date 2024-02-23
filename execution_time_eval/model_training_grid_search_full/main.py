import os
import time

# import pyRAPL
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold

if __name__ == "__main__":
    # pyRAPL.setup()
    # measure = pyRAPL.Measurement("bar")
    data = pd.read_csv("bias_symptoms.csv", index_col=[0, 1])
    model = XGBClassifier()
    params = {
        "learning_rate": [0.05, 0.01, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    os.makedirs("measures", exist_ok=True)
    os.makedirs("times", exist_ok=True)
    for i in range(20):
        # csv_output = pyRAPL.outputs.CSVOutput(
        #     os.path.join("measures", f"measure_{i}.csv")
        # )
        # measure.begin()
        start_time = time.time()
        grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
        grid.fit(
            data.drop(
                columns=["statistical_parity", "equal_opportunity", "average_odds"]
            ).values,
            data[["statistical_parity", "equal_opportunity", "average_odds"]].values,
        )
        end_time = time.time()
        # measure.end()
        # measure.export(csv_output)
        with open(os.path.join("times", f"time_{i}.txt"), "w") as f:
            f.write(str(end_time - start_time) + "\n")
        # csv_output.save()
        print(f"Round: {i} completed")
