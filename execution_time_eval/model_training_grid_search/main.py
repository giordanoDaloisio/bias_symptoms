import time
import pyRAPL
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold

if __name__ == "__main__":
    pyRAPL.setup()
    measure = pyRAPL.Measurement("bar")
    times = []
    csv_output = pyRAPL.outputs.CSVOutput("measures.csv")
    data = pd.read_csv("bias_symptoms.csv", index_col=[0, 1])
    model = XGBClassifier()
    params = {
        "learning_rate": [0.05, 0.01, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    for i in range(20):
        measure.begin()
        start_time = time.time()
        # kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        # for itrain, itest in kfold.split(data.index.unique().values):
        #     train_index = data.index.unique()[itrain]
        #     test_index = data.index.unique()[itest]
        #     train = data.loc[train_index]
        #     test = data.loc[test_index]
        # Grid Search hyperparam selection
        grid = GridSearchCV(model, params, cv=5)
        grid.fit(
            data.drop(
                columns=["statistical_parity", "equal_opportunity", "average_odds"]
            ).values,
            data[["statistical_parity", "equal_opportunity", "average_odds"]].values,
        )
        end_time = time.time()
        measure.end()
        measure.export(csv_output)
        times.append(end_time - start_time)
        print(f"Round: {i} completed")
    with open("times.txt", "w") as f:
        for time in times:
            f.write(str(time) + "\n")
    csv_output.save()
