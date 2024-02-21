from aequitas import Audit
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("../data/cmc_proc.csv", index_col=0)
train, test = train_test_split(data, test_size=0.2, random_state=42)
X_train = train.drop("contr_use", axis=1)
y_train = train["contr_use"]
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(test.drop("contr_use", axis=1))
test["score"] = y_pred
test.rename(columns={"contr_use": "label"}, inplace=True)
test["wife_work"] = test["wife_work"].astype(str)
audit = Audit(test[["score", "label", "wife_work"]])
audit.audit()
print(audit.metrics.round(2))
