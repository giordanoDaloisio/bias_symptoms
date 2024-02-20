import time
import pyRAPL
import pandas as pd
from xgboost import XGBClassifier

if __name__ == '__main__':
  pyRAPL.setup()
  measure = pyRAPL.Measurement('bar')
  times = []
  csv_output = pyRAPL.outputs.CSVOutput('measures.csv')
  data = pd.read_csv('bias_symptoms.csv', index_col=[0,1])
  x = data.drop(columns=['statistical_parity', 'equal_opportunity', 'average_odds'])
  y = data[['statistical_parity', 'equal_opportunity', 'average_odds']]
  model = XGBClassifier(colsample_bytree=0.8, gamma=0, learning_rate=0.05, max_depth=6, subsample=0.6)
  for i in range(20):
    measure.begin()
    start_time = time.time()
    model.fit(x, y)
    end_time = time.time()
    measure.end()
    measure.export(csv_output)
    times.append(end_time - start_time)
    print(f'Round: {i} completed')
  with open("times.txt", "w") as f:
      for time in times:
          f.write(str(time) + "\n")
  csv_output.save()