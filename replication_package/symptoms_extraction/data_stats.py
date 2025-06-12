import pandas as pd
import numpy as np
import os
from utils import get_label
from csv import writer
from sklearn.linear_model import LogisticRegression
from cleanlab.classification import CleanLearning

def get_unbalance(df, label, pos_class):
    label_nums = df[label].value_counts()
    unbalance = label_nums[pos_class] / len(df[label])

    unbalance = (unbalance - (1/len(df[label].unique()))) / (1 - (1/len(df[label].unique())))
    return abs(unbalance)

def get_noise(df, label):
  cls = LogisticRegression()
  cleaner = CleanLearning(cls)
  cleaner.fit(df.drop(columns=[label]), df[label])
  noise_labels = cleaner.noise_matrix
  noisy_labels_estimate = noise_labels.sum() - np.trace(noise_labels)
  total_samples = noise_labels.sum()
  overall_noise_rate = noisy_labels_estimate / total_samples
  return overall_noise_rate

if __name__ == "__main__":

  with open("data_stats.csv", "w", newline='') as csvfile:
      csv_writer = writer(csvfile)
      csv_writer.writerow(["File", "Unbalance", "Noise"])

  for data in os.listdir("data"):
    df = pd.read_csv(f"data/{data}")
    label, pos_class = get_label(data)
    unbalance = get_unbalance(df, label, pos_class)
    # cleaner.fit(df.drop(columns=[label]), df[label])
    # noise_labels = cleaner.noise_matrix
    # noisy_labels_estimate = noise_labels.sum() - np.trace(noise_labels)
    # total_samples = noise_labels.sum()
    # noise = noisy_labels_estimate / total_samples
    
    with open("data_stats.csv", "a") as f:
      csv_writer = writer(f)
      csv_writer.writerow([data, unbalance,])