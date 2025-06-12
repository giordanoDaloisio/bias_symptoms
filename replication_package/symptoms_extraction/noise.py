import pandas as pd
import numpy as np
from csv import writer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from cleanlab.classification import CleanLearning
from argparse import ArgumentParser
from utils import get_label

def get_noise(df, label):
  cls = LogisticRegression(n_jobs=-1)
  cleaner = CleanLearning(cls)
  encoder = LabelEncoder()
  labels = encoder.fit_transform(df[label])
  cleaner.fit(df.drop(columns=[label]), labels)
  noise_labels = cleaner.noise_matrix
  noisy_labels_estimate = noise_labels.sum() - np.trace(noise_labels)
  total_samples = noise_labels.sum()
  overall_noise_rate = noisy_labels_estimate / total_samples
  return overall_noise_rate

if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument("--data", type=str, required=True, help="Path to the data directory")
  args = parser.parse_args()

  df = pd.read_csv(args.data)
  label, pos_class = get_label(args.data)
  noise = get_noise(df, label)
  data_name = args.data.split("/")[-1]
  stats = pd.read_csv("data_stats.csv")
  stats.loc[stats["File"] == data_name, 'Noise'] = noise
  stats.to_csv("data_stats.csv", index=False)