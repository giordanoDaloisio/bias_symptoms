# Replication package

This repository contains the replication package of the experimental evaluation reported in the paper.

## Requirements

### Miniconda

The suggested way to run the experiments is by creating a virtual environment using [Miniconda](https://docs.anaconda.com/free/miniconda/index.html). Next you can install the requirements using the following command from the `replication_package` directory:

```bash
conda env create -f environment.yml
```

### Manual installation

The experiments have been executed using `Python 3.10.13`.

Next install the following packages using `pip`:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `jupyterlab`
- `pyRAPL`

## Structure

The replication package is structured as follows:

- `data/`: contains the bias symptoms datasets used in the experiments. In particular, `bias_symptoms.csv` is the dataset used in RQ1, RQ2 and RQ4. `bias_symptoms_mlp.csv` and `bias_symptoms_rf.csv` are the datasets used in RQ3. `bias_symptoms_raw.csv` is the dataset with raw bias metrics, before the post-processing phase.

- `data_analysis/`: contains the Jupyter notebooks used to perform the statistics reported in Section 3 of the paper.

- `rq1/`: contains the Jupyter notebook used to answer RQ1, as well as the raw results and figures.

- `rq2/`: contains the Jupyter notebook used to answer RQ2, and the figures.

- `rq3/`: contains the Jupyter notebook used to answer RQ3, and the figures.

- `rq4/`: contains the Jupyter notebook used to answer RQ4, and the source code of the two baselines employed (i.e., `standard` and `aequitas`).

- `small_data_analysis`: contains the Jupyter notebook used to replicate RQ2 and the statistics reported in Section 3 of the paper using only datasets having a number of binary variables within the 75% of the total employed datasets.

- `symptoms_extraction`: contains the code used to generate the bias symptoms dataset. In particular, `main.py` is the script used to extract the symptoms and the raw bias metrics from all the 24 datasets, `merge_data.py` is the script used to preprocess and merge the different symptoms into a single dataset.

- `tables_generation`: contains the Jupyter notebook used to generate the tables reported in the paper.

- `training/`: contains the code to tune the hyperparameters of the models used in the experimental evaluation.

## Running the experiments

### Research questions

To replicate the research questions you can run the Jupyter notebooks in their respective directories. The notebooks are self-contained and contain all the necessary code to reproduce the results reported in the paper.

### Bias symptoms dataset creation

To extract the symptoms, first obtain and preprocess the datasets reported in the [DATASETS.md](../DATASETS.md) file. Next, from the `symptoms_extraction` directory , first run `main.py` script to extract the symptoms and the raw bias metrics with the following command:

```bash
python main.py -d <folder_containing_datasets> -m <either 'logreg', 'mlp', 'rf'>
```

Finally, run the `merge_data.py` script to preprocess and merge the different symptoms into a single dataset:

```bash
python merge_data.py -f <either 'logreg', 'mlp', 'rf'>
```

The files are saved in the `data/` directory.

### Training

To tune the hyperparameters of the models used in the experimental evaluation, run the following command from the `training` directory:

```bash
python <model_name>.py -d <path_to_bias_symptoms_dataset>
```

where `<model_name>` can be either `xgb`, `mlp` or `rf`.
