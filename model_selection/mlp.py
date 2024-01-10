import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from trainer import train_model
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data")
    args = parser.parse_args()

    data = pd.read_csv(args.data, index_col=[0, 1])
    data_name = args.data.split("/")[1].split(".")[0]

    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50, 25)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter": [200, 300, 500],
    }

    train, val = train_test_split(data, test_size=0.2)
    r2_scores, mape_scores = train_model(
        MLPRegressor(activation="relu", solver="adam"),
        train,
        val,
        param_grid,
        data_name,
        "mlp",
    )
    r2_scores.to_csv(f"scores/mlp_r2_scores_{data_name}.csv")
    mape_scores.to_csv(f"scores/mlp_mape_scores_{data_name}.csv")
