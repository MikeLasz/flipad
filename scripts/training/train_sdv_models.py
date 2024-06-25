from sdv.tabular import CTGAN, TVAE, CopulaGAN
from argparse import ArgumentParser
from collections import OrderedDict
import os
import pickle


import pandas as pd
import numpy as np

import flipad.klwgan_data_utils as D
from train_klwgan import kde, mmd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str, choices=["ctgan", "tvae", "copulagan"])
    parser.add_argument("--load_model", action="store_true")
    args = parser.parse_args()
    return args


n_epochs = 500
MODELS = {
    "ctgan": CTGAN(verbose=True, epochs=n_epochs),
    "tvae": TVAE(epochs=n_epochs),
    "copulagan": CopulaGAN(verbose=True, epochs=n_epochs),
}

name_to_dataset = OrderedDict(
    [
        ("redwine", D.RedWine),
        ("whitewine", D.WhiteWine),
        ("parkinsons", D.Parkinsons),
        ("hepmass", D.HepMass),
        ("gas", D.Gas),
        ("power", D.Power),
    ]
)

paths = {
    "redwine": "data/wine+quality/wine-red_",
    "whitewine": "data/wine+quality/wine-white_",
}

COLUMNS = {
    "redwine": [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ],
    "whitewine": [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ],
}


if __name__ == "__main__":
    args = parse_args()

    dataset = name_to_dataset[args.data]()
    train_data = dataset.data
    test_data = dataset.test_data

    for j in range(5):
        model = MODELS[args.model]
        model_save_path = f"trained_models/{args.model}/"
        os.makedirs(model_save_path, exist_ok=True)
        if args.load_model and os.path.exists(model_save_path + f"{args.data}_{j}.pkl"):
            with open(model_save_path + f"{args.data}_{j}.pkl", "rb") as f:
                model = pickle.load(f)
        else:
            model.fit(
                pd.DataFrame(
                    train_data, columns=COLUMNS[args.data][: len(train_data[0])]
                )
            )
            model.save(model_save_path + f"{args.data}_{j}.pkl")

        # evaluation:
        data = model.sample(5000).to_numpy()
        kde_score = kde(data, test_data)
        mmd_score = mmd(data, test_data)
        print(f"KDE={kde_score}; MMD={mmd_score}")
        with open(f"{model_save_path}{args.data}_{j}_scores.txt", "w") as f:
            f.write(f"KDE={kde_score}; MMD={mmd_score}")

        # save samples
        data_train = model.sample(100000).to_numpy()
        data_test = model.sample(10000).to_numpy()
        data_val = model.sample(10000).to_numpy()
        synthetic_data = {"test": data_test, "train": data_train, "val": data_val}
        for subdir in ["test", "train", "val"]:
            samples_save_path = f"data/{args.data}_{j}/{args.model}/{subdir}"
            os.makedirs(samples_save_path, exist_ok=True)
            np.savetxt(
                samples_save_path + "/data.csv", synthetic_data[subdir], delimiter=","
            )
