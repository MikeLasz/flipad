import argparse
import json
import logging
import pickle
import random
from functools import partial
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from flipad.data_utils import DeepSADDataset
from flipad.optimization import (
    baseline_features_tabular,
    compute_avg_activation,
    reconstruct_activations_tabular,
    select_dimensions,
)

from sad.DeepSAD import DeepSAD
from flipad.utils import remove_paths, sanitize
from flipad.visualization import plot_losses, plot_tensors, find_factors_closest_to_sqrt
from flipad.wrappers.tabular_models import KLWGANWrapper

WRAPPERS = {wrapper.name: wrapper for wrapper in [KLWGANWrapper]}

DATASETS = [data + "_" + str(k) for data in ["whitewine", "redwine"] for k in range(5)]


torch.multiprocessing.set_sharing_strategy("file_system")


def main(args):
    # SETUP ============================================================================
    if args.seed == -1:
        args.seed = random.randint(1, 999)
        print(f"No seed selected. Randomly sampled seed={args.seed}")
    torch.manual_seed(args.seed)

    # create output paths
    output_path = (
        args.output_path
        / f"model={args.model}-checkpoint={sanitize(args.checkpoint_path)}-feat={args.feat}-n_epochs={args.n_epochs}-lr={args.lr}-lr_milestones={args.lr_milestones}-rec_alpha={args.rec_alpha}-rec_lr={args.rec_lr}-rec_momentum={args.rec_momentum}-rec_max_iter={args.rec_max_iter}"
    )
    plot_path = output_path / "plots"
    model_path = output_path / "models"
    result_path = output_path / "results"
    cache_path = args.output_path / "cache"
    for path in [plot_path, model_path, result_path, cache_path]:
        path.mkdir(parents=True, exist_ok=True)

    # save configuration
    with open(output_path / "args.txt", "w") as f:
        f.write(pformat(vars(args)))

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(output_path / f"log.log"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    logger = logging.getLogger(__name__)

    # by default use all models in dataset folder except train model
    if args.test_dirs is None:
        args.test_dirs = [
            path.name
            for path in (args.data_path / args.dataset).iterdir()
            if path.is_dir()
        ]
        args.test_dirs.remove(args.train_dir)
        args.test_dirs.remove(args.real_dir)
        args.test_dirs.insert(0, args.real_dir)  # make sure real comes first
        # include only directories that fit to args.modelnr
        args.test_dirs = remove_paths(paths=args.test_dirs, modelnr=args.modelnr)

    print(f"Testing dirs: {args.test_dirs}")

    # initialize wrapper
    if args.dataset.startswith("redwine"):
        output_dim = 11
    elif args.dataset.startswith("whitewine"):
        output_dim = 11
    elif args.dataset.startswith("parkinsons"):
        output_dim = 15

    wrapper = WRAPPERS[args.model](
        output_dim=output_dim, checkpoint_path=args.checkpoint_path
    )

    path_fake_train = args.data_path / args.dataset / args.train_dir / "train"

    # construct feature func
    if args.feat == "act":
        # get average activation
        avg_act_path = (
            cache_path
            / "avg_act"
            / f"model={args.model}-checkpoint={sanitize(args.checkpoint_path)}-num_avg_act={args.num_avg_act}.pkl"
        )
        if avg_act_path.exists():
            logger.info(f"Loading cached average activations from {avg_act_path}.")
            with open(avg_act_path, "rb") as f:
                reg_anchor = pickle.load(f)
        else:
            reg_anchor = compute_avg_activation(
                wrapper=wrapper,
                num_samples=args.num_avg_act,
                batch_size=args.batch_size,
            )

            avg_act_path.parent.mkdir(parents=True, exist_ok=True)
            with open(avg_act_path, "wb") as f:
                pickle.dump(reg_anchor, f)

        feature_func = partial(
            reconstruct_activations_tabular,
            wrapper=wrapper,
            reg_anchor=reg_anchor,
            alpha=args.rec_alpha,
            momentum=args.rec_momentum,
            lr=args.rec_lr,
            max_iter=args.rec_max_iter,
            num_workers=args.num_workers,
            cache_path=cache_path,
            downsampling=args.downsampling,
            avoid_caching=args.avoid_caching,
        )

    elif args.feat in ["raw", "dct"]:
        feature_func = partial(
            baseline_features_tabular,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # TRAINING =========================================================================
    logger.info("TRAINING")

    # get train data
    feat_fake_train, _, _ = feature_func(
        path=path_fake_train, num_samples=args.num_train, perturbation=None
    )

    path_real_train = args.data_path / args.dataset / args.real_dir / "train"
    feat_real_train, _, _ = feature_func(
        path=path_real_train, num_samples=args.num_train, perturbation=None
    )

    X_train = torch.cat([feat_fake_train, feat_real_train])
    y_train = torch.cat(
        [torch.ones(len(feat_fake_train)), -torch.ones(len(feat_real_train))]
    )

    # scaling/normalization
    EPS_std = 0.05
    USE_NORMALIZATION = args.use_normalization

    if args.feat == "act":
        std_train = X_train[y_train == 1].std(dim=0)
        mean_train = X_train[y_train == 1].mean(dim=0)
        if USE_NORMALIZATION:
            X_train = (X_train - mean_train) / (std_train + EPS_std)

        # Plotting
        visualization_shape = (20, 15)

        plot_tensors(
            [
                value.reshape(visualization_shape)
                for value in [
                    reg_anchor.cpu(),
                    feat_fake_train.mean(dim=0).cpu(),
                    feat_real_train.mean(dim=0).cpu(),
                ]
            ],
            save_path=plot_path / f"avg_train_features.pdf",
            col_labels=["avg", args.train_dir, args.real_dir],
        )

    else:
        mean_train, std_train = X_train.mean(dim=0), X_train.std(dim=0)
        if USE_NORMALIZATION:
            X_train = (X_train - mean_train) / (std_train + EPS_std)
    if args.feat in ["raw", "dct"]:
        # Plotting
        print("No plotting implemented so far!")

    # get validation data
    path_fake_val = args.data_path / args.dataset / args.train_dir / "val"
    X_val, _, _ = feature_func(
        path=path_fake_val, num_samples=args.num_val, perturbation=None
    )

    # scaling/normalization
    if USE_NORMALIZATION:
        X_val = (X_val - mean_train) / (std_train + EPS_std)

    # train DeepSAD
    deep_SAD = DeepSAD()
    input_dim = 300 if args.feat == "act" else output_dim

    if args.feat == "act":
        selected_dimensions = select_dimensions(
            data=X_train,
            num_samples=args.num_train,
            select_dimensions=args.num_select_dimensions,
        )
        input_dim = len(selected_dimensions)
        # plot selected dimensions:
        visualization_shape_selected = find_factors_closest_to_sqrt(
            args.num_select_dimensions
        )
        if visualization_shape_selected[0] == None:
            visualization_shape_selected = (visualization_shape_selected[1], 1)
        plot_tensors(
            [
                value[selected_dimensions].reshape(visualization_shape_selected)
                for value in [
                    reg_anchor.cpu(),
                    feat_fake_train.mean(dim=0).cpu(),
                    feat_real_train.mean(dim=0).cpu(),
                ]
            ],
            save_path=plot_path
            / f"avg_train_features_selected={args.num_select_dimensions}.pdf",
            col_labels=["avg", args.train_dir, args.real_dir],
        )
    else:
        selected_dimensions = range(input_dim)
        visualization_shape_selected = find_factors_closest_to_sqrt(input_dim)
        if visualization_shape_selected[0] == None:
            visualization_shape_selected = (visualization_shape_selected[1], 1)

    kwargs_train = {
        "use_bias": args.mlp_bias,
        "rep_dim": args.rep_dim,
        "h_dims": args.mlp_hdims,
        "x_dim": input_dim,
        "dropout": args.dropout,
    }

    deep_SAD.set_network(
        args.net_name, input_dim, rep_dim=kwargs_train["rep_dim"], kwargs=kwargs_train
    )

    deepsad_path = (
        model_path
        / f"deepsad_{args.net_name}_num_channels={args.num_select_dimensions}.tar"
    )
    if deepsad_path.exists() and args.load_pretrained_ad:
        deep_SAD.load_model(model_path=deepsad_path)
        logger.info(f"Loading cached DeepSAD checkpoint from {deepsad_path}.")
    else:
        dataset = DeepSADDataset(
            X_train=X_train[:, selected_dimensions],
            X_test=X_train[:, selected_dimensions],
            X_val=X_val[:, selected_dimensions],
            y_train=y_train,
            y_test=y_train,
        )

        if args.use_ae:
            deep_SAD.pretrain(
                dataset,
                optimizer_name=args.optimizer_name,
                lr=args.lr,
                n_epochs=args.n_epochs,
                lr_milestones=tuple(args.lr_milestones),
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                n_jobs_dataloader=args.num_workers,
                kwargs=kwargs_train,
            )
        deep_SAD.train(
            dataset,
            optimizer_name=args.optimizer_name,
            lr=args.lr,
            n_epochs=args.n_epochs,
            lr_milestones=tuple(args.lr_milestones),
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            n_jobs_dataloader=args.num_workers,
            tolerable_fnr=args.tolerable_fnr,
        )
        model_path.parent.mkdir(exist_ok=True, parents=True)
        deep_SAD.save_model(export_model=deepsad_path, save_ae=False)

    deep_SAD.net.eval()
    deep_SAD.net.cuda()

    # COMPUTING THRESHOLDS =============================================================
    logger.info("COMPUTING THRESHOLDS")

    val_distances = []
    for batch in DataLoader(
        TensorDataset(X_val[:, selected_dimensions]), batch_size=args.batch_size
    ):
        output = deep_SAD.net(batch[0].cuda()).detach().cpu()
        distance = torch.sum((output - torch.Tensor(deep_SAD.c)) ** 2, dim=1)
        val_distances.append(distance)
    val_distances = torch.cat(val_distances)

    thresholds = {}
    for fnr in args.tolerable_fnr:
        thresholds[f"fnr={fnr}"] = torch.quantile(val_distances, q=1 - fnr).item()
    logger.info(f"Thresholds: {thresholds}.")

    # TESTING ==========================================================================
    logger.info("TESTING")

    # load test data
    path_fake_test = args.data_path / args.dataset / args.train_dir / "test"
    if args.feat == "act":
        feat_fake_test, rec_loss, l1_loss = feature_func(
            path=path_fake_test,
            num_samples=args.num_test,
            perturbation=None,
            compute_loss=False,
        )
        # measure reconstruction loss and l1-loss
        recon_losses = {test_dir: None for test_dir in args.test_dirs}
        l1_losses = {test_dir: None for test_dir in args.test_dirs}

        recon_losses[args.model] = rec_loss
        l1_losses[args.model] = l1_loss
    else:
        feat_fake_test, _, _ = feature_func(
            path=path_fake_test,
            num_samples=args.num_test,
            perturbation=None,
        )

    avg_features = {}
    avg_features[args.model] = feat_fake_test.mean(dim=0)

    results = []
    df = None
    for test_dir in args.test_dirs:
        logger.info(f"Testing {test_dir}.")
        path_other_test = args.data_path / args.dataset / test_dir / "test"

        if args.feat == "act":
            feat_other_test, recon_loss, l1_loss = feature_func(
                path=path_other_test, num_samples=args.num_test, perturbation=None
            )
            recon_losses[test_dir] = recon_loss
            l1_losses[test_dir] = l1_loss
        else:
            feat_other_test, _, _ = feature_func(
                path=path_other_test, num_samples=args.num_test, perturbation=None
            )

        X_test = torch.cat([feat_fake_test, feat_other_test])
        y_test = torch.cat(
            [torch.ones(len(feat_fake_test)), -torch.ones(len(feat_other_test))]
        )

        # scaling/normalization
        if USE_NORMALIZATION:
            X_test = (X_test - mean_train) / (std_train + EPS_std)

        test_distances = []
        for batch in DataLoader(
            TensorDataset(X_test[:, selected_dimensions]), batch_size=args.batch_size
        ):
            output = deep_SAD.net(batch[0].cuda()).detach().cpu()
            distance = torch.sum((output - torch.Tensor(deep_SAD.c)) ** 2, dim=1)
            test_distances.append(distance)
        test_distances = torch.cat(test_distances)

        accuracies = {}
        for threshold_name, threshold_value in thresholds.items():
            y_test_pred = torch.where(
                test_distances > threshold_value, -1, 1
            )  # greater -> -1 = outlier
            accuracies[threshold_name] = accuracy_score(
                y_true=y_test, y_pred=y_test_pred
            )
        # log df of accuracies:
        if df is None:
            df = pd.DataFrame(columns=["fnr", test_dir])
            df["fnr"] = accuracies.keys()
            df[test_dir] = accuracies.values()
        else:
            df_to_append = pd.DataFrame(columns=[test_dir])
            df_to_append[test_dir] = accuracies.values()
            df = df.join(df_to_append)

        result = {
            "model": args.model,
            "checkpoint": str(args.checkpoint_path),
            "seed": args.seed,
            "my": args.train_dir,
            "other": test_dir,
            "deepsadnet": args.net_name,
            "weight_decay": args.weight_decay,
            "channels": args.num_select_dimensions,
        }
        result.update(accuracies)
        results.append(result)
        logger.info(result)

        avg_features[test_dir] = feat_other_test.mean(dim=0)

    # Compute AUC
    X_test = feat_fake_test
    y_test = torch.ones(len(feat_fake_test))

    for test_dir in args.test_dirs:
        path_other_test = args.data_path / args.dataset / test_dir / "test"
        if args.feat == "act":
            feat_other_test, recon_loss, l1_loss = feature_func(
                path=path_other_test,
                num_samples=args.num_test,
                perturbation=None,
            )
            recon_losses[test_dir] = recon_loss
            l1_losses[test_dir] = l1_loss
        else:
            feat_other_test, _, _ = feature_func(
                path=path_other_test,
                num_samples=args.num_test,
                perturbation=None,
            )

        X_test = torch.cat(
            [X_test, feat_other_test[: int(args.num_test / len(args.test_dirs))]]
        )
        y_test = torch.cat(
            [y_test, -torch.ones(int(args.num_test / len(args.test_dirs)))]
        )

    # scaling/normalization
    if USE_NORMALIZATION:
        X_test = (X_test - mean_train) / (std_train + EPS_std)

    test_distances = []
    for batch in DataLoader(
        TensorDataset(X_test[:, selected_dimensions]), batch_size=args.batch_size
    ):
        output = deep_SAD.net(batch[0].cuda()).detach().cpu()
        distance = torch.sum((output - torch.Tensor(deep_SAD.c)) ** 2, dim=1)
        test_distances.append(distance)
    test_distances = np.concatenate(test_distances)

    auc = roc_auc_score(y_test, -test_distances)
    logger.info(f"AUC: {auc}")
    np.save(result_path / f"auc_different.npy", auc)

    for result in results:
        with open(result_path / f"results.json", "a") as f:
            json.dump(result, f)
            f.write(",\n")

    # Plot averages
    print("test")
    if args.feat == "act":
        visualization_shape = (20, 15)

        plot_tensors(
            [value.reshape(visualization_shape) for value in avg_features.values()],
            save_path=plot_path / f"avg_features.pdf",
            col_labels=list(avg_features.keys()),
        )
    else:
        # flat tensors
        avg_features_flat = {
            key: value.flatten() for key, value in avg_features.items()
        }

    # plots of averages
    if args.feat == "act":
        visualization_shape = (20, 15)
        test_avgs = {"avg": reg_anchor.cpu()}
        test_avgs.update(avg_features)
        plot_tensors(
            [value.reshape(visualization_shape) for value in test_avgs.values()],
            save_path=plot_path / f"avg_test_features.pdf",
            col_labels=list(test_avgs.keys()),
        )

        # plot selected dimensions
        plot_tensors(
            [
                value[selected_dimensions].reshape(visualization_shape_selected)
                for value in test_avgs.values()
            ],
            save_path=plot_path
            / f"avg_test_features_selected={args.num_select_dimensions}.pdf",
            col_labels=list(test_avgs.keys()),
        )

    if USE_NORMALIZATION:
        features = (feat_fake_test - mean_train) / (std_train + EPS_std)
    else:
        features = feat_fake_test
    features = features[:, selected_dimensions]
    labels = [args.model]
    num_samples_to_plot = 10
    features_plotting = [features[:num_samples_to_plot]]
    for test_dir in args.test_dirs:
        path_other_test = args.data_path / args.dataset / test_dir / "test"
        if args.feat == "act":
            feat_other_test, recon_loss, l1_loss = feature_func(
                path=path_other_test,
                num_samples=args.num_test,
                perturbation=None,
            )
            recon_losses[test_dir] = recon_loss
            l1_losses[test_dir] = l1_loss
        else:
            feat_other_test, _, _ = feature_func(
                path=path_other_test,
                num_samples=args.num_test,
                perturbation=None,
            )

        if USE_NORMALIZATION:
            feat_other_test = (feat_other_test - mean_train) / (std_train + EPS_std)
        feat_other_test = feat_other_test[:, selected_dimensions]
        features = torch.concatenate([features, feat_other_test])
        labels.append(test_dir)

        features_plotting.append(feat_other_test[:num_samples_to_plot])

    # plot test samples
    sample_visualization_shape = (
        num_samples_to_plot,
        visualization_shape_selected[0],
        visualization_shape_selected[1],
    )

    if args.feat == "act":
        col_labels = list(test_avgs.keys())[1:]
    else:
        col_labels = [args.train_dir]
        col_labels.extend(args.test_dirs)
    plot_tensors(
        [
            sample_plot.reshape(sample_visualization_shape)
            for sample_plot in features_plotting
        ],
        save_path=plot_path
        / f"sample_test_features_selected={args.num_select_dimensions}.pdf",
        col_labels=col_labels,
        row_labels=range(num_samples_to_plot),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--data-path", type=Path, default="data")
    parser.add_argument("--output-path", type=Path, default="output")

    # data
    parser.add_argument("--train-dir", help="Directory of the training data.")
    parser.add_argument("--real-dir", help="Directory of the real data.")
    parser.add_argument("--test-dirs", nargs="*", help="Directories of the test data.")

    parser.add_argument("--dataset", choices=DATASETS, default=DATASETS[0])
    parser.add_argument(
        "--feat", choices=["act", "raw", "dct"], default="act", help="Feature to use."
    )
    parser.add_argument(
        "--num-train", type=int, default=10000, help="Number of training samples."
    )
    parser.add_argument(
        "--num-test", type=int, default=1000, help="Number of test samples."
    )
    parser.add_argument(
        "--num-val", type=int, default=1000, help="Number of validation samples."
    )

    # thresholds
    parser.add_argument(
        "--tolerable-fnr",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.05, 0.1, 0.0, 0.001],
        help="False negative rates for which to compute thresholds.",
    )

    # activations
    parser.add_argument(
        "--model",
        choices=WRAPPERS,
        default=list(WRAPPERS.keys())[0],
        help="Type of generative model.",
    )
    parser.add_argument("--modelnr", type=int, default=1, help="Model number.")
    parser.add_argument(
        "--checkpoint-path", type=Path, help="Checkpoint path of the model"
    )
    parser.add_argument(
        "--load-pretrained-ad",
        action="store_true",
        help="If true, then load pretrained DeepSAD model.",
    )
    parser.add_argument(
        "--rec-alpha",
        type=float,
        default=0.0005,
        help="Regularization Parameter that penalizes the l1-diff.",
    )
    parser.add_argument("--num-avg-act", type=int, default=5000)
    parser.add_argument("--rec-momentum", type=float, default=0.0)
    parser.add_argument("--rec-lr", type=float, default=0.025)
    parser.add_argument("--rec-max-iter", type=int, default=10000)

    parser.add_argument(
        "--avoid-caching",
        action="store_true",
        help="Avoid caching recovered activations.",
    )

    # technical
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=-1)

    # DeepSAD
    parser.add_argument(
        "--net-name",
        choices=[
            "mlp",
        ],
        default="mlp",
    )
    parser.add_argument(
        "--use-normalization",
        action="store_true",
        help="Normalize the data before training",
    )
    parser.add_argument(
        "--use-ae",
        action="store_true",
        help="Train autoencoder to initialize a favorable c",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=32,
        help="Dimension of the representation space of DeepSAD.",
    )
    parser.add_argument(
        "--mlp-bias",
        action="store_true",
        help="If net-name=mlp, this flag decides if the mlp has bias or not.",
    )
    parser.add_argument(
        "--mlp-hdims",
        nargs="+",
        type=int,
        default=[128, 64],
        help="hidden dimensions of mlp",
    )

    parser.add_argument(
        "--optimizer-name",
        choices=["adam", "amsgrad"],
        default="adam",
        help="Name of the optimizer to use for Deep SAD network training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for Deep SAD network training. Default=0.001",
    )
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        type=int,
        default=[25, 50, 100],
        help="Epochs at which learning rate is decreased by one order of magnitude.",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=50, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for mini-batch training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.5e-6,
        help="Weight decay (L2 penalty) hyperparameter for Deep SAD objective.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate of DeepSAD MLP."
    )

    parser.add_argument(
        "--downsampling",
        type=str,
        default="avg",
        help="Spatial downsampling of the Stable Diffusion Activations. avg, max, center_crop, or random_crop",
    )

    parser.add_argument(
        "--num-select-dimensions",
        type=int,
        default=300,
        help="Determines how many channels (among 128 channels in total in SD) to pick. We pick those based on the avg difference in training data.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
