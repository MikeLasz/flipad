import argparse
import json
import logging
import pickle
import random
import re
import time
from functools import partial
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch.nn.functional import interpolate, max_pool2d
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import center_crop

from flipad.data_utils import DeepSADDataset
from flipad.optimization import (
    baseline_features,
    compute_avg_activation,
    reconstruct_activations,
    select_channels,
)
from flipad.perturbations import (
    GaussianNoise,
    JPEGCompression,
    MyGaussianBlur,
    ResizedCrop,
)
from sad.DeepSAD import DeepSAD
from flipad.utils import remove_paths, sanitize
from flipad.visualization import plot_losses, plot_tensors
from flipad.wrappers.small_models import (
    DCGANWrapper,
    EBGANWrapper,
    LSGANWrapper,
    WGANGPWrapper,
)
from flipad.wrappers.stablediffusion import StableDiffusionWrapper
from flipad.wrappers.stylegan import StyleGANWrapper
from flipad.wrappers.medigan import MEDIDCGANWrapper


WRAPPERS = {
    wrapper.name: wrapper
    for wrapper in [
        DCGANWrapper,
        LSGANWrapper,
        WGANGPWrapper,
        EBGANWrapper,
        StableDiffusionWrapper,
        StyleGANWrapper,
        MEDIDCGANWrapper,
    ]
}

DATASETS = [
    data + "_" + str(k)
    for data in ["lsun", "celeba", "breastmass", "ffhq"]
    for k in range(5)
]
DATASETS.append("coco2014train")

PERTURBATIONS = {
    "crop": ResizedCrop,
    "blur": MyGaussianBlur,
    "noise": GaussianNoise,
    "jpeg": JPEGCompression,
}


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
        if args.test_models == "different":
            # include only directories that fit to args.modelnr
            args.test_dirs = remove_paths(paths=args.test_dirs, modelnr=args.modelnr)
        elif args.test_models == "same":
            # include only directories that fit to args.model but have different args.modelnr
            args.test_dirs = remove_paths(paths=args.test_dirs, model=args.model)

    print(f"Testing dirs: {args.test_dirs}")

    # set up perturbation
    if args.immunization:
        if args.perturbation_param is None:
            raise ValueError("'--perturbation-param' has to be set.")
        train_perturbation = PERTURBATIONS[args.perturbation](args.perturbation_param)
    else:
        train_perturbation = None

    # initialize wrapper
    wrapper = WRAPPERS[args.model](checkpoint_path=args.checkpoint_path)

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
                avg_act = pickle.load(f)
        else:
            avg_act = compute_avg_activation(
                wrapper=wrapper,
                num_samples=args.num_avg_act,
                batch_size=args.batch_size,
            )

            avg_act_path.parent.mkdir(parents=True, exist_ok=True)
            with open(avg_act_path, "wb") as f:
                pickle.dump(avg_act, f)

        feature_func = partial(
            reconstruct_activations,
            wrapper=wrapper,
            reg_anchor=avg_act,
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
        if args.model in ["stablediffusion", "stylegan2"]:
            resize_to = 128
        elif "medigan" in args.model:
            resize_to = 128  # no resizing
        else:
            resize_to = 32
        print(f"Resize: {resize_to}")
        feature_func = partial(
            baseline_features,
            transform=args.feat,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resize_to=resize_to,
            downsampling=args.downsampling,
        )

    # TRAINING =========================================================================
    logger.info("TRAINING")

    # get train data
    time_compute_features_start = time.time()
    feat_fake_train, _, _ = feature_func(
        img_path=path_fake_train,
        num_samples=args.num_train,
        perturbation=train_perturbation,
    )
    time_compute_features_end = time.time()
    if args.save_time:
        time_compute_features = time_compute_features_end - time_compute_features_start
        logger.info(
            f"Computing the features took {time_compute_features} seconds for {args.num_train} samples."
        )

    path_real_train = args.data_path / args.dataset / args.real_dir / "train"
    feat_real_train, _, _ = feature_func(
        img_path=path_real_train,
        num_samples=args.num_train,
        perturbation=train_perturbation,
    )

    X_train = torch.cat([feat_fake_train, feat_real_train])
    y_train = torch.cat(
        [torch.ones(len(feat_fake_train)), -torch.ones(len(feat_real_train))]
    )

    # scaling/normalization
    EPS_std = 0.05
    if args.feat == "act":
        std_train = X_train[y_train == 1].std(dim=0)
        mean_train = X_train[y_train == 1].mean(dim=0)
        X_train = (X_train - mean_train) / (std_train + EPS_std)

        if wrapper.name in ["stablediffusion", "stylegan2"]:
            downsamp_size = 128
        elif wrapper.name == "medigan_dcgan":
            downsamp_size = 64  # i.e. no downsampling
        else:
            downsamp_size = 32
        if args.downsampling == "max":
            # Apply max-pooling with kernel size of size kernel_size
            if wrapper.name == "stablediffusion":
                kernel_size = 16  # max-downsamping: 128x512x512 -> 128x128x128
            elif wrapper.name == "lsgan":
                kernel_size = 2
            elif wrapper.name in ["dcgan", "wgangp", "ebgan"]:
                kernel_size = 1  # i.e. no downsamlping
            else:  # medigan experiments
                kernel_size = 1  # i.e. no downsampling
            avg_act_downsampled = max_pool2d(
                avg_act.unsqueeze(0), kernel_size=kernel_size, stride=kernel_size
            ).squeeze(0)
        elif args.downsampling == "avg":
            avg_act_downsampled = interpolate(
                avg_act.unsqueeze(0), size=(downsamp_size, downsamp_size)
            ).squeeze(0)
        elif args.downsampling == "center_crop":
            avg_act_downsampled = center_crop(
                avg_act.unsqueeze(0), (downsamp_size, downsamp_size)
            ).squeeze(0)
        else:
            raise NotImplementedError

        # Plotting
        plot_tensors(
            tensors=[
                avg_act_downsampled.cpu(),
                feat_fake_train.mean(dim=0).cpu(),
                feat_real_train.mean(dim=0).cpu(),
            ],
            col_labels=["avg. act.", args.model, "real"],
            save_path=plot_path / "avg_train_features.png",
        )
    elif args.feat == "dct":
        mean_train, std_train = X_train.mean(dim=0), X_train.std(dim=0)
        X_train = (X_train - mean_train) / (std_train + EPS_std)
    if args.feat in ["raw", "dct"]:
        # Plotting
        plot_tensors(
            tensors=[
                feat_fake_train.mean(dim=0).cpu(),
                feat_real_train.mean(dim=0).cpu(),
            ],
            col_labels=[args.model, "real"],
            save_path=plot_path / "avg_train_features.png",
        )

    # get validation data
    path_fake_val = args.data_path / args.dataset / args.train_dir / "val"
    X_val, _, _ = feature_func(
        img_path=path_fake_val,
        num_samples=args.num_val,
        perturbation=train_perturbation,
    )

    # scaling/normalization
    if args.feat in ["act", "dct"]:
        X_val = (X_val - mean_train) / (std_train + EPS_std)

    if args.feat == "act":
        channels = select_channels(
            data=X_train,
            num_samples=args.num_train,
            select_channels=args.num_select_channels,
        )
        top_3_channels = select_channels(
            data=X_train, num_samples=args.num_train, select_channels=3
        )  # for visualization purposes
    elif args.feat in ["dct", "raw"]:
        if wrapper.name == "medigan_dcgan":
            # only one channel in medical images (grayscale)
            channels = range(1)
        else:
            channels = range(3)
            top_3_channels = range(3)

    # train DeepSAD
    deep_SAD = DeepSAD()
    deep_SAD.set_network(args.net_name, len(channels))

    if args.immunization:
        deepsad_path = (
            model_path
            / f"deepsad_{args.net_name}_immun_perturb={args.perturbation}_param={args.perturbation_param}.tar"
        )
    else:
        deepsad_path = (
            model_path
            / f"deepsad_{args.net_name}_num_channels={args.num_select_channels}.tar"
        )
    if deepsad_path.exists() and args.load_pretrained_ad:
        deep_SAD.load_model(model_path=deepsad_path)
        logger.info(f"Loading cached DeepSAD checkpoint from {deepsad_path}.")
    else:
        dataset = DeepSADDataset(
            X_train=X_train[:, channels, :, :],
            X_test=X_train[:, channels, :, :],
            X_val=X_val[:, channels, :, :],
            y_train=y_train,
            y_test=y_train,
        )

        deep_sad_time_start = time.time()
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
        deep_sad_time_end = time.time()
        if args.save_time:
            deep_sad_time = deep_sad_time_end - deep_sad_time_start
            logger.info(f"Training DeepSAD took {deep_sad_time} seconds.")
        model_path.parent.mkdir(exist_ok=True, parents=True)
        deep_SAD.save_model(export_model=deepsad_path, save_ae=False)

    deep_SAD.net.eval()
    deep_SAD.net.cuda()

    # COMPUTING THRESHOLDS =============================================================
    logger.info("COMPUTING THRESHOLDS")

    val_distances = []
    for batch in DataLoader(
        TensorDataset(X_val[:, channels, :, :]), batch_size=args.batch_size
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

    # set up perturbation
    if args.perturbation is None:
        perturbation = None
    else:
        if args.perturbation_param is None:
            raise ValueError("'--perturbation-param' has to be set.")
        perturbation = PERTURBATIONS[args.perturbation](args.perturbation_param)

    # load test data (with perturbation)
    path_fake_test = args.data_path / args.dataset / args.train_dir / "test"
    if args.feat == "act" and wrapper.name in ["dcgan", "wgangp", "lsgan", "ebgan"]:
        feat_fake_test, rec_loss, l1_loss = feature_func(
            img_path=path_fake_test,
            num_samples=args.num_test,
            perturbation=perturbation,
            compute_loss=True,
        )
        # measure reconstruction loss and l1-loss
        recon_losses = {test_dir: None for test_dir in args.test_dirs}
        l1_losses = {test_dir: None for test_dir in args.test_dirs}

        recon_losses[args.model] = rec_loss
        l1_losses[args.model] = l1_loss
    else:
        feat_fake_test, _, _ = feature_func(
            img_path=path_fake_test,
            num_samples=args.num_test,
            perturbation=perturbation,
            compute_loss=False,
        )

    avg_features = {}
    avg_features[args.model] = feat_fake_test.mean(dim=0)[channels, :, :]
    if len(channels) >= 3:
        # for visualization purposes
        avg_features_top3 = {}
        avg_features_top3[args.model] = feat_fake_test.mean(dim=0)[top_3_channels, :, :]

    results = []
    for test_dir in args.test_dirs:
        logger.info(f"Testing {test_dir}.")
        path_other_test = args.data_path / args.dataset / test_dir / "test"

        if args.feat == "act" and wrapper.name in ["dcgan", "wgangp", "lsgan", "ebgan"]:
            feat_other_test, recon_loss, l1_loss = feature_func(
                img_path=path_other_test,
                num_samples=args.num_test,
                perturbation=perturbation,
                compute_loss=True,
            )
            recon_losses[test_dir] = recon_loss
            l1_losses[test_dir] = l1_loss
        else:
            feat_other_test, _, _ = feature_func(
                img_path=path_other_test,
                num_samples=args.num_test,
                perturbation=perturbation,
                compute_loss=False,
            )

        X_test = torch.cat([feat_fake_test, feat_other_test])
        y_test = torch.cat(
            [torch.ones(len(feat_fake_test)), -torch.ones(len(feat_other_test))]
        )

        # scaling/normalization
        if args.feat in ["act", "dct"]:
            X_test = (X_test - mean_train) / (std_train + EPS_std)

        test_distances = []
        for batch in DataLoader(
            TensorDataset(X_test[:, channels, :, :]), batch_size=args.batch_size
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

        modelname = args.model + "_immun" if args.immunization else args.model
        result = {
            "model": modelname,
            "checkpoint": str(args.checkpoint_path),
            "seed": args.seed,
            "my": args.train_dir,
            "other": test_dir,
            "perturbation": args.perturbation,
            "perturbation-param": args.perturbation_param,
            "deepsadnet": args.net_name,
            "weight_decay": args.weight_decay,
            "channels": args.num_select_channels,
        }
        result.update(accuracies)
        results.append(result)
        logger.info(result)

        avg_features[test_dir] = feat_other_test.mean(dim=0)[channels, :, :]
        if len(channels) > 3:
            avg_features_top3[test_dir] = feat_other_test.mean(dim=0)[
                top_3_channels, :, :
            ]

        if args.confusion_matrices:
            tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_pred).ravel()
            # TODO plot
            logger.info(f"{test_dir}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Compute AUC
    X_test = feat_fake_test
    y_test = torch.ones(len(feat_fake_test))

    for test_dir in args.test_dirs:
        logger.info(f"Computing AUC: Loading {test_dir}.")
        path_other_test = args.data_path / args.dataset / test_dir / "test"
        if args.feat == "act" and wrapper.name in ["dcgan", "wgangp", "lsgan", "ebgan"]:
            feat_other_test, recon_loss, l1_loss = feature_func(
                img_path=path_other_test,
                num_samples=args.num_test,
                perturbation=perturbation,
                compute_loss=True,
            )
            recon_losses[test_dir] = recon_loss
            l1_losses[test_dir] = l1_loss
        else:
            feat_other_test, _, _ = feature_func(
                img_path=path_other_test,
                num_samples=args.num_test,
                perturbation=perturbation,
                compute_loss=False,
            )

        X_test = torch.cat(
            [X_test, feat_other_test[: int(args.num_test / len(args.test_dirs))]]
        )
        y_test = torch.cat(
            [y_test, -torch.ones(int(args.num_test / len(args.test_dirs)))]
        )

    # scaling/normalization
    if args.feat in ["act", "dct"]:
        X_test = (X_test - mean_train) / (std_train + EPS_std)

    test_distances = []
    for batch in DataLoader(
        TensorDataset(X_test[:, channels, :, :]), batch_size=args.batch_size
    ):
        output = deep_SAD.net(batch[0].cuda()).detach().cpu()
        distance = torch.sum((output - torch.Tensor(deep_SAD.c)) ** 2, dim=1)
        test_distances.append(distance)
    test_distances = np.concatenate(test_distances)

    auc = roc_auc_score(y_test, -test_distances)
    logger.info(f"AUC: {auc}")
    np.save(result_path / f"auc_{args.test_models}.npy", auc)

    for result in results:
        with open(result_path / f"results.json", "a") as f:
            json.dump(result, f)
            f.write(",\n")

    if args.feat == "act":
        # plot average activations
        # Top 3 channels:
        avg_features_top3_channels = {
            **{"avg. act": avg_act_downsampled[top_3_channels].cpu()},
            **avg_features_top3,
        }
        plot_tensors(
            tensors=list(avg_features_top3_channels.values()),
            save_path=plot_path
            / f"avg_test_features-perturb={args.perturbation}-no_labels-top3_channels.pdf",
        )

        avg_features = {
            **{"avg. act": avg_act_downsampled[channels].cpu()},
            **avg_features,
        }
        plot_tensors(
            tensors=list(avg_features.values()),
            col_labels=[
                re.sub(r"_[0-9]+", "", label) for label in list(avg_features.keys())
            ],
            row_labels=channels,
            save_path=plot_path / f"avg_test_features-perturb={args.perturbation}.png",
        )

        # plot reconststruction losses and l1 loss
        if wrapper.name in ["dcgan", "wgangp", "lsgan", "ebgan"]:
            plot_losses(recon_losses, l1_losses, plot_path)
    else:
        avg_features = {**avg_features}
        plot_tensors(
            tensors=list(avg_features.values()),
            col_labels=[
                re.sub(r"_[0-9]+", "", label) for label in list(avg_features.keys())
            ],
            # row_labels=channels,
            save_path=plot_path / f"avg_test_features-perturb={args.perturbation}.png",
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
    parser.add_argument(
        "--test-models",
        type=str,
        choices=["different", "same"],
        default="different",
        help="Same (but different seeds) or different models to test against?",
    )
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
    parser.add_argument(
        "--perturbation", choices=PERTURBATIONS.keys(), help="Perturbation to apply."
    )
    parser.add_argument(
        "--perturbation-param",
        type=float,
        help="Perturbation parameter (depends on the type of perturbation such as the kernel size in Gaussian Blurring).",
    )
    parser.add_argument(
        "--immunization",
        action="store_true",
        help="enables immunization (adds perturbation to training data)",
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
            "cifar10_LeNet",
            "cifar10_LeNet_ELU",
            "lenet_64channels",
            "lenet_128x128",
            "cifar10_biglenet",
            "lenet_64channels_medigan",
            "lenet_128x128_medigan",
        ],
        default="lenet_64channels",
        help="Name of the Deep SAD network to use.",
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
        "--downsampling",
        type=str,
        default="avg",
        help="Spatial downsampling of the Stable Diffusion Activations. avg, max, center_crop, or random_crop",
    )

    parser.add_argument(
        "--num_select_channels",
        type=int,
        default=128,
        help="Determines how many channels (among 128 channels in total in SD) to pick. We pick those based on the avg difference in training data.",
    )

    parser.add_argument(
        "--save-time",
        action="store_true",
        help="If flag is set, log time for training DeepSAD + for inversion.",
    )
    parser.add_argument(
        "--confusion-matrices",
        action="store_true",
        help="If flag is set, plot confusion matrices.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
