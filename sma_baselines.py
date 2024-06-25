import argparse
import json
import logging
import pickle
import random
from pathlib import Path
from pprint import pformat
from time import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy

from flipad.data_utils import load_dataset
from flipad.perturbations import (
    GaussianNoise,
    JPEGCompression,
    MyGaussianBlur,
    ResizedCrop,
)
from flipad.utils import remove_paths, sanitize
from flipad.wrappers.fingerprinting import FingerprintingWrapper
from flipad.wrappers.inversion import InceptionInversionWrapper, L2InversionWrapper
from flipad.wrappers.tabular_models import KLWGANWrapper
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
        KLWGANWrapper,
    ]
}

ATTRIBUTION_WRAPPERS = {
    wrapper.name: wrapper
    for wrapper in [
        L2InversionWrapper,
        InceptionInversionWrapper,
        FingerprintingWrapper,
    ]
}

DATASETS = [
    data + "_" + str(k)
    for data in ["lsun", "celeba", "breastmass", "whitewine", "redwine", "ffhq"]
    for k in range(5)
]
DATASETS.append("coco2014train")
DATASETS.append("ffhq")

PERTURBATIONS = {
    "crop": ResizedCrop,
    "blur": MyGaussianBlur,
    "noise": GaussianNoise,
    "jpeg": JPEGCompression,
}


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    # SETUP ============================================================================
    if args.seed == -1:
        args.seed = random.randint(1, 999)
        print(f"No seed selected. Randomly sampled seed={args.seed}")
    torch.manual_seed(args.seed)

    if args.perturbation is not None:
        raise NotImplementedError(
            "Perturbations are not supported yet. "
            "Caching paths need to be adjusted to avoid deleting unperturbed results."
        )

    # create output paths
    if args.attr == "fingerprint":
        output_path = (
            args.output_path
            / "baselines"
            / f"model={args.model}-checkpoint={sanitize(args.checkpoint_path)}-attr={args.attr}"
        )
    else:
        output_path = (
            args.output_path
            / "baselines"
            / f"model={args.model}-checkpoint={sanitize(args.checkpoint_path)}-attr={args.attr}-lr_inv={args.lr_inv}-num_steps_inv={args.num_steps_inv}-num_inits_inv={args.num_inits_inv}"
        )
    plot_path = output_path / "plots"
    result_path = output_path / "results"
    cache_path = output_path / "cache"
    for path in [plot_path, result_path, cache_path]:
        path.mkdir(parents=True, exist_ok=True)

    # save configuration
    with open(output_path / "args.txt", "w") as f:
        f.write(pformat(vars(args)))

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(output_path / "log.log"),
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
            if Path(path).is_dir()
        ]
        args.test_dirs.remove(args.train_dir)
        args.test_dirs.remove(str(args.real_dir))
        args.test_dirs.insert(0, str(args.real_dir))  # make sure real comes first
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

    # TRAINING =========================================================================
    logger.info("TRAINING")

    # get train data (without perturbation)
    if args.dataset == "coco2014train":
        resize_to = 512
    elif args.dataset == "hyperkvasir":
        resize_to = 256
    else:
        resize_to = None
    path_fake_train = args.data_path / args.dataset / args.train_dir / "train"

    train_ds = load_dataset(
        path_class1=path_fake_train,
        path_class2="",
        sample_label=False,
        perturbation=train_perturbation,
        num_samples=args.num_train,
        num_workers=args.num_workers,
        resize_to=resize_to,
    )
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # train Attribution Method
    if args.attr == "fingerprint":
        attribution_wrapper = ATTRIBUTION_WRAPPERS["fingerprint"](
            num_samples=args.num_train
        )
    else:
        if args.model == "klwgan":
            if args.dataset.startswith("redwine"):
                output_dim = 11
            elif args.dataset.startswith("whitewine"):
                output_dim = 11
            elif args.dataset.startswith("parkinsons"):
                output_dim = 15
            generator_wrapper = WRAPPERS[args.model](
                checkpoint_path=args.checkpoint_path, output_dim=output_dim
            )
        else:
            generator_wrapper = WRAPPERS[args.model](
                checkpoint_path=args.checkpoint_path
            )

        if args.model == "stablediffusion":
            if args.use_ae:
                attribution_wrapper = ATTRIBUTION_WRAPPERS[args.attr](
                    generator_wrapper=generator_wrapper,
                    lr=args.lr_inv,
                    num_inits=args.num_inits_inv,
                    num_steps=args.num_steps_inv,
                    ae=generator_wrapper.ae.cuda(),
                )
            else:
                attribution_wrapper = ATTRIBUTION_WRAPPERS[args.attr](
                    generator_wrapper=generator_wrapper,
                    lr=args.lr_inv,
                    num_inits=args.num_inits_inv,
                    num_steps=args.num_steps_inv,
                )
        else:
            attribution_wrapper = ATTRIBUTION_WRAPPERS[args.attr](
                generator_wrapper=generator_wrapper,
                lr=args.lr_inv,
                num_inits=args.num_inits_inv,
                num_steps=args.num_steps_inv,
            )

    time_train_start = time()
    attribution_wrapper.train(train_loader)
    time_train_end = time()
    if args.save_time:
        time_train = time_train_end - time_train_start
        logger.info(f"Training took {time_train} seconds for {args.num_train} samples.")
    del train_loader

    # COMPUTING THRESHOLDS =============================================================
    logger.info("COMPUTING THRESHOLDS")
    path_fake_val = args.data_path / args.dataset / args.train_dir / "val"
    if args.attr == "fingerprint":
        val_ds = load_dataset(
            path_class1=path_fake_val,
            path_class2="",
            perturbation=train_perturbation,
            sample_label=False,
            num_samples=args.num_val,
            num_workers=args.num_workers,
            resize_to=resize_to,
        )
    elif args.attr in ["l2_inversion", "inception_inversion"]:
        # inversion does not need any training set, therefore we replace the small validation set by the larger training
        # set to make it fair
        val_ds = load_dataset(
            path_class1=path_fake_train,
            path_class2="",
            perturbation=train_perturbation,
            sample_label=False,
            num_samples=args.num_train,
            num_workers=args.num_workers,
            resize_to=resize_to,
        )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    thresholds_path = cache_path / f"thresholds_{args.test_models}.pkl"
    if thresholds_path.exists() and args.load_cached:
        with open(thresholds_path, "rb") as f:
            thresholds = pickle.load(f)
        thresholds = thresholds
    else:
        val_path = cache_path / f"val_distances_{args.test_models}.npy"
        if val_path.exists() and args.load_cached:
            val_distance = np.load(val_path)
        else:
            val_distance = []
            time_validation_start = time()
            for batch in tqdm(val_loader, desc="Computing validation distances"):
                if args.attr == "fingerprint":
                    for img in batch[0]:
                        distance = attribution_wrapper.forward(img)
                        val_distance.append(distance)
                else:
                    # batchwise
                    distance = attribution_wrapper.forward(batch[0])
                    val_distance.append(distance)

            val_distance = np.array(val_distance)
            time_validation_end = time()
            if args.save_time:
                time_validation = time_validation_end - time_validation_start
                logger.info(
                    f"Validation took {time_validation} seconds for {args.num_val} samples."
                )
            if args.attr != "fingerprint":
                val_distance = np.concatenate(val_distance)

            np.save(cache_path / f"val_distances_{args.test_models}.npy", val_distance)

        thresholds = {}
        for fnr in args.tolerable_fnr:
            if args.attr == "fingerprint":
                thresholds[f"fnr={fnr}"] = np.quantile(val_distance, q=fnr).item()
            else:
                thresholds[f"fnr={fnr}"] = np.quantile(val_distance, q=1 - fnr).item()

        with open(thresholds_path, "wb") as f:
            pickle.dump(thresholds, f)

    logger.info(f"Thresholds: {thresholds}.")
    del val_loader

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
    # Compute anomaly scores for my GAN:
    path_fake_test = args.data_path / args.dataset / args.train_dir / "test"
    logger.info(f"Testing {args.train_dir}.")
    cache_distances = cache_path / f"{args.train_dir}_distances.npy"
    if cache_distances.exists() and args.load_cached:
        test_distances_mygan = np.load(cache_distances)
    else:
        test_ds = load_dataset(
            path_class1=path_fake_test,
            path_class2="",
            sample_label=False,
            perturbation=perturbation,
            num_samples=args.num_test,
            num_workers=args.num_workers,
            resize_to=resize_to,
        )
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_distances = []
        for batch in test_loader:
            output = attribution_wrapper.forward(batch[0])
            test_distances.append(output)
        test_distances_mygan = np.concatenate(test_distances)
        np.save(cache_path / f"{args.train_dir}_distances.npy", test_distances_mygan)

    # Compute anomaly scores for other GANs:
    results = []
    for test_dir in args.test_dirs:
        logger.info(f"Testing {test_dir}.")
        cache_distances = cache_path / f"{test_dir}_distances.npy"
        if cache_distances.exists() and args.load_cached:
            test_distances = np.load(cache_distances)
        else:
            path_other_test = args.data_path / args.dataset / test_dir / "test"

            test_ds = load_dataset(
                path_class1=path_other_test,
                path_class2="",
                sample_label=False,
                perturbation=perturbation,
                num_samples=args.num_test,
                num_workers=args.num_workers,
                resize_to=resize_to,
            )
            test_loader = DataLoader(
                dataset=test_ds,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            test_distances = []
            for batch in test_loader:
                output = attribution_wrapper.forward(batch[0])
                test_distances.append(output)
            test_distances = np.concatenate(test_distances)
            np.save(cache_path / f"{test_dir}_distances.npy", test_distances)

        y_test = np.concatenate(
            [np.ones(len(test_distances_mygan)), np.zeros(len(test_distances))]
        )
        test_distances = np.concatenate([test_distances_mygan, test_distances])
        accuracies = {}
        for threshold_name, threshold_value in thresholds.items():
            if args.attr == "fingerprint":
                # large correlation -> inlier
                y_test_pred = np.where(
                    test_distances > threshold_value, 1, 0
                )  # larger -> 1 = inlier
            else:
                # large reconstruction error -> outlier
                y_test_pred = np.where(
                    test_distances < threshold_value, 1, 0
                )  # smaller -> 1 = inlier
            accuracies[threshold_name] = accuracy_score(
                y_true=y_test, y_pred=y_test_pred
            )

        result = {
            "model": args.model,
            "checkpoint": str(args.checkpoint_path),
            "seed": args.seed,
            "my": args.train_dir,
            "other": test_dir,
            "perturbation": args.perturbation,
            "perturbation-param": args.perturbation_param,
        }
        result.update(accuracies)
        results.append(result)
        logger.info(result)

        if args.confusion_matrices:
            tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_pred).ravel()
            # TODO plot
            logger.info(f"{test_dir}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    for result in results:
        with open(result_path / "results.json", "a") as f:
            json.dump(result, f)
            f.write(",\n")

    # Compute AUC
    num_other_test = int(args.num_test / len(args.test_dirs))
    test_distances = np.load(cache_path / f"{args.train_dir}_distances.npy")
    y_test = np.ones(len(test_distances))
    for test_dir in args.test_dirs:
        cache_distances = cache_path / f"{test_dir}_distances.npy"
        dist_other_test = np.load(cache_distances)
        test_distances = np.concatenate(
            [test_distances, dist_other_test[:num_other_test]]
        )
        y_test = np.concatenate([y_test, np.zeros(num_other_test)])

    if args.attr == "fingerprint":
        # high test_distance -> inlier class
        auc = roc_auc_score(y_test, test_distances)
    else:
        # high test_distance -> outlier class
        auc = roc_auc_score(y_test, -test_distances)
    logger.info(f"AUC: {auc}")
    np.save(result_path / f"auc_{args.test_models}.npy", auc)


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
        "--use-ae",
        action="store_true",
        help="If set, use AE in SD-experiments to invert the image.",
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
    # attribution method
    parser.add_argument(
        "--attr",
        choices=["l2_inversion", "inception_inversion", "fingerprint"],
        help="Attribution method to use.",
    )
    parser.add_argument(
        "--load-cached",
        action="store_true",
        help="Activate to load cached distances and thresholds.",
    )
    parser.add_argument(
        "--lr_inv",
        type=float,
        default=0.1,
        help="Learning rate in SGD-based Inversion.",
    )
    parser.add_argument(
        "--num-inits-inv",
        type=int,
        default=10,
        help="Number of initializations in Inversion.",
    )
    parser.add_argument(
        "--num-steps-inv",
        type=int,
        default=1000,
        help="Number of iteration steps in Inversion.",
    )
    parser.add_argument("--batch-size", type=int, default=256)

    # thresholds
    parser.add_argument(
        "--tolerable-fnr",
        type=float,
        nargs="*",
        default=[0.005, 0.01, 0.05, 0.1, 0.0, 0.001],
        help="False negative rates for which to compute thresholds.",
    )

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

    # technical
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=-1)

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
    start_time = time()
    main(parse_args())
    end_time = time()
    seconds = end_time - start_time
    print(f"Computations took {seconds // 60} Minutes.")
