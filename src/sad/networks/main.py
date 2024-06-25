from .dgm import DeepGenerativeModel, StackedDeepGenerativeModel
from .fmnist_LeNet import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder
from .LeNet import (
    CIFAR10_BigLeNet,
    CIFAR10_LeNet,
    CIFAR10_LeNet_Decoder,
    CIFAR10_LeNet_Autoencoder,
    LeNet_64Channels,
    LeNet_128x128,
    LeNet_64Channels_Decoder,
    LeNet_64Channels_Autoencoder
)
from .mlp import MLP, MLP_Autoencoder
from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .vae import VariationalAutoencoder


def build_network(net_name, ae_net=None, num_select_channels=0, rep_dim=128, kwargs=None):
    """Builds the neural network."""

    implemented_networks = (
        "mnist_LeNet",
        "mnist_DGM_M2",
        "mnist_DGM_M1M2",
        "fmnist_LeNet",
        "fmnist_DGM_M2",
        "fmnist_DGM_M1M2",
        "cifar10_LeNet",
        "cifar10_DGM_M2",
        "cifar10_DGM_M1M2",
        "arrhythmia_mlp",
        "cardio_mlp",
        "satellite_mlp",
        "satimage-2_mlp",
        "shuttle_mlp",
        "thyroid_mlp",
        "arrhythmia_DGM_M2",
        "cardio_DGM_M2",
        "satellite_DGM_M2",
        "satimage-2_DGM_M2",
        "shuttle_DGM_M2",
        "thyroid_DGM_M2",
        "lenet_64channels",
        "lenet_128x128",
        "cifar10_biglenet",
        "lenet_64channels_medigan",
        "lenet_128x128_medigan",
        "mlp"
    )
    assert net_name in implemented_networks

    net = None
    if net_name == "lenet_128x128":
        net = LeNet_128x128(input_channels=num_select_channels, rep_dim=rep_dim)

    if net_name == "lenet_64channels":
        net = LeNet_64Channels()

    if net_name == "lenet_64channels_medigan":
        net = LeNet_64Channels(input_spatial_size=64)

    if net_name == "lenet_128x128_medigan":
        net = LeNet_128x128(input_channels=1)

    if net_name == "mlp": # tabular experiments
        net = MLP(x_dim=num_select_channels, bias=kwargs["use_bias"], h_dims=kwargs["h_dims"], rep_dim=rep_dim, dropout=kwargs["dropout"])

    if net_name == "mnist_LeNet":
        net = MNIST_LeNet()

    if net_name == "cifar10_biglenet":
        net = CIFAR10_BigLeNet(rep_dim=rep_dim, input_channels=num_select_channels)

    if net_name == "mnist_DGM_M2":
        net = DeepGenerativeModel([1 * 28 * 28, 2, 32, [128, 64]], classifier_net=MNIST_LeNet)

    if net_name == "mnist_DGM_M1M2":
        net = StackedDeepGenerativeModel([1 * 28 * 28, 2, 32, [128, 64]], features=ae_net)

    if net_name == "fmnist_LeNet":
        net = FashionMNIST_LeNet()

    if net_name == "fmnist_DGM_M2":
        net = DeepGenerativeModel([1 * 28 * 28, 2, 64, [256, 128]], classifier_net=FashionMNIST_LeNet)

    if net_name == "fmnist_DGM_M1M2":
        net = StackedDeepGenerativeModel([1 * 28 * 28, 2, 64, [256, 128]], features=ae_net)

    if net_name == "cifar10_LeNet":
        net = CIFAR10_LeNet()

    if net_name == "cifar10_DGM_M2":
        net = DeepGenerativeModel([3 * 32 * 32, 2, 128, [512, 256]], classifier_net=CIFAR10_LeNet)

    if net_name == "cifar10_DGM_M1M2":
        net = StackedDeepGenerativeModel([3 * 32 * 32, 2, 128, [512, 256]], features=ae_net)

    if net_name == "arrhythmia_mlp":
        net = MLP(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == "cardio_mlp":
        net = MLP(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == "satellite_mlp":
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == "satimage-2_mlp":
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == "shuttle_mlp":
        net = MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == "thyroid_mlp":
        net = MLP(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)

    if net_name == "arrhythmia_DGM_M2":
        net = DeepGenerativeModel([274, 2, 32, [128, 64]])

    if net_name == "cardio_DGM_M2":
        net = DeepGenerativeModel([21, 2, 8, [32, 16]])

    if net_name == "satellite_DGM_M2":
        net = DeepGenerativeModel([36, 2, 8, [32, 16]])

    if net_name == "satimage-2_DGM_M2":
        net = DeepGenerativeModel([36, 2, 8, [32, 16]])

    if net_name == "shuttle_DGM_M2":
        net = DeepGenerativeModel([9, 2, 8, [32, 16]])

    if net_name == "thyroid_DGM_M2":
        net = DeepGenerativeModel([6, 2, 4, [32, 16]])

    return net


def build_autoencoder(net_name, kwargs=None):
    """Builds the corresponding autoencoder network."""

    implemented_networks = (
        "mnist_LeNet",
        "mnist_DGM_M1M2",
        "fmnist_LeNet",
        "fmnist_DGM_M1M2",
        "cifar10_LeNet",
        "cifar10_DGM_M1M2",
        "arrhythmia_mlp",
        "cardio_mlp",
        "satellite_mlp",
        "satimage-2_mlp",
        "shuttle_mlp",
        "thyroid_mlp",
        "mlp"
    )

    assert net_name in implemented_networks

    ae_net = None

    if net_name == "mnist_LeNet":
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == "mnist_DGM_M1M2":
        ae_net = VariationalAutoencoder([1 * 28 * 28, 32, [128, 64]])

    if net_name == "fmnist_LeNet":
        ae_net = FashionMNIST_LeNet_Autoencoder()

    if net_name == "fmnist_DGM_M1M2":
        ae_net = VariationalAutoencoder([1 * 28 * 28, 64, [256, 128]])

    if net_name == "cifar10_LeNet":
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == "cifar10_DGM_M1M2":
        ae_net = VariationalAutoencoder([3 * 32 * 32, 128, [512, 256]])

    if net_name == "arrhythmia_mlp":
        ae_net = MLP_Autoencoder(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == "cardio_mlp":
        ae_net = MLP_Autoencoder(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == "satellite_mlp":
        ae_net = MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == "satimage-2_mlp":
        ae_net = MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == "shuttle_mlp":
        ae_net = MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == "thyroid_mlp":
        ae_net = MLP_Autoencoder(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)

    if net_name == "mlp":
        ae_net = MLP_Autoencoder(x_dim=kwargs["x_dim"], h_dims=kwargs["h_dims"], rep_dim=kwargs["rep_dim"],
                                 bias=kwargs["use_bias"])

    return ae_net
