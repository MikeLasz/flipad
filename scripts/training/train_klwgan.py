"""
Taken from https://github.com/ermongroup/f-wgan/blob/master/synthetic/kl_wgan_real.py
"""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import tqdm
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import euclidean_distances

from flipad.networks.klwgan import KLWGANDiscriminator, KLWGANGenerator

import flipad.klwgan_data_utils as D


def mmd(X, Y):
    dxy = euclidean_distances(X, Y) / X.shape[1]
    dxx = euclidean_distances(X, X) / X.shape[1]
    dyy = euclidean_distances(Y, Y) / X.shape[1]
    kxy = np.exp(-(dxy**2))
    kxx = np.exp(-(dxx**2))
    kyy = np.exp(-(dyy**2))
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def kde(X, Y):
    qx = gaussian_kde(X.T, bw_method="scott")
    return qx.logpdf(Y.T).mean()


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


def hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1.0 - dis_real))
    loss_fake = torch.mean(F.relu(1.0 + dis_fake))
    return loss_real, loss_fake


def hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def kl_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1.0 - dis_real))
    dis_fake_norm = torch.exp(dis_fake).mean()
    dis_fake_ratio = torch.exp(dis_fake) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss_fake = torch.mean(F.relu(1.0 + dis_fake))
    return loss_real, loss_fake


def kl_gen(dis_fake):
    dis_fake_norm = torch.exp(dis_fake).mean()
    dis_fake_ratio = torch.exp(dis_fake) / dis_fake_norm
    dis_fake = dis_fake * dis_fake_ratio
    loss = -torch.mean(dis_fake)
    return loss


def train_network(
    name, num_epochs=500, loss_type="hinge", device="cuda", save_id=0, load_model=None
):
    print(f"Running {name}, {loss_type}")
    model_save_path = f"trained_models/klwgan-{loss_type}/"
    os.makedirs(model_save_path, exist_ok=True)

    dataset = name_to_dataset[name]()

    train_samples = dataset.data
    val_samples = dataset.test_data
    train_loader = data.DataLoader(
        data.TensorDataset(torch.from_numpy(train_samples).float()), batch_size=256
    )
    val_loader = data.DataLoader(
        data.TensorDataset(torch.from_numpy(val_samples).float()), batch_size=256
    )
    input_dim = train_samples.shape[-1]

    g_net = KLWGANGenerator(input_dim).to(device)
    g_optim = optim.RMSprop(g_net.parameters(), lr=0.0002)
    d_net = KLWGANDiscriminator(input_dim).to(device)
    d_optim = optim.RMSprop(d_net.parameters(), lr=0.0002)

    noise_dim = g_net.noise_dim
    if load_model and os.path.exists(f"{model_save_path}/{args.data}_{save_id}.pkl"):
        g_net.load_state_dict(
            torch.load(f"{model_save_path}/{args.data}_{save_id}.pkl")
        )
    else:
        for e in tqdm.tqdm(range(num_epochs)):
            for x in train_loader:
                x = x[0].to(device)
                bs = x.size(0)
                for i in range(5):
                    z = torch.randn(bs, noise_dim).to(device)
                    d_optim.zero_grad()
                    with torch.no_grad():
                        G_x = g_net(z)
                    dis_real = d_net(x)
                    dis_fake = d_net(G_x)
                    if loss_type == "kl":
                        d_loss_real, d_loss_fake = kl_dis(dis_fake, dis_real)
                    elif loss_type == "hinge":
                        d_loss_real, d_loss_fake = hinge_dis(dis_fake, dis_real)
                    else:
                        raise ValueError("loss_type")
                    d_loss = d_loss_fake + d_loss_real
                    d_loss = d_loss.mean()
                    d_loss.backward()
                    d_optim.step()

                g_optim.zero_grad()
                z = torch.randn(bs, noise_dim).to(device)
                G_x = g_net(z)
                dis_fake = d_net(G_x)
                if loss_type == "kl":
                    g_loss = kl_gen(dis_fake)
                elif loss_type == "hinge":
                    g_loss = hinge_gen(dis_fake)
                else:
                    raise ValueError("loss_type")
                g_loss = g_loss.mean()
                g_loss.backward()
                g_optim.step()
            if e % 1 == 0 and e > 0:
                with torch.no_grad():
                    z = torch.randn(10000, noise_dim).to("cuda")
                    G_x = g_net(z).cpu().numpy()

                    print(kde(G_x, dataset.test_data[:10000]))
                    print(mmd(G_x, dataset.test_data[:10000]))

    # save the generated samples
    with torch.no_grad():
        z = torch.randn(100000, noise_dim).to("cuda")
        data_train = g_net(z).cpu().numpy()
        z = torch.randn(10000, noise_dim).to("cuda")
        data_test = g_net(z).cpu().numpy()
        z = torch.randn(10000, noise_dim).to("cuda")
        data_val = g_net(z).cpu().numpy()

    synthetic_data = {"test": data_test, "train": data_train, "val": data_val}

    for subdir in ["test", "train", "val"]:
        samples_save_path = f"data/{name}_{save_id}/klwgan-{loss_type}/{subdir}"
        os.makedirs(samples_save_path, exist_ok=True)
        np.savetxt(
            samples_save_path + "/data.csv", synthetic_data[subdir], delimiter=","
        )

    torch.save(g_net.state_dict(), model_save_path + f"{args.data}_{save_id}.pkl")

    # save finale test scores:
    kde_score = kde(data_test, dataset.test_data[:10000])
    mmd_score = mmd(data_test, dataset.test_data[:10000])
    with open(f"{model_save_path}{name}_{save_id}_scores.txt", "w") as f:
        f.write(f"KDE={kde_score}; MMD={mmd_score}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--load_model", action="store_true")
    args = parser.parse_args()
    for name in [args.data]:
        for save_id in range(5):
            loss_type = "hinge"
            train_network(
                name,
                num_epochs=500,
                loss_type=loss_type,
                save_id=save_id,
                load_model=args.load_model,
            )
