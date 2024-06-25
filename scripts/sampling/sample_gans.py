import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from flipad.data_utils import get_data_loader
from flipad.networks.dcgan import DCGANGenerator
from flipad.networks.ebgan import EBGANGenerator
from flipad.networks.lsgan import LSGANGenerator
from flipad.networks.wgan import WGANGenerator
from flipad.utils import float2image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

GENERATORS = ["dcgan", "wgangp", "lsgan", "ebgan"]
DATASET = ["celeba", "lsun"]
EPS = 0.0001  # constant used for numerical stability: inverting tanh at 1 gives infity
BATCH_SIZE = 128
n_jobs_dataloader = 4


def main(data_root):
    torch.manual_seed(33)

    output_channels = 3
    # configs:
    nz = 100
    ngf = 64

    train_set_size = 10000
    test_set_size = 1000
    val_set_size = 1000
    batch_size = 500

    for iter in range(5):
        for data in DATASET:
            dcgan = DCGANGenerator(nz, ngf, output_channels, 1).to(device)
            wgangp = WGANGenerator(nz, ngf, output_channels).to(device)
            lsgan = LSGANGenerator(nz, output_channels).to(device)
            ebgan = EBGANGenerator(nz, output_channels).to(device)
            if data == "celeba":
                n_iters = {"dcgan": 50, "wgangp": 200, "lsgan": 100, "ebgan": 100}

            elif data == "lsun":
                n_iters = {"dcgan": 10, "wgangp": 10, "lsgan": 10, "ebgan": 5}

            generator_models = {
                "dcgan": dcgan,
                "wgangp": wgangp,
                "lsgan": lsgan,
                "ebgan": ebgan,
            }

            # real images:
            os.makedirs(f"data/{data}_{iter}/real/train", exist_ok=True)
            os.makedirs(f"data/{data}_{iter}/real/test", exist_ok=True)
            os.makedirs(f"data/{data}_{iter}/real/val", exist_ok=True)
            data_loader = get_data_loader(batch_size, data=data, root=data_root)
            img_counter = 0
            for batch in data_loader:
                imgs = batch[0]
                for img in imgs:
                    if img_counter < train_set_size:
                        PATH_img = f"data/{data}_{iter}/real/train/{img_counter}.png"
                    elif img_counter < train_set_size + test_set_size:
                        PATH_img = f"data/{data}_{iter}/real/test/{img_counter - train_set_size}.png"
                    else:
                        PATH_img = f"data/{data}_{iter}/real/val/{img_counter - train_set_size - test_set_size}.png"

                    save_image(float2image(img), PATH_img)
                    img_counter += 1
                    if img_counter == train_set_size + test_set_size + val_set_size:
                        break
                if img_counter == train_set_size + test_set_size + val_set_size:
                    break

            for modelnr in tqdm(range(1, 6)):
                #######################################################
                # load model, infer type of last layer and parameters #
                #######################################################
                for generator_name in tqdm(generator_models.keys()):
                    PATH_MODEL = os.path.join(
                        "trained_models",
                        generator_name,
                        data,
                        f"nz={nz}_niter={n_iters[generator_name]}_model={modelnr}",
                        "checkpoints",
                        f"netG_epoch_{n_iters[generator_name] - 1}.pth",
                    )
                    generator = generator_models[generator_name]
                    generator.load_state_dict(torch.load(PATH_MODEL))
                    generator_models[generator_name] = generator

                    setup_name = f"{generator_name}_{modelnr}"
                    os.makedirs(f"data/{data}_{iter}/{setup_name}/train", exist_ok=True)
                    os.makedirs(f"data/{data}_{iter}/{setup_name}/test", exist_ok=True)
                    os.makedirs(f"data/{data}_{iter}/{setup_name}/val", exist_ok=True)
                    img_counter = 0
                    while img_counter < train_set_size + test_set_size + val_set_size:
                        # sample noise
                        if generator_name in ["wgangp", "dcgan"]:
                            noise = torch.randn(batch_size, 100, 1, 1)
                        else:
                            noise = torch.randn(batch_size, 100)
                        # generate data
                        imgs = generator_models[generator_name](noise.to(device))
                        # save data
                        for img in imgs:
                            if img_counter < train_set_size:
                                PATH_img = f"data/{data}_{iter}/{setup_name}/train/{img_counter}.png"
                            elif img_counter < train_set_size + test_set_size:
                                PATH_img = f"data/{data}_{iter}/{setup_name}/test/{img_counter - train_set_size}.png"
                            else:
                                PATH_img = f"data/{data}_{iter}/{setup_name}/val/{img_counter - train_set_size - test_set_size}.png"

                            save_image(float2image(img), PATH_img)
                            img_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    main(**vars(parser.parse_args()))
