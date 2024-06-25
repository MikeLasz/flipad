# import medigan and initialize Generators
import os

from pathlib import Path

import torch

from medigan import Generators  # pip install medigan


generators = Generators()

MODEL_IDS = {
    "dcgan_bcdr": 5,
    "wgangp_bcdr": 6,
    "cdgan_bcdr": 12,
}


if __name__ == "__main__":
    samples_sizes = {"train": 10000, "val": 1000, "test": 1000}
    num_repeat = 5  # number of times to repeat the sampling process
    for num in range(num_repeat):
        for setting in samples_sizes.keys():
            n = samples_sizes[setting]
            data_path = Path("data") / f"breastmass_{num}"
            for model in MODEL_IDS.keys():
                (data_path / model / setting).mkdir(exist_ok=True, parents=True)

                for model in MODEL_IDS.keys():
                    print("Generating {} samples for {}...".format(n, model))
                    model_id = MODEL_IDS[model]
                    generators.generate(
                        model_id=model_id,
                        num_samples=n,
                        install_dependencies=True,
                        output_path=data_path / model / setting,
                    )

    # save generative models
    for model in MODEL_IDS.keys():
        model_id = MODEL_IDS[model]
        me = generators.get_model_executor(model_id=model_id, install_dependencies=True)
        if model_id == 12:
            model_instance = (
                me.deserialized_model_as_lib.Generator(
                    nz=100,
                    ngf=64,
                    nc=2,
                    ngpu=1,
                    image_size=128,
                    leakiness=0.1,
                    conditional=True,
                )
                .cuda()
                .eval()
            )
        else:
            model_instance = (
                me.deserialized_model_as_lib.Generator(
                    nz=100,
                    ngf=64,
                    nc=1,
                    ngpu=1,
                    image_size=128,
                    leakiness=0.1,
                    conditional=False,
                )
                .cuda()
                .eval()
            )
        model_instance.load_state_dict(
            state_dict=torch.load(me.package_path, map_location="cuda")["generator"]
        )
        os.makedirs(f"trained_models/{model}/", exist_ok=True)
        # save model (for testing whether wrapper is working correctly)
        # torch.save(model_instance, f"trained_models/{model}/model.pt")
        # save model state dict
        torch.save(
            model_instance.state_dict(), f"trained_models/{model}/model_state_dict.pt"
        )
