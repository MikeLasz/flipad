import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL

MODELS = [
    "CompVis/stable-diffusion-v1-1",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-2-1-base",
]


def patched_decode_latents(self, latents):
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    return image


def main(
    model: str,
    prompt_file: Path,
    out: Path,
    avg_act: bool,
    sample_act: bool,
    batch_size: int,
    seed: int,
    alternative_vae: bool = False,
    num_samples: int = math.inf,
):
    out.mkdir(exist_ok=True, parents=True)

    if alternative_vae:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        pipe = StableDiffusionPipeline.from_pretrained(
            model, safety_checker=None, vae=vae
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model, safety_checker=None)
    pipe = pipe.to("cuda")

    if avg_act or sample_act:
        pipe.components["vae"].decoder.conv_out = torch.nn.Identity()
        pipe.__class__.decode_latents = patched_decode_latents
        pipe.__class__.numpy_to_pil = lambda self, img: img
        avg_activation = None

    with open(prompt_file, "r") as f:
        prompts = f.readlines()

    i = 0
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_samples = min(len(prompts), num_samples)

    metrics_dict = {
        "overall": {"mean": None, "var": None},
        "per_sample_and_channel": {"mean": None, "var": None},
        "per_sample": {"mean": None, "var": None},
    }
    sum = 0  # used for running variance computations
    sum_sq = 0  # used for running variance computations
    sum_sample_and_channel = 0  # used for running variance computations
    sum_sq_sample_and_channel = 0  # used for running variance computations
    sum_overall = 0  # used for running variance computations
    sum_sq_overall = 0  # used for running variance computations
    while True:
        start = i
        end = i + batch_size if i + batch_size < num_samples else num_samples
        prompt_batch = prompts[start:end]

        print(f"Generating from prompts {start} to {end - 1}.")

        images = pipe(prompt_batch, generator=generator).images

        if avg_act:
            """
            We store 3 different versions of mean and sd:
            i) Overall mean and sd, i.e. both are tensors in R
            ii) mean and sd over all samples and channels, i.e. both are tensors in R^(width x height)
            iii) mean and sd over all samples, i.e. both are tensors in R^(channels x width x height)
            """
            if metrics_dict["per_sample"]["mean"] is None:
                metrics_dict["per_sample"]["mean"] = torch.zeros(
                    images.shape[1:]
                ).cuda()

            metrics_dict["per_sample"]["mean"] = running_update_mean(
                images, metrics_dict["per_sample"]["mean"], i
            )
            metrics_dict["per_sample"]["var"], sum, sum_sq = running_update_var(
                images, sum, sum_sq, i
            )

            # per channel and sample mean, and overall can be computed as mean over per sample at the end
            (
                metrics_dict["per_sample_and_channel"]["var"],
                sum_sample_and_channel,
                sum_sq_sample_and_channel,
            ) = running_update_var(
                images, sum_sample_and_channel, sum_sq_sample_and_channel, i
            )
            metrics_dict["overall"]["var"], sum_overall, sum_sq_overall = (
                running_update_var(images, sum_overall, sum_sq_overall, i)
            )

            i += len(images)
        elif sample_act:
            # save activations:
            for act in images:
                torch.save(act.clone(), out / f"{i:06}.pt")
                i += 1
        else:
            for image in images:
                image.save(out / f"{i:06}.png")
                i += 1

        if end == num_samples:
            break

    if avg_act:
        metrics_dict["per_sample_and_channel"]["mean"] = metrics_dict["per_sample"][
            "mean"
        ].mean(dim=0)
        metrics_dict["overall"] = metrics_dict["per_sample"]["mean"].mean()

        torch.save(metrics_dict, out / f"metrics-{model.split('/')[-1]}-samples={i}.pt")

        # torch.save(avg_activation, out / f"avg_act-{model.split('/')[-1]}-samples={i}.pt")


def running_update_mean(x, mean, n):
    return (n * mean + x.sum(dim=0)) / (n + len(x))


def running_update_var(x, sum, sum_sq, n):
    sum += x.sum(dim=0)
    sum_sq += (x * x).sum(dim=0)
    var = (sum_sq - (sum * sum) / (n + len(x))) / (n + len(x))
    return var, sum, sum_sq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--avg-act", action="store_true")
    parser.add_argument("--sample-act", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alternative-vae", action="store_true")
    parser.add_argument("--num-samples", type=int, default=math.inf)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
