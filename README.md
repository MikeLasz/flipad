# [ICML 2024] Single-Model Attribution of Generative Models Through Final-Layer Inversion

This repository will contain the code for reproducing the experiments in the paper [Single-Model Attribution of Generative Models Through Final-Layer Inversion](https://arxiv.org/abs/2306.06210) by Mike Laszkiewicz, Jonas Ricker, Johannes Lederer, and Asja Fischer.

This paper has been accepted at ICML 2024.

> Recent breakthroughs in generative modeling have sparked interest in practical single-model attribution. Such methods predict whether a sample was generated by a specific generator or not, for instance, to prove intellectual property theft. However, previous works are either limited to the closed-world setting or require undesirable changes to the generative model. We address these shortcomings by, first, viewing single-model attribution through the lens of anomaly detection. Arising from this change of perspective, we propose FLIPAD, a new approach for single-model attribution in the open-world setting based on final-layer inversion and anomaly detection. We show that the utilized final-layer inversion can be reduced to a convex lasso optimization problem, making our approach theoretically sound and computationally efficient. The theoretical findings are accompanied by an experimental study demonstrating the effectiveness of our approach and its flexibility to various domains.


## Setup
Set up and activate a virtual environment (we tested with Python 3.8, but newer versions should work as well) and install the required packages with 
```
pip install -r requirements.txt
pip install -e .
```


## Repository Structure
The scripts for reproducing the results from the paper are located in `scripts`, which contains the training scripts (`scripts/training`) of the generative models, the sampling scripts (`scripts/sampling`) for creating synthetic data, and the scripts for performing single-model attribution (`scripts/single_model_attribution`). 
The single-model attribution scripts call `sma_deepsad.py`, `sma_deepsad_tabular.py`, or `sma_deepsad_baselines.py` using the arguments as specified in the paper. 

# Reproducing the Experiments
## Data Preparation
### GANs
Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [LSUN Bedroom](https://www.yf.io/p/lsun) to a folder which we will refer to as `DATA_ROOT`. It should have the following structure:
```
DATA_ROOT
├── bedroom_train_lmdb
│   ├── data.mdb
│   └── lock.mdb
└── celeba
    ├── identity_CelebA.txt
    ├── img_align_celeba
    ├── img_align_celeba.zip
    ├── list_attr_celeba.txt
    ├── list_bbox_celeba.txt
    ├── list_eval_partition.txt
    └── list_landmarks_align_celeba.txt
```

To train DCGAN, WGAN-GP, LSGAN, and EBGAN on these datasets, run `./scripts/training/train_small_gans.sh `.

Then, run `python3 scripts/sampling/sample_gans.py data` to save real and generated samples in `data`.

### Stable Diffusion
To generate images using different versions of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) run `./scripts/sampling/sample_stable_diffusion.sh`.

### Style-based Generative Models 
The pretrained style-based models, including their sampling commands, can be found in their official implementations:

| Model       | Link                                            |
|-------------|-------------------------------------------------|
| StyleGAN2   | https://github.com/NVlabs/stylegan3             |
| StyleGAN-XL | https://github.com/autonomousvision/stylegan-xl |
| StyleNAT    | https://github.com/SHI-Labs/StyleNAT            |
| StyleSwin   | https://github.com/microsoft/StyleSwin          |

### Medical Image Generative Models
We use the pretrained models provided by the [medigan](https://github.com/RichardObi/medigan) library. 
To generate images using medigan, run `sample_medigan.sh`.


### Generative Models for Tabular Data
To train all models, run `./scripts/training/train_tabular_models.sh `. This script also generates train, test, and validation data.  

## Single-Model Attribution
To reproduce the results from the paper, please execute the following scripts:
- Table 1 
```
./scripts/single_model_attribution/gans.sh
```

- Table 2
``` 
./scripts/single_model_attribution/gans_same.sh
```

- Table 3
```
./scripts/single_model_attribution/gans_perturbation.sh 
```

- Table 4 
```
./scripts/single_model_attribution/sd.sh 
./scripts/single_model_attribution/stylegan.sh 
./scripts/single_model_attribution/medigan.sh 
```

- Table 5
```
./scripts/single_model_attribution/tabular.sh 
```

To summarize and evaluate all experiments, we provide the jupyter notebook `evaluate_results.ipynb`. 
