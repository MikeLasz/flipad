#!/bin/bash
# Samples from Stable Diffusion 2.1: Train, Test, Val, Average Act.
python3 scripts/sampling/sample_stable_diffusion.py --model stabilityai/stable-diffusion-2-1-base --out data/coco2014train/stable-diffusion-2-1-base/train --prompt-file data/coco2014train/real/train/captions.txt 
python3 scripts/sampling/sample_stable_diffusion.py --model stabilityai/stable-diffusion-2-1-base --out data/coco2014train/stable-diffusion-2-1-base/val --prompt-file data/coco2014train/real/val/captions.txt --num-samples 100
python3 scripts/sampling/sample_stable_diffusion.py --model stabilityai/stable-diffusion-2-1-base --out data/coco2014train/stable-diffusion-2-1-base/test --prompt-file data/coco2014train/real/test/captions.txt --num-samples 200
python3 scripts/sampling/sample_stable_diffusion.py --model stabilityai/stable-diffusion-2-1-base --out data/coco2014train/stable-diffusion-2-1-base/avg_act --prompt-file data/coco2014train/real/avg_act/captions.txt --avg-act

# Test Samples from Stable Diffusion 1.1, 1.1+, 1.4, 2.0
python3 scripts/sampling/sample_stable_diffusion.py --model stabilityai/stable-diffusion-2-base --out data/coco2014train/stable-diffusion-2-base/test --prompt-file data/coco2014train/real/test/captions.txt --seed 4 --num-samples 200
python3 scripts/sampling/sample_stable_diffusion.py --model CompVis/stable-diffusion-v1-4 --out data/coco2014train/stable-diffusion-v1-4/test --prompt-file data/coco2014train/real/test/captions.txt --seed 5 --num-samples 200
python3 scripts/sampling/sample_stable_diffusion.py --model CompVis/stable-diffusion-v1-1 --out data/coco2014train/stable-diffusion-v1-1/test --prompt-file data/coco2014train/real/test/captions.txt  --seed 6 --num-samples 200
python3 scripts/sampling/sample_stable_diffusion.py --model CompVis/stable-diffusion-v1-1 --out data/coco2014train/stable-diffusion-v1-1-vae/test --prompt-file data/coco2014train/real/test/captions.txt --alternative-vae --seed 23 --num-samples 200
