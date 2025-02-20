import os, sys
from datetime import datetime
sys.path.append("..")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision
import gc

from src import distributions
import torch.nn.functional as F
from diffusers import AutoencoderKL

from src.resnet2 import ResNet_D
from src.unet import UNet

from src.tools import unfreeze, freeze
from src.tools import weights_init_D
from src.tools import load_latents_dataset, get_latent_Z_pushed_loader_stats
from src.fid_score import calculate_frechet_distance
from src.plotters import plot_random_latent_images, plot_latent_images

from copy import deepcopy
import json

from tqdm import tqdm

import wandb # <--- online logging of the results
from src.tools import fig2data, fig2img # for wandb

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

################
# CONFIG

DEVICE_IDS = [0]
NUM_WORKERS = 8 * len(DEVICE_IDS)

# DATASET1, DATASET1_PATH = 'handbag', '../../data/handbag_128.hdf5'
# DATASET2, DATASET2_PATH = 'shoes', '../../data/shoes_128.hdf5'

DATASET1, DATASET1_PATH = 'celeba_female', '/gpfs/gpfs0/groznyy.sergey/Datasets/celeba_preprocessed_512'
DATASET2, DATASET2_PATH = 'aligned_anime_faces', '/gpfs/gpfs0/groznyy.sergey/Datasets/anime_preprocessed_512_v2'

batch_scale = 1  # to try linear scaling rule

T_ITERS = 10
f_LR, T_LR = 1e-4 * np.sqrt(batch_scale), 1e-4 * np.sqrt(batch_scale)  # baseline is 1e-4
SRC_IMG_SIZE = 512
LATENTS_SIZE = 64

BATCH_SIZE = int(64 * batch_scale)  # baseline is 64

PLOT_INTERVAL = 100
CKPT_INTERVAL = 100
COST = 'mse' # Mean Squared Error
FID_INTERVAL = 1000
MAX_STEPS = 100001
SEED = 0x000000

EXP_NAME = f'{DATASET1}_{DATASET2}_T{T_ITERS}_{COST}_{LATENTS_SIZE}_BS{BATCH_SIZE}'
cur_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_PATH = '../checkpoints/{}/{}_{}_{}_{}/'.format(COST, DATASET1, DATASET2, LATENTS_SIZE, cur_datetime)

# END OF CONFIG
################


if __name__ == "__main__":
    config = dict(
        DATASET1=DATASET1,
        DATASET2=DATASET2,
        T_ITERS=T_ITERS,
        f_LR=f_LR, T_LR=T_LR,
        BATCH_SIZE=BATCH_SIZE,
        OUTPUT_PATH=OUTPUT_PATH,
    )

    assert torch.cuda.is_available()
    torch.cuda.set_device(f'cuda:{DEVICE_IDS[0]}')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    filename = '../stats/{}_{}_test.json'.format(DATASET2, SRC_IMG_SIZE)
    with open(filename, 'r') as fp:
        data_stats = json.load(fp)
        mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
    del data_stats

    X_sampler, X_test_sampler = load_latents_dataset(
        DATASET1_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )
    Y_sampler, Y_test_sampler = load_latents_dataset(
        DATASET2_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )

    torch.cuda.empty_cache()
    gc.collect()

    f = ResNet_D(size=LATENTS_SIZE, nc=3).cuda()
    f.apply(weights_init_D)

    T = UNet(3, 3, base_factor=48).cuda()

    vae_decoder = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae_decoder = vae_decoder.to(f'cuda:{DEVICE_IDS[0]}')
    freeze(vae_decoder)

    if len(DEVICE_IDS) > 1:
        T = nn.DataParallel(T, device_ids=DEVICE_IDS)
        f = nn.DataParallel(f, device_ids=DEVICE_IDS)
        vae_decoder = nn.DataParallel(vae_decoder, device_ids=DEVICE_IDS)


    print("===> Initialized networks:")
    print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
    print('f params:', np.sum([np.prod(p.shape) for p in f.parameters()]), "\n")

    torch.manual_seed(0xBADBEEF)
    np.random.seed(0xBADBEEF)
    X_fixed = X_sampler.sample(10)
    Y_fixed = Y_sampler.sample(10)
    X_test_fixed = X_test_sampler.sample(10)
    Y_test_fixed = Y_test_sampler.sample(10)

    #######
    # Training
    wandb.init(name=EXP_NAME, project='notreallyweakot', entity='metra4ok', config=config)

    T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)

    print(f"===> Learning rates:")
    print(f"\tT_LR = {T_LR}\n\tf_LR = {f_LR}\n\n")

    for step in tqdm(range(MAX_STEPS)):
        # T optimization
        unfreeze(T); freeze(f)
        potentials, costs, T_losses = [], [], []  # to compute means for logging
        for t_iter in range(T_ITERS):
            T_opt.zero_grad()
            X = X_sampler.sample(BATCH_SIZE)
            T_X = T(X)
            potential_T_X = f(T_X).mean()
            potentials.append(potential_T_X.item())
            if COST == 'mse':
                cost_value = F.mse_loss(X, T_X).mean()
                costs.append(cost_value.item())
            else:
                raise Exception('Unknown COST')
            T_loss = cost_value - potential_T_X
            T_losses.append(T_loss.item())
            T_loss.backward()
            T_opt.step()
        wandb.log(
            {
                "mean_f(T_X)": np.mean(potentials),
                "mean_Cost(X, T_X)": np.mean(costs),
                "mean_T_loss": np.mean(T_losses)
            },
            step=step,
        )
        del potentials, costs, T_losses, potential_T_X, cost_value, T_loss, T_X, X
        gc.collect()
        torch.cuda.empty_cache()

        # f optimization
        freeze(T); unfreeze(f)
        X = X_sampler.sample(BATCH_SIZE)
        with torch.no_grad():
            T_X = T(X)
        Y = Y_sampler.sample(BATCH_SIZE)
        f_opt.zero_grad()
        potential_T_X = f(T_X).mean()
        potential_Y = f(Y).mean()
        f_loss = potential_T_X - potential_Y
        f_loss.backward()
        f_opt.step()
        wandb.log(
            {
                "f(T_X)": potential_T_X.item(),
                "f(Y)": potential_Y.item(),
                "f_loss": f_loss.item(),
            },
            step=step,
        )
        del potential_T_X, potential_Y, f_loss, Y, X, T_X
        gc.collect()
        torch.cuda.empty_cache()

        if step % PLOT_INTERVAL == 0:
            print('Plotting')

            fig, axes = plot_latent_images(X_fixed, Y_fixed, T, vae_decoder)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close(fig)

            fig, axes = plot_random_latent_images(X_sampler,  Y_sampler, T, vae_decoder)
            wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close(fig)

            fig, axes = plot_latent_images(X_test_fixed, Y_test_fixed, T, vae_decoder)
            wandb.log({'Fixed Test Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close(fig)

            fig, axes = plot_random_latent_images(X_test_sampler, Y_test_sampler, T, vae_decoder)
            wandb.log({'Random Test Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close(fig)

        if step % CKPT_INTERVAL == CKPT_INTERVAL - 1:
            state_dict = {
                "T_state_dict": T.state_dict(),
                "f_state_dict": f.state_dict(),
                "T_opt_state_dict": T_opt.state_dict(),
                "f_opt_state_dict": f_opt.state_dict(),
            }
            ckpt_path = os.path.join(OUTPUT_PATH, f'{SEED}_{step}.pt')
            torch.save(state_dict, ckpt_path)
            # wandb.save(ckpt_path, base_path="../")

        if step % FID_INTERVAL == FID_INTERVAL - 1:
            freeze(T)

            print('Computing FID')
            mu, sigma = get_latent_Z_pushed_loader_stats(T, vae_decoder, X_test_sampler.loader)
            fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
            wandb.log({f'FID (Test)' : fid}, step=step)
            del mu, sigma

        gc.collect()
        torch.cuda.empty_cache()
