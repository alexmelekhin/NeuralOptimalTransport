import os, sys
sys.path.append("..")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import torchvision
import gc

from src import distributions
import torch.nn.functional as F
from diffusers import AutoencoderKL

from src.resnet2 import ResNet_D
from src.unet import UNet

from src.tools import unfreeze, freeze
from src.tools import load_latents_dataset, get_latent_Z_pushed_loader_stats
from src.fid_score import calculate_frechet_distance
from src.tools import weights_init_D
from src.plotters import plot_latent_Z_images, plot_random_latent_Z_images

from copy import deepcopy
import json

from tqdm import tqdm

import wandb
from src.tools import fig2data, fig2img # for wandb

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


################
# CONFIG

DEVICE_IDS = [0]
NUM_WORKERS = 8

DATASET1, DATASET1_PATH = 'celeba_female', '/mnt/ssd1/Datasets/celeba_preprocessed'
DATASET2, DATASET2_PATH = 'aligned_anime_faces', '/mnt/ssd1/Datasets/anime_preprocessed'

batch_scale = 1  # to try linear scaling rule

T_ITERS = 10
f_LR, T_LR = 1e-4 * batch_scale, 1e-4 * batch_scale  # baseline is 1e-4
SRC_IMG_SIZE = 128
LATENTS_SIZE = 16

ZC = 1
Z_STD = 0.1

BATCH_SIZE = int(64 * batch_scale)  # baseline is 64
Z_SIZE = 8

PLOT_INTERVAL = 100
COST = 'weak_mse'
CPKT_INTERVAL = 1000
MAX_STEPS = 100001
SEED = 0x000000

# Gamma will linearly increase from 0 to 0.66 during first 25k iters of the potential
GAMMA0, GAMMA1 = 0.0, 0.66
GAMMA_ITERS = int(25000 / batch_scale)  # TODO: not sure whether need to scale it

EXP_NAME = f'{DATASET1}_{DATASET2}_T{T_ITERS}_{COST}_{LATENTS_SIZE}_BS{BATCH_SIZE}'
OUTPUT_PATH = '../checkpoints/{}/{}_{}_{}/'.format(COST, DATASET1, DATASET2, LATENTS_SIZE)

# END OF CONFIG
################


if __name__ == "__main__":
    config = dict(
        DATASET1=DATASET1,
        DATASET2=DATASET2,
        T_ITERS=T_ITERS,
        f_LR=f_LR, T_LR=T_LR,
        BATCH_SIZE=BATCH_SIZE
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

    f = ResNet_D(size=LATENTS_SIZE, nc=4, nfilter_max=256).cuda()  # lower max filter size just because %)
    f.apply(weights_init_D)

    T = UNet(4+ZC, 4, base_factor=48).cuda() # ZC - noise input channels z

    vae_decoder = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae_decoder = vae_decoder.to(f'cuda:{DEVICE_IDS[0]}')
    freeze(vae_decoder)

    if len(DEVICE_IDS) > 1:
        T = nn.DataParallel(T, device_ids=DEVICE_IDS)
        f = nn.DataParallel(f, device_ids=DEVICE_IDS)
        vae_decoder = nn.DataParallel(vae_decoder, device_ids=DEVICE_IDS)

    print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
    print('f params:', np.sum([np.prod(p.shape) for p in f.parameters()]))

    torch.manual_seed(0xBADBEEF); np.random.seed(0xBADBEEF)
    X_fixed = X_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z_fixed = torch.randn(10, 4, ZC, LATENTS_SIZE, LATENTS_SIZE, device='cuda') * Z_STD
        XZ_fixed = torch.cat([X_fixed, Z_fixed], dim=2)
    del X_fixed, Z_fixed
    Y_fixed = Y_sampler.sample(10)

    X_test_fixed = X_test_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z_test_fixed = torch.randn(10, 4, ZC, LATENTS_SIZE, LATENTS_SIZE, device='cuda') * Z_STD
        XZ_test_fixed = torch.cat([X_test_fixed, Z_test_fixed], dim=2)
    del X_test_fixed, Z_test_fixed
    Y_test_fixed = Y_test_sampler.sample(10)

    #######
    # Training
    wandb.init(name=EXP_NAME, project='notreallyweakot', entity='metra4ok', config=config)

    T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)

    print(f"===> Learning rates:")
    print(f"\tT_LR = {T_LR}\n\tf_LR = {f_LR}\n\n")

    for step in tqdm(range(MAX_STEPS)):
        gamma = min(GAMMA1, GAMMA0 + (GAMMA1-GAMMA0) * step / GAMMA_ITERS)
        # T optimization
        unfreeze(T); freeze(f)
        potentials, costs, T_losses = [], [], []  # to compute means for logging
        for t_iter in range(T_ITERS):
            T_opt.zero_grad()
            X = X_sampler.sample(BATCH_SIZE)[:,None].repeat(1,Z_SIZE,1,1,1)
            with torch.no_grad():
                Z = torch.randn(BATCH_SIZE, Z_SIZE, ZC, LATENTS_SIZE, LATENTS_SIZE, device='cuda') * Z_STD
                XZ = torch.cat([X, Z], dim=2)
            T_XZ = T(
                XZ.flatten(start_dim=0, end_dim=1)
            ).permute(1,2,3,0).reshape(4, LATENTS_SIZE, LATENTS_SIZE, -1, Z_SIZE).permute(3,4,0,1,2)
            # T_XZ shape: [512, 4, 16, 16] -> [4, 16, 16, 512] -> [4, 16, 16, 64, 8] -> [64, 8, 4, 16, 16]
            weak_cost = F.mse_loss(X[:,0], T_XZ.mean(dim=1)).mean() \
                        + T_XZ.var(dim=1).mean() * (1 - gamma - 1. / Z_SIZE)
            costs.append(weak_cost.item())
            # T_XZ.flatten(start_dim=0, end_dim=1).shape) == [512, 4, 16, 16]
            potential_T_XZ = f(T_XZ.flatten(start_dim=0, end_dim=1)).mean()
            potentials.append(potential_T_XZ.item())
            T_loss = weak_cost - potential_T_XZ
            T_losses.append(T_loss.item())
            T_loss.backward()
            T_opt.step()
        wandb.log(
            {
                "mean_f(T_XZ)": np.mean(potentials),
                "mean_Weak_Cost(X, T_XZ)": np.mean(costs),
                "mean_T_loss": np.mean(T_losses)
            },
            step=step,
        )
        del potentials, costs, T_losses, potential_T_XZ, weak_cost, T_loss, T_XZ, X, Z
        gc.collect()
        torch.cuda.empty_cache()

        # f optimization
        freeze(T); unfreeze(f)
        X = X_sampler.sample(BATCH_SIZE)
        with torch.no_grad():
            Z = torch.randn(BATCH_SIZE, ZC, X.size(2), X.size(3), device='cuda') * Z_STD
            XZ = torch.cat([X,Z], dim=1)
            T_XZ = T(XZ)
        Y = Y_sampler.sample(BATCH_SIZE)
        f_opt.zero_grad()
        potential_T_XZ = f(T_XZ).mean()
        potential_Y = f(Y).mean()
        f_loss = potential_T_XZ - potential_Y
        f_loss.backward()
        f_opt.step()
        wandb.log(
            {
                "f(T_XZ)": potential_T_XZ.item(),
                "f(Y)": potential_Y.item(),
                "f_loss": f_loss.item(),
            },
            step=step,
        )
        del potential_T_XZ, potential_Y, f_loss, Y, X, T_XZ, Z, XZ
        gc.collect()
        torch.cuda.empty_cache()

        if step % PLOT_INTERVAL == 0:
            print('Plotting')

            fig, axes = plot_latent_Z_images(XZ_fixed, Y_fixed, T, decoder=vae_decoder)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close(fig)

            fig, axes = plot_random_latent_Z_images(X_sampler, ZC, Z_STD,  Y_sampler, T)
            wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close(fig)

            fig, axes = plot_latent_Z_images(XZ_test_fixed, Y_test_fixed, T)
            wandb.log({'Fixed Test Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close(fig)

            fig, axes = plot_random_latent_Z_images(X_test_sampler, ZC, Z_STD,  Y_test_sampler, T)
            wandb.log({'Random Test Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close(fig)

        if step % CPKT_INTERVAL == CPKT_INTERVAL - 1:
            freeze(T)

            print('Computing FID')
            mu, sigma = get_latent_Z_pushed_loader_stats(T, vae_decoder, X_test_sampler.loader, ZC=ZC, Z_STD=Z_STD)
            fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
            wandb.log({f'FID (Test)' : fid}, step=step)
            del mu, sigma

            state_dict = {
                "T_state_dict": T.state_dict(),
                "f_state_dict": f.state_dict(),
                "T_opt_state_dict": T_opt.state_dict(),
                "f_opt_state_dict": f_opt.state_dict(),
            }

            torch.save(state_dict, os.path.join(OUTPUT_PATH, f'{SEED}_{step}.pt'))

        gc.collect()
        torch.cuda.empty_cache()
