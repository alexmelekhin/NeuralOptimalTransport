import os, sys
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

from src.resnet2 import ResNet_D
from src.unet import UNet

from src.tools import unfreeze, freeze
from src.tools import weights_init_D
from src.tools import load_dataset, get_pushed_loader_stats
from src.fid_score import calculate_frechet_distance
from src.plotters import plot_random_images, plot_images

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
NUM_WORKERS = 8

# DATASET1, DATASET1_PATH = 'handbag', '../../data/handbag_128.hdf5'
# DATASET2, DATASET2_PATH = 'shoes', '../../data/shoes_128.hdf5'

DATASET1, DATASET1_PATH = 'celeba_female', '/mnt/ssd1/Datasets/CelebA'
DATASET2, DATASET2_PATH = 'aligned_anime_faces', '/mnt/ssd1/Datasets/AlignedAnimeFaces_128'

batch_scale = 1  # to try linear scaling rule

T_ITERS = 10
f_LR, T_LR = 1e-4 * batch_scale, 1e-4 * batch_scale  # baseline is 1e-4
IMG_SIZE = 128

BATCH_SIZE = int(64 * batch_scale)  # baseline is 64

PLOT_INTERVAL = 100
COST = 'mse' # Mean Squared Error
CPKT_INTERVAL = 2000
MAX_STEPS = 100001
SEED = 0x000000

EXP_NAME = f'{DATASET1}_{DATASET2}_T{T_ITERS}_{COST}_{IMG_SIZE}_BS{BATCH_SIZE}'
OUTPUT_PATH = '../checkpoints/{}/{}_{}_{}/'.format(COST, DATASET1, DATASET2, IMG_SIZE)

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

    filename = '../stats/{}_{}_test.json'.format(DATASET2, IMG_SIZE)
    with open(filename, 'r') as fp:
        data_stats = json.load(fp)
        mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
    del data_stats

    X_sampler, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=IMG_SIZE, num_workers=NUM_WORKERS)
    Y_sampler, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE, num_workers=NUM_WORKERS)

    torch.cuda.empty_cache()
    gc.collect()

    f = ResNet_D(IMG_SIZE, nc=3).cuda()
    f.apply(weights_init_D)

    T = UNet(3, 3, base_factor=48).cuda()

    if len(DEVICE_IDS) > 1:
        T = nn.DataParallel(T, device_ids=DEVICE_IDS)
        f = nn.DataParallel(f, device_ids=DEVICE_IDS)

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

            fig, axes = plot_images(X_fixed, Y_fixed, T)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close()

            fig, axes = plot_random_images(X_sampler,  Y_sampler, T)
            wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close()

            fig, axes = plot_images(X_test_fixed, Y_test_fixed, T)
            wandb.log({'Fixed Test Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close()

            fig, axes = plot_random_images(X_test_sampler, Y_test_sampler, T)
            wandb.log({'Random Test Images' : [wandb.Image(fig2img(fig))]}, step=step)
            plt.close()

        if step % CPKT_INTERVAL == CPKT_INTERVAL - 1:
            freeze(T)

            print('Computing FID')
            mu, sigma = get_pushed_loader_stats(T, X_test_sampler.loader)
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
