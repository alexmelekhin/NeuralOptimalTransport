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

from src.tools import unfreeze, freeze
from src.tools import load_dataset, get_loader_stats

from copy import deepcopy
import json

from tqdm import tqdm

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

DEVICE_ID = 0

DATASET_LIST = [
    # ('handbag', '../../data/handbag_128.hdf5', 64),
    # ('handbag', '../../data/handbag_128.hdf5', 128),
    # ('shoes', '../../data/shoes_128.hdf5', 64),
    # ('shoes', '../../data/shoes_128.hdf5', 128),
    # ('celeba_female', '/mnt/ssd1/Datasets/CelebA', 64),
    # ('celeba_female', '/mnt/ssd1/Datasets/CelebA', 128),
    ('celeba_female', '/gpfs/gpfs0/groznyy.sergey/Datasets/CelebA', 512),
    # ('aligned_anime_faces', '/mnt/ssd1/Datasets/AlignedAnimeFaces', 64),
    # ('aligned_anime_faces', '/mnt/ssd1/Datasets/AlignedAnimeFaces', 128),
    ('aligned_anime_faces', '/gpfs/gpfs0/groznyy.sergey/Datasets/AlignedAnimeFaces_256', 512),
]

assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:{DEVICE_ID}')

for DATASET, DATASET_PATH, IMG_SIZE in tqdm(DATASET_LIST):
    print('Processing {}'.format(DATASET))
    sampler, test_sampler = load_dataset(DATASET, DATASET_PATH, img_size=IMG_SIZE)
    print('Dataset {} loaded'.format(DATASET))

    mu, sigma = get_loader_stats(test_sampler.loader)
    print('Trace of sigma: {}'.format(np.trace(sigma)))
    stats = {'mu' : mu.tolist(), 'sigma' : sigma.tolist()}
    print('Stats computed')

    filename = '{}_{}_test.json'.format(DATASET, IMG_SIZE)
    with open(filename, 'w') as fp:
        json.dump(stats, fp)
    print('States saved to {}'.format(filename))
