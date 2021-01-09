
import os
import torch
import numpy as np

def build_device(use_cuda=True):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    elif use_cuda:
        use_cuda = len(os.environ['CUDA_VISIBLE_DEVICES']) > 0
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def reproducibility(seed=12345):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load(df, coluser='user', colmovie='movie'):
    data = dict()

    for user, group in df.groupby(coluser):
        data[user] = list(group[colmovie])

    return data

