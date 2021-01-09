import datetime as dt
import json
import os
import numpy as np
import pickle
import torch

from vpr.common.learning import Learner
from vpr.common.logger import log
from vpr.common.util import build_device, reproducibility
from vpr.vpr.data_loaders import VPRDataLoader
from vpr.vpr.loss_functions import VPRLoss
from vpr.vpr.models import VPR
from vpr.vpr.vpr_hyper_param_handler import get_params


def load_readers(hp, device, debug=False):
    tmp_file = os.path.join(hp['tmp_dir'], 'tmp_data.pickle')

    if debug:
        tmp_file = None

    data_loader = VPRDataLoader(hp, device)
    train_reader, val_reader, test_reader, total_items, total_users = data_loader.load_data(tmp_file)

    return train_reader, val_reader, test_reader, total_items, total_users


def log_info(hp, train_reader, test_reader, total_users, total_items):
    log("\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n", hp['log_file'])
    log("Data reading complete!", hp['log_file'])
    log("Number of train batches: {:4d}".format(len(train_reader)), hp['log_file'])
    log("Number of test batches: {:4d}".format(len(test_reader)), hp['log_file'])
    log("Total Users: " + str(total_users), hp['log_file'])
    log("Total Items: " + str(total_items) + "\n", hp['log_file'])


if __name__ == '__main__':
    case_study = 'ml-1m'
    # case_study = 'ml-20m'
    # case_study = 'netflix_sample'
    # case_study = 'netflix'
    run_id = str(1)

    hp = get_params(case_study, run_id)

    with open(hp['param_file'], "w") as outfile:
        json.dump(hp, outfile)

    reproducibility()
    device = build_device()
    print('USE DEVICE:', device)
    train_reader, val_reader, test_reader, total_items, total_users = load_readers(hp, device)
    log_info(hp, train_reader, test_reader, total_users, total_items)

    model = VPR(total_items, hp).to(device)
    loss_function = VPRLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=.01)

    learner = Learner(train_reader, val_reader, test_reader, model, device, optimizer, loss_function, hp)

    learner.train()
