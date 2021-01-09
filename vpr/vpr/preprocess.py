import json

import numpy as np
import os
import pandas as pd
import torch
import bz2
import pickle
import _pickle as cPickle
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from vpr.vpr.vpr_hyper_param_handler import get_params


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


def load(df):
    data = dict()

    for user, group in df.groupby('user'):
        data[user] = list(group['movie'])

    return data


class _PosNegsData(Dataset):
    def __init__(self, data):
        super(_PosNegsData, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        pos = self.data[idx][1]
        negs = self.data[idx][2]

        return user, pos, negs


class _TripletData(Dataset):
    def __init__(self, data):
        super(_TripletData, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        pos = self.data[idx][1]
        neg = self.data[idx][2]

        return user, pos, neg


class VPRDataLoader:
    def __init__(self, hyper_params, device):
        self.hp = hyper_params
        self.device = device

    def _pack_collate_fn(self, data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths), device=self.device).long()

            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = torch.tensor(seq[:end], device=self.device)

            return padded_seqs, lengths

        data.sort(key=lambda x: len(x[1]), reverse=True)
        user, pos, neg = zip(*data)
        padded_pos, pos_length = merge(pos)
        padded_neg, neg_length = merge(neg)

        return torch.tensor(user, device=self.device, dtype=torch.float), \
               padded_pos, \
               torch.tensor(pos_length, device=self.device, dtype=torch.long), \
               padded_neg, \
               torch.tensor(neg_length, device=self.device, dtype=torch.long)

    def load_data(self, cache_filename):
        """
        if cache_filename exists, data is loaded from cache
        else from input file
        if data is read from file and cache_filename is not None, loaded data is stored to file

        :param cache_dir: None or
        """
        if cache_filename and os.path.exists(cache_filename):
            for fname in glob.glob(os.path.join(cache_filename, '*')):
                os.remove(fname)

        self._load_from_input_file()
        # data, data_te, data_vd, num_items, num_users = self._load_from_input_file()
        #
        # os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        #
        # with open(cache_filename, 'wb') as fp:
        #     pickle.dump((data, data_te, data_vd, num_items, num_users),
        #                 fp,
        #                 pickle.HIGHEST_PROTOCOL)

    def _load_from_input_file(self):
        print('load csv files')
        cols = ['user', 'movie', 'rating']
        train = pd.read_csv(os.path.join(self.hp['input_dir'], 'train.csv'), header=0, names=cols)
        vad_tr = pd.read_csv(os.path.join(self.hp['input_dir'], 'validation_tr.csv'), header=0, names=cols)
        test_tr = pd.read_csv(os.path.join(self.hp['input_dir'], 'test_tr.csv'), header=0, names=cols)
        vad_te = pd.read_csv(os.path.join(self.hp['input_dir'], 'validation_te.csv'), header=0, names=cols)
        test_te = pd.read_csv(os.path.join(self.hp['input_dir'], 'test_te.csv'), header=0, names=cols)
        if not self.hp['dataset_with_one_id']:
            for column in ('user', 'movie'):
                train[column] = train[column] + 1
                vad_tr[column] = vad_tr[column] + 1
                test_tr[column] = test_tr[column] + 1
                vad_te[column] = vad_te[column] + 1
                test_te[column] = test_te[column] + 1
        _unique_sid = list()
        with open(os.path.join(self.hp['input_dir'], 'unique_sid.txt'), 'r') as f:
            for line in f:
                _unique_sid.append(line.strip())
        num_items = len(_unique_sid)

        del _unique_sid

        train = load(train)
        vad_tr = load(vad_tr)
        vad_te = load(vad_te)
        test_tr = load(test_tr)
        test_te = load(test_te)
        user_ids = set(train.keys())
        for k in (vad_tr, vad_te, test_tr, test_te):
            user_ids.update(k.keys())
        num_users = len(user_ids)

        del user_ids

        all_items = set(range(1, num_items+1))
        datafile = '/home/pisani/svae_2020_new/vpr_history_dataset_netflix_sample/dataFileTR.tmp'
        data_vd_file = '/home/pisani/svae_2020_new/vpr_history_dataset_netflix_sample/dataFileVD.tmp'
        data_te_file = '/home/pisani/svae_2020_new/vpr_history_dataset_netflix_sample/dataFileTE.tmp'

        print('Build training set')
        data = {}
        cnt = 0
        for u_id in tqdm(list(train.keys())):
            u_items = train[u_id]

            negatives = list(all_items.difference(u_items))
            data[f'U{u_id}'] = np.array(u_items, dtype=np.uint16)

            chunks = [u_items[x:x + 100] for x in range(0, len(u_items), 100)]
            if len(chunks[-1]) != 100:
                while len(chunks[-1]) < 100:
                    chunks[-1].append(0)

            for pos_list in chunks:
                neg_list = [np.random.choice(negatives, self.hp['tr_n_negatives']) for _ in pos_list]

                pos_list = np.array(pos_list, dtype=np.uint16).repeat(len(neg_list[0]))
                neg_list = np.array(neg_list, dtype=np.uint16).flatten()

                data[str(cnt)] = (u_id, pos_list, neg_list)
                cnt += 1

            del negatives
            del train[u_id]

        data['COUNT'] = cnt

        compressed_pickle(datafile, data)
        del data

        print('Build validation set')
        data = {}
        cnt = 0
        for u_id in tqdm(list(vad_te.keys())):
            u_items = vad_te[u_id]
            data[f'U{u_id}'] = np.array(vad_tr[u_id], dtype=np.uint16)

            negatives = list(all_items.difference(u_items + vad_tr[u_id]))

            chunks = [u_items[x:x + 100] for x in range(0, len(u_items), 100)]
            if len(chunks[-1]) != 100:
                while len(chunks[-1]) < 100:
                    chunks[-1].append(0)

            for pos_list in chunks:
                neg_list = [np.random.choice(negatives, self.hp['tr_n_negatives']) for _ in pos_list]

                pos_list = np.array(pos_list, dtype=np.uint16).repeat(len(neg_list[0]))
                neg_list = np.array(neg_list, dtype=np.uint16).flatten()

                data[str(cnt)] = (u_id, pos_list, neg_list)
                cnt += 1

            del negatives
            del vad_te[u_id]

        data['COUNT'] = cnt
        compressed_pickle(data_vd_file, data)
        del data

        print('Build test set')
        data = {}
        for i, (u_id, u_items) in tqdm(enumerate(test_te.items())):
            negatives = list(all_items.difference(u_items + test_tr[u_id]))

            data[str(i)] = (test_tr[u_id], u_items, negatives)

        data['COUNT'] = len(test_te)
        compressed_pickle(data_te_file, data)

        # reset
        del data
        del train
        del vad_tr
        del vad_te
        del test_tr
        del test_te
        del all_items

        for fname in (datafile, data_vd_file, data_te_file):
            for k in glob.glob(fname+'*'):
                print(f'{k} is {os.path.getsize(k)} bytes')

        with open(os.path.join(os.path.dirname(datafile), 'users.json'), 'w') as fp:
            json.dump([num_users, num_items], fp)

        print('datasets created.')


def check_space(fname):
    moregiga = 200 * (1024**3)
    space = 0
    for k in glob.glob('/home/pisani/svae_2020_new/vpr_history_dataset_netflix_sample/*'):
        space += os.path.getsize(k)
    if space > moregiga:
        return True


def create_dataset(case_study):
    hp = get_params(case_study, '0')
    # tmp_file = os.path.join(hp['tmp_dir'], 'tmp_data.pickle')
    tmp_file = '/home/pisani/svae_2020_new/vpr_history_dataset_netflix_sample/'

    data_loader = VPRDataLoader(hp, 'cpu')
    data_loader.load_data(tmp_file)


if __name__ == '__main__':
    # case_study = 'ml-1m'
    # case_study = 'ml-20m'
    case_study = 'netflix_sample'
    # case_study = 'netflix'
    create_dataset(case_study)
