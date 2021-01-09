import numpy as np
import os
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, data, num_items):
        super(_TripletData, self).__init__()
        self.data = data
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        pos = self.data[idx][1]
        neg = self.data[idx][2]

        user = np.array(user, dtype=np.int) - 1
        user_array = torch.zeros(self.num_items, dtype=torch.float32)
        user_array[user] = 1

        pos = pos.astype(np.long)
        neg = neg.astype(np.long)

        return user_array, torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long)


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

    def load_data(self, cache_filename=None):
        """
        if cache_filename exists, data is loaded from cache
        else from input file
        if data is read from file and cache_filename is not None, loaded data is stored to file

        :param cache_dir: None or
        :return: training_dl, validation_dl, test_dl, num_items, num_users
        """
        if cache_filename and os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                (data, data_te, data_vd, num_items, num_users) = pickle.load(f)
        else:
            data, data_te, data_vd, num_items, num_users = self._load_from_input_file()

            if cache_filename:
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

                with open(cache_filename, 'wb') as fp:
                    pickle.dump((data, data_te, data_vd, num_items, num_users),
                                fp,
                                pickle.HIGHEST_PROTOCOL)

        training_dl = DataLoader(_TripletData(data, num_items), batch_size=self.hp['batch_size'], shuffle=True)
        validation_dl = DataLoader(_TripletData(data_vd, num_items), batch_size=self.hp['batch_size'], shuffle=True)
        test_dl = DataLoader(_PosNegsData(data_te), batch_size=self.hp['batch_size'], shuffle=True,
                             collate_fn=self._pack_collate_fn)

        return training_dl, validation_dl, test_dl, num_items, num_users

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
        data = []
        data_te = []
        data_vd = []

        row_size = self.hp['row_size']

        print('Build training set')
        for u_id, u_items in train.items():

            user = np.zeros(num_items, dtype=np.float32)
            user[np.array(u_items)] = 1

            negatives = list(all_items.difference(u_items))
            
            chunks = [u_items[x:x + row_size] for x in range(0, len(u_items), row_size)]
            if len(chunks[-1]) != row_size:
                while len(chunks[-1]) < row_size:
                    chunks[-1].append(0)
                    
                    
            for pos_list in chunks:
                neg_list = [np.random.choice(negatives, self.hp['tr_n_negatives']) for _ in pos_list]

                pos_list = np.array(pos_list, dtype=np.uint16).repeat(len(neg_list[0]))
                neg_list = np.array(neg_list, dtype=np.uint16).flatten()

                data.append((user, pos_list, neg_list))

            del negatives

        print('Build validation set')
        for u_id, u_items in vad_te.items():
            user = np.zeros(num_items, dtype=np.float32)
            user[np.array(vad_tr[u_id])] = 1

            negatives = list(all_items.difference(u_items + vad_tr[u_id]))
            
            chunks = [u_items[x:x + row_size] for x in range(0, len(u_items), row_size)]
            if len(chunks[-1]) != row_size:
                while len(chunks[-1]) < row_size:
                    chunks[-1].append(0)

            for pos_list in chunks:
                neg_list = [np.random.choice(negatives, self.hp['tr_n_negatives']) for _ in pos_list]

                pos_list = np.array(pos_list, dtype=np.uint16).repeat(len(neg_list[0]))
                neg_list = np.array(neg_list, dtype=np.uint16).flatten()

                data_vd.append((user, pos_list, neg_list))

            del negatives

        print('Build test set')
        for u_id, u_items in test_te.items():
            user = np.zeros(num_items, dtype=np.float32)
            user[np.array(test_tr[u_id])] = 1

            negatives = list(all_items.difference(u_items + test_tr[u_id]))

            data_te.append((user, u_items, negatives))

        assert len(data) > 0 and len(data_vd) > 0 and len(data_te) > 0
        print('datasets created.')
        return data, data_te, data_vd, num_items, num_users
