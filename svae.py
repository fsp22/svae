#!/usr/bin/env python
# coding: utf-8

import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
import matplotlib.pyplot as plt
import warnings
from datetime import datetime


warnings.filterwarnings('ignore')

# # Hyper Parameters
hyper_params = {
    'data_base': 'datasets/ml-1m/',  # Don't remove the '/' at the end please :)
    'project_name': 'svae_ml1m',
    # 'data_base': 'datasets/ml-20m/',
    # 'project_name': 'svae_ml-20m',
    # 'data_base': 'datasets/netflix_sample/',
    # 'project_name': 'svae_netflix_sample',
    # 'data_base': 'datasets/netflix/',
    # 'project_name': 'svae_netflix',
    'n_heldout_users': 750,  # dataset dependent ML-1M
    # 'n_heldout_users': 57500,  # netflix full
    # 'n_heldout_users': 9500,  # netflix sample
    # 'n_heldout_users': 17000,  # ml20m
    'model_file_name': '',
    'log_file': '',
    'history_split_test': [0.8, 0.2], # Part of test history to train on : Part of test history to test

    # model: PE, SLD, MLD, RLD
    'model_type': 'SLD',

    'learning_rate': 0.001,  # learning rate
    'optimizer': 'adam',
    'weight_decay': float(5e-3),

    'epochs': 25,
    'batch_size': 32,  # Needs to be 1, because we don't pack multiple sequences in the same batch

    # movielens
    'item_embed_size': 64,
    'rnn_size': 512,
    'hidden_size': 512,
    'latent_size': 64,
    'loss_type': 'predict_next',  # [predict_next, same, prefix, postfix, exp_decay, next_k]
    'next_k': 4,

    # netflix
    # 'item_embed_size': 256,
    # 'rnn_size': 256,
    # 'hidden_size': 256,
    # 'latent_size': 64,
    # 'loss_type': 'next_k', # [predict_next, same, prefix, postfix, exp_decay, next_k]
    # 'next_k': 4,

    # MLD
    'latent_size_rnn': 32,

    # KLD
    'kld_type': 'kld_x1t',  # old, kld_x1t
    'kld_avg_on_batch': False,
    'anneal_on_epoch': True,
    'anneal_cap': 0.2,
    'total_anneal_steps': 20000,

    'number_users_to_keep': 1000000000,
    'batch_log_interval': 16,
    'train_cp_users': 200,
    'exploding_clip': 0.25,
    'save_raw_prediction': True,
    'seed': 98765,
}

### change `DATA_DIR` to the location where the dataset sits
### compatible datasets: ML-1M, Netflix-full


file_name = '_optimizer_' + str(hyper_params['optimizer'])
if hyper_params['optimizer'] == 'adagrad':
    file_name += '_lr_' + str(hyper_params['learning_rate'])
file_name += '_batch_size_' + str(hyper_params['batch_size'])
file_name += '_weight_decay_' + str(hyper_params['weight_decay'])
file_name += '_loss_type_' + str(hyper_params['loss_type'])
file_name += '_item_embed_size_' + str(hyper_params['item_embed_size'])
file_name += '_rnn_size_' + str(hyper_params['rnn_size'])
file_name += '_latent_size_' + str(hyper_params['latent_size'])
file_name += '_kld_avg_on_batch_' + str(hyper_params['kld_avg_on_batch'])
file_name += '_anneal_on_epoch_' + str(hyper_params['anneal_on_epoch'])

rundate = datetime.today().strftime('%Y%m%d_%H%M')
log_file_root = f"saved_logs/run{rundate}/" # Don't remove the '/' at the end please :)
model_file_root = "saved_models/" # Don't remove the '/' at the end please :)

if not os.path.isdir(log_file_root): os.makedirs(log_file_root)
if not os.path.isdir(model_file_root): os.mkdir(model_file_root)
hyper_params['log_file'] = log_file_root + hyper_params['project_name'] + '_log' + file_name + '.txt'
hyper_params['model_file_name'] = model_file_root + hyper_params['project_name'] + '_model' + file_name + '.pt'


# # Data Preprocessing
# 
# **Courtesy:** Dawen Liang et al. "*Variational autoencoders for collaborative filtering*" published at WWW '18. <br>
# **Link:** https://github.com/dawenl/vae_cf


DATA_DIR = hyper_params['data_base']
pro_dir = os.path.join(DATA_DIR, 'pro_sg') # Path where preprocessed data will be saved
hyper_params['data_base'] += 'pro_sg/'

if not os.path.isdir(pro_dir): # We don't want to keep preprocessing every time we run the notebook
    cols = ['userId', 'movieId', 'rating', 'timestamp']
    dtypes = {'userId': 'int', 'movieId': 'int', 'timestamp': 'int', 'rating': 'int'}
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv.bz2'), sep=',', names=cols, parse_dates=['timestamp'])

    max_seq_len = 1000
    n_heldout_users = hyper_params['n_heldout_users']  # If total users = N; train_users = N - 2*heldout; test_users & val_users = heldout

    # binarize the data (only keep ratings >= 4)
    raw_data = raw_data[raw_data['rating'] > 3.5]

    # Remove users with greater than $max_seq_len number of watched movies
    raw_data = raw_data.groupby(["userId"]).filter(lambda x: len(x) <= max_seq_len)

    # Sort data values with the timestamp
    raw_data = raw_data.groupby(["userId"]).apply(lambda x: x.sort_values(["timestamp"], ascending = True)).reset_index(drop=True)

    raw_data.head()


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users. 
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount[itemcount.iloc[:, 1] >= min_sc]['movieId'])]
    
    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount[usercount.iloc[:, 1] >= min_uc]['userId'])]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId') 
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(hyper_params['seed'])

    skipped = 0
    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        # if n_items_u < 5:
        #     raise Exception("MIN items failed")
        start_idx_test = int((1.0 - test_prop) * n_items_u)
        if start_idx_test < 2:
            start_idx_test = 2

        if start_idx_test >= n_items_u:
            start_idx_test = n_items_u - 1

        # in train/test at least 2 items must be present
        if n_items_u >= 5 and start_idx_test >= 2 and n_items_u - start_idx_test >= 1:
            idx = np.zeros(n_items_u, dtype='bool')
            # idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            idx[start_idx_test:] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
            assert len(tr_list[-1]) >= 2, f'train set len for user is {len(tr_list[-1])}'
            assert len(te_list[-1]) >= 1, f'test set len for user is {len(te_list[-1])}'
        else:
            skipped += 1
            # tr_list.append(group)
            # assert len(tr_list[-1]) >= 2, f'train set len for user is {len(tr_list[-1])}'

        if i > 0 and i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    if skipped > 0:
        print('skippped users %s' % skipped)

    return data_tr, data_te


def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    ra = list(map(lambda x: x, tp['rating']))
    ret = pd.DataFrame(data={'uid': uid, 'sid': sid, 'rating': ra}, columns=['uid', 'sid', 'rating'])
    ret['rating'] = ret['rating'].apply(pd.to_numeric)
    return ret


if not os.path.isdir(pro_dir): # We don't want to keep preprocessing every time we run the notebook

    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    unique_uid = user_activity.index

    np.random.seed(hyper_params['seed'])
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    # create train/validation/test users
    n_users = unique_uid.size

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i + 1) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i + 1) for (i, pid) in enumerate(unique_uid))

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays, hyper_params['history_split_test'][1])

    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays, hyper_params['history_split_test'][1])

    train_data = numerize(train_plays)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)


# # Utility functions

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

is_cuda_available = torch.cuda.is_available()

if is_cuda_available: 
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_obj_json(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_obj_json(name):
    with open(name + '.json', 'r') as f:
        return json.load(f)


def file_write(log_file, s):
    print(s)
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()


def clear_log_file(log_file):
    f = open(log_file, 'w')
    f.write('')
    f.close()


def pretty_print(h):
    print("{")
    for key in h:
        print(' ' * 4 + str(key) + ': ' + h[key])
    print('}\n')


def plot_len_vs_ndcg(len_to_ndcg_at_100_map):
    
    lens = list(len_to_ndcg_at_100_map.keys())
    lens.sort()
    X, Y = [], []
    
    for le in lens:
        X.append(le)
        ans = 0.0
        for i in len_to_ndcg_at_100_map[le]: ans += float(i)
        ans = ans / float(len(len_to_ndcg_at_100_map[le]))
        Y.append(ans * 100.0)
    
    # Smoothening
    Y_mine = []
    prev_5 = []
    for i in Y:
        prev_5.append(i)
        if len(prev_5) > 5: del prev_5[0]

        temp = 0.0
        for j in prev_5: temp += float(j)
        temp = float(temp) / float(len(prev_5))
        Y_mine.append(temp)
    
    plt.figure(figsize=(12, 5))
    plt.plot(X, Y_mine, label='SVAE')
    plt.xlabel("Number of items in the fold-out set")
    plt.ylabel("Average NDCG@100")
    plt.title(hyper_params['project_name'])
    if not os.path.isdir("saved_plots/"): os.mkdir("saved_plots/")
    plt.savefig("saved_plots/seq_len_vs_ndcg_" + hyper_params['project_name'] + ".pdf")

    plt.legend(loc='best', ncol=2)
    plt.show()


# # Data Parsing
def load_data(hyper_params):
    
    file_write(hyper_params['log_file'], "Started reading data file")
    
    f = open(hyper_params['data_base'] + 'train.csv')
    lines_train = f.readlines()[1:]
    
    f = open(hyper_params['data_base'] + 'validation_tr.csv')
    lines_val_tr = f.readlines()[1:]
    
    f = open(hyper_params['data_base'] + 'validation_te.csv')
    lines_val_te = f.readlines()[1:]
    
    f = open(hyper_params['data_base'] + 'test_tr.csv')
    lines_test_tr = f.readlines()[1:]
    
    f = open(hyper_params['data_base'] + 'test_te.csv')
    lines_test_te = f.readlines()[1:]
    
    unique_sid = list()
    with open(hyper_params['data_base'] + 'unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    num_items = len(unique_sid)
    
    file_write(hyper_params['log_file'], "Data Files loaded!")

    train_reader = DataReader(hyper_params, lines_train, None, num_items, True)
    val_reader = DataReader(hyper_params, lines_val_tr, lines_val_te, num_items, False)
    test_reader = DataReader(hyper_params, lines_test_tr, lines_test_te, num_items, False)

    return train_reader, val_reader, test_reader, num_items


class DataReader:
    def __init__(self, hyper_params, a, b, num_items, is_training):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']

        self.num_users = 0
        self.num_items = num_items
        
        self.data_train = a
        self.data_test = b
        self.is_training = is_training
        self.all_users = []
        
        self.prep()
        self.number()

    def prep(self):
        dict_user_id = collections.defaultdict(lambda: len(dict_user_id))
        self.data = []

        for i in tqdm(range(len(self.data_train))):
            line = self.data_train[i]
            line = line.strip().split(",")
            user_id = dict_user_id[int(line[0])]
            if user_id == len(self.data):
                self.data.append([])
            self.data[user_id].append([ int(line[1]), 1 ])

        self.num_users = len(self.data)

        if self.is_training is False:
            self.data_te = [[] for _ in range(len(self.data))]

            for i in tqdm(range(len(self.data_test))):
                line = self.data_test[i]
                line = line.strip().split(",")
                user_id = dict_user_id[int(line[0])]
                self.data_te[user_id].append([ int(line[1]), 1 ])

        for i in range(self.num_users):
            assert len(self.data[i]) >= 2, f'user {i} with {len(self.data[i])} items in train'
            if not self.is_training:
                assert len(self.data_te[i]) > 0, f'user {i} without items in test'

    def number(self):
        self.num_b = int(min(len(self.data), self.hyper_params['number_users_to_keep']) / self.batch_size)

    def pad(self, to_pack, mid_dimension, for_scatter=False):
        for b in range(len(to_pack)):
            for num_to_add in range(mid_dimension - len(to_pack[b])):
                to_pack[b].append(0)

            if for_scatter is True:
                for seq_num in range(mid_dimension):
                    if self.hyper_params['loss_type'] == 'predict_next':
                        to_pack[b][seq_num] = [ to_pack[b][seq_num] ]
                    
                    elif self.hyper_params['loss_type'] == 'next_k':
                        to_pack[b][seq_num] = to_pack[b][seq_num:][:self.hyper_params['next_k']]
                        have = len(to_pack[b][seq_num])
                        last_elem = to_pack[b][seq_num][-1]
                        for pack in range(self.hyper_params['next_k'] - have):
                            to_pack[b][seq_num].append(last_elem) # Wouldn't do nothing

                    elif self.hyper_params['loss_type'] == 'postfix':
                        to_pack[b][seq_num] = to_pack[b][seq_num:]
                        have = len(to_pack[b][seq_num])
                        last_elem = to_pack[b][seq_num][-1]
                        for pack in range(mid_dimension - have):
                            to_pack[b][seq_num].append(last_elem)

        return to_pack
        
    def iter(self):
        users_done = 0

        x_batch, y_batch, seq_lens = [], [], []
        
        user_iterate_order = list(range(len(self.data)))
        
        # Randomly shuffle the training order
        np.random.shuffle(user_iterate_order)
        
        for user in user_iterate_order:

            if users_done > self.hyper_params['number_users_to_keep']: break
            users_done += 1
            
            curlen = len(self.data[user]) - 1
            x_batch.append([ i[0] for i in self.data[user][:-1] ])
            y_batch.append([ i[0] - 1 for i in self.data[user][1:] ])
            seq_lens.append(curlen)
            assert curlen > 0, f'user with empty seq_len: {user}'

            if len(x_batch) == self.batch_size:
                mid_dimension = max(seq_lens)
                
                x_batch = self.pad(x_batch, mid_dimension)
                y_batch = self.pad(y_batch, mid_dimension, for_scatter = True)

                indexes = sorted(range(len(x_batch)), key=lambda x: - seq_lens[x])
                x_batch = [x_batch[i] for i in indexes]
                y_batch = [y_batch[i] for i in indexes]
                seq_lens = [seq_lens[i] for i in indexes]

                y_batch_s = torch.zeros(self.batch_size, mid_dimension, self.num_items)
                if is_cuda_available:
                    y_batch_s = y_batch_s.cuda()

                y_batch_s.scatter_(-1, LongTensor(y_batch), 1.0)

                yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False), seq_lens
                x_batch, y_batch, seq_lens = [], [], []

    def iter_eval(self):

        x_batch, y_batch, seq_lens = [], [], []
        test_movies, test_movies_r = [], []
        
        users_done = 0
        
        for user in range(len(self.data)):
            
            users_done += 1
            if users_done > self.hyper_params['number_users_to_keep']: break
            
            if self.is_training is True:
                split = float(self.hyper_params['history_split_test'][0])
                split_idx = int(split * len(self.data[user]))
                if split_idx <= 1:
                    # at least 2 items in base_predictions_on
                    split_idx = 2
                base_predictions_on = self.data[user][:split_idx]
                heldout_movies = self.data[user][split_idx:]
            else:
                base_predictions_on = self.data[user]
                heldout_movies = self.data_te[user]
                
            test_movies.append([ i[0] for i in heldout_movies ])
            test_movies_r.append([ i[1] for i in heldout_movies ])
            x_batch.append([ i[0] for i in base_predictions_on[:-1] ])
            y_batch.append([ i[0] - 1 for i in base_predictions_on[1:] ])
            seq_lens.append(len(base_predictions_on) - 1)

            assert len(base_predictions_on) >= 2 and len(heldout_movies) > 0, \
                f'bad input data. base is {len(base_predictions_on)} and held data is {len(heldout_movies)}'
            
            if len(x_batch) == self.batch_size:
                mid_dimension = max(seq_lens)
                
                x_batch = self.pad(x_batch, mid_dimension)
                y_batch = self.pad(y_batch, mid_dimension, for_scatter = True)

                # x_batch = list(reversed(x_batch))
                # y_batch = list(reversed(y_batch))
                # seq_lens = list(reversed(seq_lens))
                # test_movies = list(reversed(test_movies))
                indexes = sorted(range(len(x_batch)), key=lambda x: - seq_lens[x])
                x_batch = [x_batch[i] for i in indexes]
                y_batch = [y_batch[i] for i in indexes]
                seq_lens = [seq_lens[i] for i in indexes]
                test_movies = [test_movies[i] for i in indexes]
                test_movies_r = [test_movies_r[i] for i in indexes]

                y_batch_s = torch.zeros(self.batch_size, mid_dimension, self.num_items)
                if is_cuda_available: y_batch_s = y_batch_s.cuda()

                y_batch_s.scatter_(-1, LongTensor(y_batch), 1.0)
                
                yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False), test_movies, test_movies_r, seq_lens
                x_batch, y_batch, seq_lens = [], [], []
                test_movies, test_movies_r = [], []


# # Model
class EncoderDecoder(nn.Module):
    def __init__(self, p_dims):
        super(EncoderDecoder, self).__init__()
        self.p_dims = p_dims

        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        for layer in self.p_layers:
            nn.init.xavier_normal(layer.weight)

        self.activation = nn.Tanh()

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = self.activation(h)
        return h


class ModelSLD(nn.Module):
    def __init__(self, hyper_params):
        super(ModelSLD, self).__init__()
        self.hyper_params = hyper_params

        self.encoder = EncoderDecoder((hyper_params['rnn_size'],
                                       hyper_params['hidden_size'],
                                       2 * hyper_params['latent_size']))

        self.decoder = EncoderDecoder((hyper_params['latent_size'],
                               hyper_params['hidden_size'],
                               hyper_params['total_items']))

        self.item_embed = nn.Embedding(hyper_params['total_items'] + 1, hyper_params['item_embed_size'])

        self.gru = nn.GRU(
            hyper_params['item_embed_size'], hyper_params['rnn_size'],
            batch_first=True, num_layers=1
        )

        self.tanh = nn.Tanh()

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = h_enc[:, :self.hyper_params['latent_size']]
        log_sigma = h_enc[:, self.hyper_params['latent_size']:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if is_cuda_available: std_z = std_z.cuda()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x, x_lens):
        in_shape = x.shape  # [bsz x seq_len] = [1 x seq_len]
        x = x.view(-1)  # [seq_len]

        x = self.item_embed(x)  # [seq_len x embed_size]
        x = x.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x embed_size]

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)

        rnn_out, _ = self.gru(x)  # [1 x seq_len x rnn_size]
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out = rnn_out.contiguous().view(in_shape[0] * in_shape[1], -1)  # [seq_len x rnn_size]

        enc_out = self.encoder(rnn_out)  # [seq_len x 2*hidden_size]
        sampled_z = self.sample_latent(enc_out)  # [seq_len x latent_size]

        dec_out = self.decoder(sampled_z)  # [seq_len x total_items]
        dec_out = dec_out.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x total_items]

        return dec_out, self.z_mean, self.z_log_sigma


class ModelPE(nn.Module):
    def __init__(self, hyper_params):
        super(ModelPE, self).__init__()
        self.hyper_params = hyper_params
        self.n_items = hyper_params['total_items']

        self.encoder = EncoderDecoder((hyper_params['item_embed_size'],
                                       hyper_params['hidden_size'],
                                       2 * hyper_params['latent_size']))

        self.decoder = EncoderDecoder((hyper_params['latent_size'] + hyper_params['rnn_size'],
                               hyper_params['hidden_size'],
                               hyper_params['total_items']))

        self.item_embed = nn.Embedding(hyper_params['total_items'] + 1, hyper_params['item_embed_size'])

        self.gru = nn.GRU(
            hyper_params['item_embed_size'], hyper_params['rnn_size'],
            batch_first=True, num_layers=1
        )

        self.tanh = nn.Tanh()

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = h_enc[:, :self.hyper_params['latent_size']]
        log_sigma = h_enc[:, self.hyper_params['latent_size']:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if is_cuda_available:
            std_z = std_z.cuda()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x, x_lens):
        bsz, seqlen = x.shape  # [bsz x seq_len]

        x_embed = self.item_embed(x)  # [bsz x seq_len x embed_size]

        # encoder -> Z
        x_pre = x_embed[:, :(seqlen - 1), :]  # [bsz x (seq_len-1) x embed_size]
        x_pre = x_pre.sum(1)  # [bsz x embed_size]
        x_enc = self.encoder(x_pre)  # [bsz x 2 * latent_size]

        sampled_z = self.sample_latent(x_enc)  # [bsz x latent_size]
        sampled_z = sampled_z.repeat_interleave(seqlen, dim=0)  # [bsz x seqlen x latent_size]
        sampled_z = sampled_z.view(bsz, seqlen, -1)

        assert len([i for i in x_lens if i == 0]) == 0
        # GRU -> hist(x)
        x_hist = x_embed.view(bsz, seqlen, -1)  # [bsz x seq_len x embed_size]
        x_hist = torch.nn.utils.rnn.pack_padded_sequence(x_hist, x_lens, batch_first=True)
        rnn_out, _ = self.gru(x_hist)  # [bsz x seq_len x rnn_size]
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out = rnn_out.contiguous().view(bsz, seqlen, -1)  # [bsz x seq_len x rnn_size]

        # decoder -> y
        dec_in = torch.cat((rnn_out, sampled_z), -1)
        dec_out = self.decoder(dec_in)  # [bsz x seq_len x total_items]

        dec_out = dec_out.view(bsz, seqlen, self.n_items)  # [bsz x seq_len x total_items]

        # fix shape of mu and sigma
        z_mu = self.z_mean.repeat_interleave(seqlen, dim=0).view(bsz * seqlen, -1)
        z_log_sigma = self.z_log_sigma.repeat_interleave(seqlen, dim=0).view(bsz * seqlen, -1)

        return dec_out, z_mu, z_log_sigma


class ModelMLD(nn.Module):
    def __init__(self, hyper_params):
        super(ModelMLD, self).__init__()
        self.hyper_params = hyper_params

        self.encoder = EncoderDecoder((hyper_params['rnn_size'],
                                       hyper_params['hidden_size'],
                                       2 * hyper_params['latent_size']))

        self.decoder = EncoderDecoder((hyper_params['latent_size_rnn'],
                               hyper_params['hidden_size'],
                               hyper_params['total_items']))

        self.item_embed = nn.Embedding(hyper_params['total_items'] + 1, hyper_params['item_embed_size'])

        self.gru = nn.GRU(
            hyper_params['item_embed_size'], hyper_params['rnn_size'],
            batch_first=True, num_layers=1
        )
        self.gru_z = nn.GRU(
            hyper_params['latent_size'], hyper_params['latent_size_rnn'],
            batch_first=True, num_layers=1
        )

        self.tanh = nn.Tanh()

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = h_enc[:, :, :self.hyper_params['latent_size']]
        log_sigma = h_enc[:, :, self.hyper_params['latent_size']:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if is_cuda_available: std_z = std_z.cuda()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x, x_lens):
        bsz, seqlen = x.shape  # [bsz x seq_len] = [1 x seq_len]
        x = x.view(-1)  # [seq_len]

        x = self.item_embed(x)  # [seq_len x embed_size]
        x = x.view(bsz, seqlen, -1)  # [bsz x seq_len x embed_size]

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        rnn_out, _ = self.gru(x)  # [bsz x seq_len x rnn_size]
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out = rnn_out.contiguous().view(bsz, seqlen, -1)  # [bsz * seq_len x rnn_size]

        enc_out = self.encoder(rnn_out)  # [bsz x seq_len x 2*hidden_size]
        sampled_z = self.sample_latent(enc_out)  # [bsz x seq_len x latent_size]

        sampled_z = torch.nn.utils.rnn.pack_padded_sequence(sampled_z, x_lens, batch_first=True)
        sampled_z, _ = self.gru_z(sampled_z)
        sampled_z, _ = torch.nn.utils.rnn.pad_packed_sequence(sampled_z, batch_first=True)
        sampled_z = sampled_z.contiguous().view(bsz * seqlen, -1)  # [bsz * seq_len x latent_rnn_size]

        dec_out = self.decoder(sampled_z)  # [bsz * seq_len x total_items]
        dec_out = dec_out.view(bsz, seqlen, -1)  # [bsz x seq_len x total_items]

        return dec_out, self.z_mean.view(bsz * seqlen, -1), self.z_log_sigma.view(bsz * seqlen, -1)


class ZClippingActivation(nn.Module):
    def forward(self, tensor):
        return nn.functional.threshold(tensor, threshold=1e-3, value=1e-3)


class ModelRLD(nn.Module):
    def __init__(self, hyper_params):
        super(ModelRLD, self).__init__()
        self.hyper_params = hyper_params
        self.latent_size = hyper_params['latent_size']

        self.encoder = EncoderDecoder((hyper_params['rnn_size'],
                                       hyper_params['hidden_size'],
                                       2 * hyper_params['latent_size']))

        self.decoder = EncoderDecoder((hyper_params['latent_size'],
                               hyper_params['hidden_size'],
                               hyper_params['total_items']))

        self.item_embed = nn.Embedding(hyper_params['total_items'] + 1, hyper_params['item_embed_size'])

        self.gru = nn.GRU(
            hyper_params['item_embed_size'], hyper_params['rnn_size'],
            batch_first=True, num_layers=1
        )

        self.gru_x = nn.GRU(
            hyper_params['item_embed_size'], 2 * hyper_params['latent_size'],
            batch_first=True, num_layers=1
        )

        self._z_clipping = nn.Linear(2 * hyper_params['latent_size'], 2 * hyper_params['latent_size'])
        self._z_clipping_activation = ZClippingActivation()

        self.tanh = nn.Tanh()

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = h_enc[:, :self.hyper_params['latent_size']]
        log_sigma = h_enc[:, self.hyper_params['latent_size']:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if is_cuda_available: std_z = std_z.cuda()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x, x_lens):
        bsz, seqlen = x.shape  # [bsz x seq_len]
        x = x.view(-1)  # [bsz * seq_len]

        x = self.item_embed(x)  # [bsz * seq_len * embed_size]
        x = x.view(bsz, seqlen, -1)  # [bsz x seq_len x embed_size]

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        rnn_out, _ = self.gru(x)  # [bsz x  seq_len x rnn_size]
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out = rnn_out.contiguous().view(bsz * seqlen, -1)  # [bsz * seq_len x rnn_size]

        enc_out = self.encoder(rnn_out)  # [bsz * seq_len x 2*hidden_size]
        sampled_z = self.sample_latent(enc_out)  # [bsz * seq_len x latent_size]

        dec_out = self.decoder(sampled_z)  # [bsz * seq_len x total_items]
        dec_out = dec_out.view(bsz, seqlen, -1)  # [bsz x seq_len x total_items]

        # Zt | X1:t-1
        # x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        rnn_z_out, _ = self.gru_x(x)  # [bsz x  seq_len x 2 * latent_size]
        rnn_z_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_z_out, batch_first=True)
        rnn_z_out = rnn_z_out.contiguous().view(bsz * seqlen, -1)  # [bsz * seq_len x 2 * latent_size]

        rnn_z_out = self._z_clipping(rnn_z_out)
        rnn_z_out = self._z_clipping_activation(rnn_z_out)
        z_t_mean = rnn_z_out[:, :self.latent_size]
        z_t_log_sigma = rnn_z_out[:, self.latent_size:]

        return dec_out, self.z_mean, self.z_log_sigma, z_t_mean, z_t_log_sigma



# # Custom loss
# 
# $$ Loss \; = \; \sum_{u \in U} Loss_u $$ <br>
# $$ Loss_u \; = \; \beta * KL( \, \phi(z \vert x) \, \Vert \, {\rm I\!N(0, I)} \, ) \; - \; log( \, P_{\phi}(g_{\theta}(x)) \, ) $$ <br>
# $ g_{\theta}(.)$ is the encoder ; $P_{\phi}(.)$ is the decoded distribution; $ \beta $ is the anneal factor.

def sequence_mask(lengths, fsize=None, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask = mask.type(dtype)
    if fsize is None:
        return mask
    return mask.unsqueeze(-1).repeat(1,1,fsize)


class VAELoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(VAELoss, self).__init__()
        self.hyper_params = hyper_params
        self._kld_avg_on_batch = self.hyper_params['kld_avg_on_batch']
        self._kld_type = 1 if self.hyper_params['kld_type'] == 'old' else 0

    def KLD(self,mu_q,logvar_q,seq_lens):

        kld_contribs = torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), -1)

        mask = sequence_mask(torch.tensor(seq_lens),dtype=torch.float).view(-1)
        if is_cuda_available:
            mask = mask.cuda()

        if self._kld_avg_on_batch:
            num_users = float(len(seq_lens))
        else:
            num_users = mask.sum()

        kld = torch.sum(mask*kld_contribs,-1)/num_users

        return kld

    def forward(self, decoder_output, mu_q, logvar_q, y_true_s, y_lens, anneal):
        # Calculate KL Divergence loss
        if self._kld_type == 0:
            kld = self.KLD(mu_q,logvar_q,y_lens)
        else:
            kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q**2 - 1), -1))

        # Calculate Likelihood
        dec_shape = decoder_output.shape # [batch_size x seq_len x total_items]

        decoder_output = F.log_softmax(decoder_output, -1)

        mask = sequence_mask(torch.tensor(y_lens),fsize=dec_shape[2],dtype=torch.float)
        if is_cuda_available: mask = mask.cuda()
        y_true_s = y_true_s*mask

        num_ones = float(len(y_lens))

        likelihood = torch.sum(
            -1.0 * y_true_s.view(-1) * decoder_output.view(-1)
        ) / num_ones

        final = (anneal * kld) + (likelihood)

        return final


class VAELossRLD(VAELoss):
    def __init__(self, hyper_params):
        super(VAELossRLD, self).__init__(hyper_params)

        if self._kld_type != 0:
            raise RuntimeError(f'only kld_type type 0 is supported')

    def KLD(self, mu_q, logvar_q, z_t_mean, z_t_log_sigma, seq_lens):
        # mu_q, logvar_q, z_t_mean, z_t_log_sigma => [bsz * seq_len x * latent_size]

        mask = sequence_mask(torch.tensor(seq_lens), dtype=torch.float).view(-1).to(mu_q.device)

        # https://stats.stackexchange.com/a/7449
        a = z_t_log_sigma - logvar_q
        b = (logvar_q ** 2 + (mu_q - z_t_mean) ** 2) / (2 * (z_t_log_sigma ** 2))
        b[torch.isinf(b)] = 0  # reset inf values
        kld_contribs = torch.sum(a + b - .5, -1)

        if self._kld_avg_on_batch:
            num_users = float(len(seq_lens))
        else:
            num_users = mask.sum()

        kld = torch.sum(mask * kld_contribs, -1) / num_users

        return kld

    def forward(self, decoder_output, mu_q, logvar_q, z_t_mean, z_t_log_sigma, y_true_s, y_lens, anneal):
        # Calculate KL Divergence loss
        kld = self.KLD(mu_q, logvar_q, z_t_mean, z_t_log_sigma, y_lens)

        # Calculate Likelihood
        dec_shape = decoder_output.shape # [batch_size x seq_len x total_items]

        decoder_output = F.log_softmax(decoder_output, -1)

        mask = sequence_mask(torch.tensor(y_lens), fsize=dec_shape[2], dtype=torch.float)
        if is_cuda_available:
            mask = mask.cuda()
        y_true_s = y_true_s * mask

        num_ones = float(len(y_lens))

        likelihood = torch.sum(
            -1.0 * y_true_s.view(-1) * decoder_output.view(-1)
        ) / num_ones

        final = (anneal * kld) + likelihood

        return final


# # Training loop
def train(reader):
    model.train()
    total_loss = 0
    start_time = time.time()
    batch = 0
    batch_last = 0
    batch_limit = int(train_reader.num_b)
    batch_log_interval = batch_limit // 5
    global total_anneal_steps
    global anneal
    global update_count
    global anneal_cap
    if hyper_params['anneal_on_epoch']:
        anneal = 0.0
        update_count = 0.0

    isRLDmodel = hyper_params['model_type'] == 'RLD'

    for x, y_s, seq_lens in reader.iter():
        batch += 1
        
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()
    
        # Forward pass
        model_out = model(x, seq_lens)
        
        # Backward pass
        if not isRLDmodel:
            decoder_output, z_mean, z_log_sigma = model_out
            loss = criterion(decoder_output, z_mean, z_log_sigma, y_s, seq_lens, anneal)
        else:
            decoder_output, z_mean, z_log_sigma, z_t_mean, z_t_log_sigma = model_out
            loss = criterion(decoder_output,
                             z_mean, z_log_sigma,
                             z_t_mean, z_t_log_sigma,
                             y_s, seq_lens, anneal)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Anneal logic
        if total_anneal_steps > 0:
            anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
        else:
            anneal = anneal_cap
        update_count += 1.0
        
        # Logging mechanism
        if (batch % batch_log_interval == 0 and batch > 0) or batch == batch_limit:

            cur_loss = total_loss / batch
            elapsed = time.time() - start_time
            
            ss = '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'.format(
                    epoch, batch, batch_limit, (elapsed * 1000) / (batch - batch_last), cur_loss
            )
            
            file_write(hyper_params['log_file'], ss)

            total_loss = 0
            batch_last = batch
            start_time = time.time()

# # Evaluation Code

def evaluate_all_tail(model, criterion, reader, hyper_params, is_train_set, vd_anneal = 0.2):
    model.eval()

    metrics = {}
    metrics['loss'] = 0.0
    Ks = [1, 10, 100]
    for k in Ks:
        metrics['NDCG@' + str(k)] = 0.0
        metrics['Rec@' + str(k)] = 0.0
        metrics['Prec@' + str(k)] = 0.0

    isRLDmodel = hyper_params['model_type'] == 'RLD'
    batch = 0
    total_users = 0

    # For plotting the results (seq length vs. NDCG@100)
    len_to_ndcg_at_100_map = {}

    for x, y_s, test_movies, _, X_length in reader.iter_eval():
        batch += 1
        if is_train_set is True and batch > hyper_params['train_cp_users']:
            break

        model_out = model(x, X_length)

        if not isRLDmodel:
            decoder_output, z_mean, z_log_sigma = model_out
            loss = criterion(decoder_output, z_mean, z_log_sigma, y_s, X_length, anneal)
        else:
            decoder_output, z_mean, z_log_sigma, z_t_mean, z_t_log_sigma = model_out
            loss = criterion(decoder_output,
                             z_mean, z_log_sigma,
                             z_t_mean, z_t_log_sigma,
                             y_s, X_length, anneal)

        metrics['loss'] += loss.item()

        # iter on batches along with sequence length of the user
        for batch_idx, seq_len in enumerate(X_length):
            # select predictions for the current user, ignoring watched movies
            last_movie_in_seq = seq_len - 2
            # find watched movie ids of current user
            actual_movies_watched = np.array(test_movies[batch_idx]) - 1

            viewed_movies = x[batch_idx, :seq_len] - 1

            for i in range(len(actual_movies_watched)):
                last_movie_in_seq += 1
                predicted_scores = decoder_output[batch_idx, last_movie_in_seq, :]  # [#items]
                # reset prediction for watched film of the user
                predicted_scores.scatter_(0, viewed_movies, -1e20)
                if i > 0:
                    predicted_scores.scatter_(0, actual_movies_watched[:i], -1e20)

                # Calculate NDCG
                idx_of_top_maxK_predictions = torch.argsort(predicted_scores, descending=True)
                for k in Ks:
                    best, dcg = 0.0, 0.0
                    hits, now_at = 0, 0

                    # build a dict with movie id as key, and relative position in the predictions as value
                    # positions are 1-based
                    iter_on_topK_prediction = enumerate(idx_of_top_maxK_predictions[:k].tolist())
                    rec_list = {movie_id: position + 1 for position, movie_id in iter_on_topK_prediction}

                    for m in range(i, len(actual_movies_watched)):
                        movie = actual_movies_watched[m]
                        now_at += 1
                        if now_at <= k:
                            best += 1.0 / np.log2(now_at + 1)

                        if movie in rec_list:
                            hits += 1
                            # dcg += 1.0 / float(np.log2(float(rec_list.index(movie) + 2)))
                            dcg += 1.0 / np.log2(rec_list[movie] + 1)

                    metrics['NDCG@' + str(k)] += dcg / best
                    metrics['Rec@' + str(k)] += float(hits) / float(len(actual_movies_watched))
                    metrics['Prec@' + str(k)] += float(hits) / float(k)

                    # Only for plotting the graph (seq length vs. NDCG@100)
                    if k == 100 and i == 0:
                        seq_len = len(actual_movies_watched) + X_length[batch_idx] + 1
                        if seq_len not in len_to_ndcg_at_100_map:
                            len_to_ndcg_at_100_map[seq_len] = []
                        len_to_ndcg_at_100_map[seq_len].append(dcg / best)

                total_users += 1

    metrics['loss'] = float(metrics['loss']) / float(batch)
    metrics['loss'] = round(metrics['loss'], 4)

    for k in Ks:
        metrics['NDCG@' + str(k)] = round((100.0 * metrics['NDCG@' + str(k)]) / float(total_users), 4)
        metrics['Rec@' + str(k)] = round((100.0 * metrics['Rec@' + str(k)]) / float(total_users), 4)
        metrics['Prec@' + str(k)] = round((100.0 * metrics['Prec@' + str(k)]) / float(total_users), 4)

    return metrics, len_to_ndcg_at_100_map


def evaluate(model, criterion, reader, hyper_params, is_train_set, vd_anneal = 0.2, is_test=False):
    model.eval()

    metrics = {}
    metrics['loss'] = 0.0
    Ks = [1, 10, 100]
    for k in Ks:
        metrics['NDCG@' + str(k)] = 0.0
        metrics['Rec@' + str(k)] = 0.0
        metrics['Prec@' + str(k)] = 0.0

    isRLDmodel = hyper_params['model_type'] == 'RLD'
    batch = 0
    total_users = 0

    # For plotting the results (seq length vs. NDCG@100)
    len_to_ndcg_at_100_map = {}

    for x, y_s, test_movies, _, X_length in reader.iter_eval():
        batch += 1
        if is_train_set is True and batch > hyper_params['train_cp_users']:
            break

        model_out = model(x, X_length)

        if not isRLDmodel:
            decoder_output, z_mean, z_log_sigma = model_out
            loss = criterion(decoder_output, z_mean, z_log_sigma, y_s, X_length, vd_anneal)
        else:
            decoder_output, z_mean, z_log_sigma, z_t_mean, z_t_log_sigma = model_out
            loss = criterion(decoder_output,
                             z_mean, z_log_sigma,
                             z_t_mean, z_t_log_sigma,
                             y_s, X_length, vd_anneal)

        metrics['loss'] += loss.item()

        # iter on batches along with sequence length of the user
        for batch_idx, seq_len in enumerate(X_length):
            # select predictions for the current user, ignoring watched movies
            last_movie_in_seq = seq_len - 1
            predicted_scores = decoder_output[batch_idx, last_movie_in_seq, :]  # [#items]
            # reset prediction for watched film of the user
            viewed_movies = x[batch_idx,:seq_len] - 1
            predicted_scores.scatter_(0, viewed_movies, -1e20)

            # find watched movie ids of current user
            actual_movies_watched = test_movies[batch_idx]

            # Calculate NDCG
            idx_of_top_maxK_predictions = torch.argsort(predicted_scores, descending=True)
            for k in Ks:
                best, dcg = 0.0, 0.0
                hits, now_at = 0, 0

                # build a dict with movie id as key, and relative position in the predictions as value
                # positions are 1-based
                iter_on_topK_prediction = enumerate(idx_of_top_maxK_predictions[:k].tolist())
                rec_list = {movie_id: position + 1 for position, movie_id in iter_on_topK_prediction}

                for m in range(len(actual_movies_watched)):
                    movie = actual_movies_watched[m] - 1
                    now_at += 1
                    if now_at <= k:
                        best += 1.0 / np.log2(now_at + 1)

                    if movie in rec_list:
                        hits += 1
                        # dcg += 1.0 / float(np.log2(float(rec_list.index(movie) + 2)))
                        dcg += 1.0 / np.log2(rec_list[movie] + 1)

                metrics['NDCG@' + str(k)] += dcg / best
                metrics['Rec@' + str(k)] += float(hits) / float(len(actual_movies_watched))
                metrics['Prec@' + str(k)] += float(hits) / float(k)

                # Only for plotting the graph (seq length vs. NDCG@100)
                if k == 100:
                    seq_len = len(actual_movies_watched) + X_length[batch_idx] + 1
                    if seq_len not in len_to_ndcg_at_100_map:
                        len_to_ndcg_at_100_map[seq_len] = []
                    len_to_ndcg_at_100_map[seq_len].append(dcg / best)

            total_users += 1

    metrics['loss'] = float(metrics['loss']) / float(batch)
    metrics['loss'] = round(metrics['loss'], 4)

    for k in Ks:
        metrics['NDCG@' + str(k)] = round((100.0 * metrics['NDCG@' + str(k)]) / float(total_users), 4)
        metrics['Rec@' + str(k)] = round((100.0 * metrics['Rec@' + str(k)]) / float(total_users), 4)
        metrics['Prec@' + str(k)] = round((100.0 * metrics['Prec@' + str(k)]) / float(total_users), 4)

    return metrics, len_to_ndcg_at_100_map


# # Main procedure
torch.manual_seed(hyper_params['seed'])
np.random.seed(hyper_params['seed'])
if is_cuda_available:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Train It..
train_reader, val_reader, test_reader, total_items = load_data(hyper_params)
hyper_params['total_items'] = total_items
hyper_params['testing_batch_limit'] = test_reader.num_b

file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
file_write(hyper_params['log_file'], "Data reading complete!")
file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(train_reader.num_b))
file_write(hyper_params['log_file'], "Number of validation batches: {:4d}".format(val_reader.num_b))
file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(test_reader.num_b))
file_write(hyper_params['log_file'], "Total Items: " + str(total_items) + "\n")
file_write(hyper_params['log_file'], "Model: " + hyper_params['model_type'] + "\n")

if hyper_params['model_type'] == 'PE':
    model = ModelPE(hyper_params)
    criterion = VAELoss(hyper_params)
elif hyper_params['model_type'] == 'SLD':
    model = ModelSLD(hyper_params)
    criterion = VAELoss(hyper_params)
elif hyper_params['model_type'] == 'MLD':
    model = ModelMLD(hyper_params)
    criterion = VAELoss(hyper_params)
elif hyper_params['model_type'] == 'RLD':
    model = ModelRLD(hyper_params)
    criterion = VAELossRLD(hyper_params)
else:
    raise RuntimeError('model type not recognized: %s' % hyper_params['model_type'])

if is_cuda_available:
    model.cuda()

if hyper_params['optimizer'] == 'adagrad':
    optimizer = torch.optim.Adagrad(
        model.parameters(), weight_decay=hyper_params['weight_decay'], lr = hyper_params['learning_rate']
    )
elif hyper_params['optimizer'] == 'adadelta':
    optimizer = torch.optim.Adadelta(
        model.parameters(), weight_decay=hyper_params['weight_decay']
    )
elif hyper_params['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=hyper_params['weight_decay'], lr = hyper_params['learning_rate']
    )
elif hyper_params['optimizer'] == 'rmsprop':
    optimizer = torch.optim.RMSprop(
        model.parameters(), weight_decay=hyper_params['weight_decay']
    )

file_write(hyper_params['log_file'], str(model))
file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

best_val_ndcg = None

try:
    total_anneal_steps = hyper_params['total_anneal_steps']
    anneal = 0.0
    update_count = 0.0
    anneal_cap = hyper_params['anneal_cap']

    for epoch in range(1, hyper_params['epochs'] + 1):
        epoch_start_time = time.time()

        train(train_reader)

        # Calulating the metrics on the train set
        metrics, _ = evaluate(model, criterion, train_reader, hyper_params, True, anneal)
        string = ""
        for m in metrics: string += " | " + m + ' = ' + str(metrics[m])
        string += ' (TRAIN)'
    
        # Calulating the metrics on the validation set
        metrics, _ = evaluate(model, criterion, val_reader, hyper_params, False, anneal)
        string2 = ""
        for m in metrics: string2 += " | " + m + ' = ' + str(metrics[m])
        string2 += ' (VAL)'

        ss  = '-' * 89
        ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        ss += string
        ss += '\n'
        ss += '-' * 89
        ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        ss += string2
        ss += '\n'
        ss += '-' * 89
        file_write(hyper_params['log_file'], ss)
        
        if not best_val_ndcg or metrics['NDCG@100'] >= best_val_ndcg:
            with open(hyper_params['model_file_name'], 'wb') as f:
                torch.save(model, f)
            best_val_ndcg = metrics['NDCG@100']

except KeyboardInterrupt: print('Exiting from training early')

# Plot Traning graph
f = open(model.hyper_params['log_file'])
lines = f.readlines()
lines.reverse()

train = []
test = []

for line in lines:
    if line[:10] == 'Simulation' and len(train) > 1: break
    elif line[:10] == 'Simulation' and len(train) <= 1: train, test = [], []
        
    if line[2:5] == 'end' and line[-5:-2] == 'VAL': test.append(line.strip().split("|"))
    elif line[2:5] == 'end' and line[-7:-2] == 'TRAIN': train.append(line.strip().split("|"))

train.reverse()
test.reverse()

train_ndcg = []
test_ndcg = []
test_loss, train_loss = [], []

for i in train:
    for metric in i:
        if metric.split("=")[0] == " NDCG@100 ":
            train_ndcg.append(float(metric.split('=')[1].split(' ')[1]))
        if metric.split("=")[0] == " loss ":
            train_loss.append(float(metric.split("=")[1].split(' ')[1]))

total, avg_runtime = 0.0, 0.0
for i in test:
    avg_runtime += float(i[2].split(" ")[2][:-1])
    total += 1.0
    
    for metric in i:
        if metric.split("=")[0] == " NDCG@100 ":
            test_ndcg.append(float(metric.split('=')[1].split(' ')[1]))
        if metric.split("=")[0] == " loss ":
            test_loss.append(float(metric.split("=")[1].split(' ')[1]))

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.set_title(hyper_params["project_name"],fontweight="bold", size=20)
ax1.plot(test_ndcg, 'b-')
ax1.set_xlabel('Epochs', fontsize = 20.0)
ax1.set_ylabel('NDCG@100', color='b', fontsize = 20.0)
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(test_loss, 'r--')
ax2.set_ylabel('Loss', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
if not os.path.isdir("saved_plots/"):
    os.mkdir("saved_plots/")
fig.savefig("saved_plots/learning_curve_" + hyper_params["project_name"] + ".pdf")
plt.show()

# Checking metrics for the test set on best saved model
with open(hyper_params['model_file_name'], 'rb') as f:
    model = torch.load(f)
metrics, len_to_ndcg_at_100_map = evaluate(model, criterion, test_reader, hyper_params, False, is_test=True)

# Plot sequence length vs NDCG@100 graph
plot_len_vs_ndcg(len_to_ndcg_at_100_map)

string = ""
for m in metrics: string += " | " + m + ' = ' + str(metrics[m])

ss  = '=' * 89
ss += '\n| End of training'
ss += string + " (TEST)"
ss += '\n'
ss += '=' * 89
file_write(hyper_params['log_file'], ss)
file_write(hyper_params['log_file'], "average runtime per epoch = %s sec" % round(avg_runtime / float(total), 4))

with open(os.path.join(log_file_root, 'config.json'), 'w') as fp:
    json.dump(hyper_params, fp)
print('DONE')
