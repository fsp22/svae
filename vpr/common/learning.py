import json
import os
from collections import defaultdict
import numpy as np
import time
import torch

from vpr.common.logger import log
from vpr.common.chart import save_loss_chart


def epoch_log(metrics, top_k, epoch, epoch_start_time, log_file):
    metric_to_print = ['train_loss', 'val_loss']

    for k in top_k:
        metric_to_print.append(f'Prec@{k}')
        metric_to_print.append(f'Rec@{k}')
        metric_to_print.append(f'NDCG@{k}')

    print_string = ' | '.join([print_metric(k, v) for k, v in metrics if k in metric_to_print])
    epochstr = f'{epoch:3d}' if isinstance(epoch, int) else epoch
    print_string = f'| Epoch {epochstr} | time: {time.time() - epoch_start_time:4.2f}s | {print_string} |'
    line_separator_len = len(print_string)

    print('-' * line_separator_len)
    log(print_string, log_file)
    print('-' * line_separator_len)


def compute_metric(all_items, scores, mask, positives, pos_lengths, k):
    def get_idcg(n_pos_items, top_k):
        idcg = 0

        for i in range(min(n_pos_items, top_k)):
            idcg += 1 / np.log2(i + 2)

        return idcg

    avg_ndcg = 0.0
    avg_pr = 0
    avg_rec = 0

    for i in range(scores.shape[0]):
        score = scores[i, mask[i, :]]
        items = all_items[i, mask[i, :]]
        positive_items = set(positives[i, :pos_lengths[i]].tolist())

        # Get the top k indices
        arg_index = torch.argsort(-score)[:k].tolist()

        hr, dcg = 0., 0.
        idcg = get_idcg(len(positive_items), k)  # ideal DCG

        for current_position, current_item in enumerate(items[arg_index]):
            if current_item.item() in positive_items:
                hr += 1
                dcg += 1 / np.log2(current_position + 2)

        avg_ndcg += dcg / idcg
        avg_rec += hr / len(positive_items)
        avg_pr += hr / k

    return avg_pr, avg_rec, avg_ndcg


def print_metric(k, v):
    if isinstance(v, str):
        return f'{k}: {v}'

    return f'{k}: {v:.4f}'


class Learner:
    def __init__(self, train_reader, val_reader, test_reader, net, device, optimizer, loss_function, hyper_params):
        self.train_reader = train_reader
        self.val_reader = val_reader
        self.test_reader = test_reader
        self.net = net
        self.device = device
        self.hp = hyper_params

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.anneal = 0.
        self.update_count = 0.
        self.best_loss = np.Inf

        self.top_k = [10, 100]

    def _sequence_mask(self, lengths, maxlen=None):
        if maxlen is None:
            maxlen = lengths.max()

        mask = ~(torch.ones((len(lengths), maxlen), device=self.device).cumsum(dim=1).t() > lengths).t()

        return mask

    def _testing_step(self):
        self.net.eval()

        test_result = defaultdict(float)
        n_users_train = 0

        with torch.no_grad():
            for batch_idx, (user, positives, pos_length, negatives, neg_length) in enumerate(self.test_reader):
                user = user.to(self.device)
                positives = positives.to(self.device)
                pos_length = pos_length.to(self.device)
                negatives = negatives.to(self.device)
                neg_length = neg_length.to(self.device)

                pos_mask = self._sequence_mask(pos_length, positives.shape[1])
                neg_mask = self._sequence_mask(neg_length, negatives.shape[1])
                mask = torch.cat([pos_mask, neg_mask], -1)

                n_users_train += user.shape[0]

                all_items = torch.cat([positives, negatives], -1)
                score = self.net.score(user, all_items)

                score[~mask] = -torch.tensor(np.Inf, device=self.device)

                for k in self.top_k:
                    prec, rec, ndcg = compute_metric(all_items, score, mask, positives, pos_length, k)

                    test_result[f'Prec@{k}'] += prec
                    test_result[f'Rec@{k}'] += rec
                    test_result[f'NDCG@{k}'] += ndcg

        # last metric is str
        for key in test_result:
            test_result[key] /= n_users_train

        return test_result

    def train(self):
        print('At any point you can hit Ctrl + C to break out of training early.')
        best_epoch = -1
        stat_metric = []
        self.best_loss = np.Inf
        evaluation_iterations = self.hp['evaluation_iterations']

        try:
            for epoch in range(1, self.hp['epochs'] + 1):
                epoch_start_time = time.time()

                if evaluation_iterations <= 0:
                    train_loss = self._training_step(epoch)
                    loss = self._validation_step()
                    result = self._testing_step()

                    # Collect results
                    result['train_loss'] = train_loss
                    result['val_loss'] = loss
                    stat_metric.append(result)

                    epoch_log(stat_metric[-1].items(), self.top_k, epoch, epoch_start_time, self.hp['log_file'])

                    # Save the model if the n100 is the best we've seen so far.
                    if self.best_loss > result['val_loss']:
                        with open(self.hp['model_file'], 'wb') as f:
                            torch.save(self.net, f)

                        self.best_loss = result['val_loss']
                        best_epoch = epoch
                else:
                    self._training_step_by_iterations(epoch, evaluation_iterations, stat_metric)

        except KeyboardInterrupt:
            print('-' * 89)
            log("\nM'Exiting from training early\n", self.hp['log_file'])
            
        log('Best epoch = ' + str(best_epoch), self.hp['log_file'])
        
        # CHART LOSS
        lossTrain = [sim['train_loss'] for sim in stat_metric]
        lossTest = [sim['val_loss'] for sim in stat_metric]
        lastHitRate = [sim['NDCG@10'] for sim in stat_metric]
        fname = os.path.join(self.hp['log_dir'], 'loss.png')
        save_loss_chart(lossTrain, lossTest, lastHitRate, fname)

        if stat_metric:
            fname = os.path.join(self.hp['log_dir'], f'results{self.hp["model_name"]}.json')
            with open(fname, 'w') as fp:
                json.dump(stat_metric, fp)

        return stat_metric[best_epoch - 1]

    def _training_step_by_iterations(self, epoch, evaluation_iterations, stat_metric):
        """
        Train every x iteration
        :param epoch: number of epoch
        :param evaluation_iterations: number of iterations of training
        :param stat_metric: list of results
        :return:
        """
        start_time = time.time()

        self.net.train()
        n_train_batches = 0
        train_loss, train_loss_cumulative = 0., 0.

        log_interval = self.hp['batch_log_interval']
        if log_interval == 0:
            log_interval = 1

        iterations = 0
        total_iteration = len(self.train_reader)
        for batch_idx, (user, pos, neg) in enumerate(self.train_reader):
            iterations += 1
            n_train_batches += 1
            user = user.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)

            self.optimizer.zero_grad()
            output = self.net(user, pos, neg)
            batch_loss = self.loss_function(self.anneal, *output)
            batch_loss.backward()
            self.optimizer.step()

            train_loss += batch_loss.item()
            train_loss_cumulative += batch_loss.item()

            # Anneal logic
            if self.hp['total_anneal_steps'] > 0:
                self.anneal = min(self.hp['anneal_cap'], 1. * self.update_count / self.hp['total_anneal_steps'])
            else:
                self.anneal = self.hp['anneal_cap']

            self.update_count += 1.0

            if batch_idx % log_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time

                print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | loss {:5.6f}'.format(
                    epoch, batch_idx, elapsed * 1000 / log_interval, train_loss / log_interval))

                start_time = time.time()
                train_loss = 0.0

            if iterations % evaluation_iterations == 0 or iterations == total_iteration:
                train_loss = train_loss_cumulative / n_train_batches

                loss = self._validation_step()
                result = self._testing_step()

                # Collect results
                result['train_loss'] = train_loss
                result['val_loss'] = loss
                stat_metric.append(result)

                epoch_log(stat_metric[-1].items(), self.top_k, f'{iterations}/{epoch}', 0, self.hp['log_file'])

                # Save the model if the n100 is the best we've seen so far.
                if self.best_loss > result['val_loss']:
                    with open(self.hp['model_file'], 'wb') as f:
                        torch.save(self.net, f)

                    self.best_loss = result['val_loss']

                self.net.train()
                n_train_batches = 0
                train_loss_cumulative = 0.

    def _training_step(self, epoch):
        start_time = time.time()

        self.net.train()
        n_train_batches = len(self.train_reader)
        train_loss, train_loss_cumulative = 0., 0.

        log_interval = self.hp['batch_log_interval']

        if log_interval == 0:
            log_interval = 1

        for batch_idx, (user, pos, neg) in enumerate(self.train_reader):
            user = user.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)

            self.optimizer.zero_grad()
            output = self.net(user, pos, neg)
            batch_loss = self.loss_function(self.anneal, *output)
            batch_loss.backward()
            self.optimizer.step()

            train_loss += batch_loss.item()
            train_loss_cumulative += batch_loss.item()

            # Anneal logic
            if self.hp['total_anneal_steps'] > 0:
                self.anneal = min(self.hp['anneal_cap'], 1. * self.update_count / self.hp['total_anneal_steps'])
            else:
                self.anneal = self.hp['anneal_cap']

            self.update_count += 1.0

            if batch_idx % log_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time

                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.6f}'.format(
                    epoch, batch_idx, n_train_batches, elapsed * 1000 / log_interval, train_loss / log_interval))

                start_time = time.time()
                train_loss = 0.0

        return train_loss_cumulative / n_train_batches

    def _validation_step(self):
        self.net.eval()
        loss = 0.

        n_val_batches = len(self.val_reader)

        with torch.no_grad():
            for user, pos, neg in self.val_reader:
                user = user.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)

                output = self.net(user, pos, neg)
                batch_loss = self.loss_function(self.anneal, *output)
                loss += batch_loss.item()

        return loss / n_val_batches
