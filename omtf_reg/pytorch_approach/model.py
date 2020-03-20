import os
import time
import json
from typing import Mapping
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import r2_score

from omtf_reg.pytorch_approach.net import omtfNet
from omtf_reg.pytorch_approach.plot_statistics import omtfPlotter


class omtfModel:
    def __init__(self, dataloaders: Mapping[str, torch.utils.data.DataLoader], loss_fn=nn.SmoothL1Loss(),
                 experiment_dirpath: str = '../omtfNet', snapshot_frequency: int = 10, net=None, init_lr=1e-3, weight_decay=0.0, lr_decay_rate=None):

        self._loss_fn = loss_fn
        self.dataloaders = dataloaders
        self.experiment_dirpath = experiment_dirpath
        self.snapshot_frequency = snapshot_frequency
        self.net = omtfNet() if net is None else net
        self.net.cuda()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=init_lr, weight_decay=weight_decay)
        self._lr_decay_rate = lr_decay_rate
        if self._lr_decay_rate is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=self._lr_decay_rate)

    def train(self, epochs=20):
        assert 'TRAIN' in self.dataloaders.keys(), 'Missing train dataloader'
        assert 'VALID' in self.dataloaders.keys(), 'Missing validation dataloader'

        things_to_save = {'loss': {'TRAIN': {}, 'VALID': {}}}

        preds_save_path = os.path.join(
            self.experiment_dirpath, 'predictions')
        labels_save_path = os.path.join(
            self.experiment_dirpath, 'labels')

        os.makedirs(preds_save_path, exist_ok=True)
        os.makedirs(labels_save_path, exist_ok=True)

        for epoch in range(1, epochs + 1):
            t1 = time.time()
            print(f'Epoch {epoch}')

            for phase in ['TRAIN', 'VALID']:
                if phase == 'TRAIN':
                    self.net.train()  # Set model to training mode
                else:
                    self.net.eval()   # Set model to evaluate mode

                phase_loss = 0
                gathered_labels = np.zeros(0)
                gathered_preds = np.zeros(0)
                for i, item in enumerate(self.dataloaders[phase]):
                    X, y_gt = item
                    X = X.cuda()
                    y_gt = y_gt.cuda()

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'TRAIN'):
                        y_pred = self.net(X)
                        loss = self._loss_fn(y_pred, y_gt)
                        phase_loss = phase_loss + loss.item()
                        gathered_labels = np.concatenate(
                            [gathered_labels, np.array(y_gt.view(-1).tolist())])
                        gathered_preds = np.concatenate(
                            [gathered_preds, np.array(y_pred.view(-1).tolist())])

                        if phase == 'TRAIN':
                            loss.backward()
                            self.optimizer.step()

                np.savez_compressed(os.path.join(preds_save_path,
                                                 f'{phase}_{epoch}.npz'), data=gathered_preds)
                np.savez_compressed(os.path.join(labels_save_path,
                                                 f'{phase}_{epoch}.npz'), data=gathered_labels)

                phase_loss /= len(self.dataloaders[phase])
                print(
                    f'{phase} LOSS {phase_loss}, {self.get_statistics(gathered_labels, gathered_preds)}')

                things_to_save['loss'][phase][epoch] = phase_loss

            if self._lr_decay_rate is not None:
                self.lr_scheduler.step()

            if epoch % self.snapshot_frequency == 0:
                self.save_model(epoch)
                print('Snapshot successfully created!')

            t2 = time.time()
            print(f'Time elapsed: {t2-t1} seconds\n')

        with open(os.path.join(self.experiment_dirpath, 'losses.json'), 'w') as f:
            json.dump(things_to_save, f)

        self.save_model(epoch)

    def predict(self):
        assert 'TEST' in self.dataloaders.keys(), 'Missing test dataloader'
        self.net.cuda()
        self.net.eval()

        phase_loss = 0
        gathered_labels = np.zeros(0)
        gathered_preds = np.zeros(0)
        os.makedirs(os.path.join(
            self.experiment_dirpath, 'test'), exist_ok=True)
        for item in self.dataloaders['TEST']:
            X, y_gt = item
            X = X.cuda()
            y_gt = y_gt.cuda()
            y_pred = self.net(X)
            loss = self._loss_fn(y_pred, y_gt)
            gathered_labels = np.concatenate(
                [gathered_labels, np.array(y_gt.view(-1).tolist())])
            gathered_preds = np.concatenate(
                [gathered_preds, np.array(y_pred.view(-1).tolist())])
            phase_loss += loss.item()
        phase_loss /= len(self.dataloaders['TEST'])
        np.savez_compressed(os.path.join(self.experiment_dirpath, 'test',
                                         f'labels_and_preds.npz'), data={'predictions': gathered_preds, 'labels': gathered_labels})
        print(
            f'\nTEST LOSS {phase_loss}, r2 {self.get_statistics(gathered_labels, gathered_preds)}')

    def save_model(self, epoch):
        os.makedirs(self.experiment_dirpath, exist_ok=True)
        checkpoint = {'epoch': epoch,
                      'state_dict': self.net.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict() if
                      self._lr_decay_rate is not None else None}
        torch.save(checkpoint, os.path.join(
            self.experiment_dirpath, 'model.pth'))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self._lr_decay_rate is not None and checkpoint['lr_scheduler'] is not None:
            self.lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler'])
        print('Model loaded successfully!')

    def get_net_size(self):
        assert self.net is not None, 'net attribute must be set earlier'
        net_parameters = filter(
            lambda p: p.requires_grad, self.net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def get_statistics(self, gathered_labels, gathered_preds):
        r2 = r2_score(gathered_labels, gathered_preds)
        pull = omtfPlotter.get_pull(gathered_labels, gathered_preds)
        results = f'r2: {r2}, pull: {pull}'
        return results
