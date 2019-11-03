import os
import time
from typing import Mapping
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import r2_score

from omtf_reg.pytorch_approach.net import omtfNet
from omtf_reg.pytorch_approach.dataset import omtfDataset


class omtfModel:
    def __init__(self, dataloaders: Mapping[str, torch.utils.data.DataLoader], loss_fn=nn.SmoothL1Loss(), experiment_dirpath: str = '../omtfNet', snapshot_frequency: int = 10):
        self._loss_fn = loss_fn
        self.dataloaders = dataloaders
        self.experiment_dirpath = experiment_dirpath
        self.snapshot_frequency = snapshot_frequency
        self.net = None

    def train(self, init_lr=1e-3, epochs=20):
        assert 'TRAIN' in self.dataloaders.keys(), 'Missing train dataloader'
        assert 'VALID' in self.dataloaders.keys(), 'Missing validation dataloader'

        self.net = omtfNet() if self.net is None else self.net
        self.net.cuda()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=init_lr)

        preds_to_save = {}
        preds_to_save['TRAIN'] = {}
        preds_to_save['VALID'] = {}

        labels_to_save = {}
        labels_to_save['TRAIN'] = []
        labels_to_save['VALID'] = []

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

                    optimizer.zero_grad()
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
                            optimizer.step()

                np.savez_compressed(os.path.join(preds_save_path,
                                     f'{phase}_{epoch}.npz'), data=gathered_preds)
                np.savez_compressed(os.path.join(labels_save_path,
                                     f'{phase}_{epoch}.npz'), data=gathered_labels)

                phase_loss /= len(self.dataloaders[phase])
                print(
                    f'{phase} LOSS {phase_loss}, r2 {self.get_statistics(gathered_labels, gathered_preds)}')

            if epoch % self.snapshot_frequency == 0:
                self.save_model()
                print('Snapshot successfully created!')

            t2 = time.time()
            print(f'Time elapsed: {t2-t1} seconds\n')


        self.save_model()

    def predict(self):
        assert 'TEST' in self.dataloaders.keys(), 'Missing test dataloader'
        self.net.cuda()
        self.net.eval()

        gathered_labels = []
        gathered_preds = []
        phase_loss = 0
        for item in self.dataloaders['TEST']:
            X, y_gt = item
            X = X.cuda()
            y_gt = y_gt.cuda()
            y_pred = self.net(X)
            loss = self._loss_fn(y_pred, y_gt)
            gathered_preds = gathered_preds + y_pred.view(-1).tolist()
            gathered_labels = gathered_labels + y_gt.view(-1).tolist()
            phase_loss += loss.item()
        phase_loss /= len(self.dataloaders['TEST'])
        print(
            f'\nTEST LOSS {phase_loss}, r2 {self.get_statistics(gathered_labels, gathered_preds)}')

    def save_model(self):
        os.makedirs(self.experiment_dirpath, exist_ok=True)
        torch.save(self.net, os.path.join(
            self.experiment_dirpath, 'model.pth'))

    def load_model(self, model_dirpath):
        net = torch.load(model_dirpath)
        self.net = net
        print('Model loaded successfully!')

    def get_statistics(self, gathered_labels, gathered_preds):
        r2 = r2_score(gathered_labels, gathered_preds)
        return r2
