import os
from typing import Mapping
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import r2_score

from net import omtfNet
from dataset import omtfDataset


class omtfModel:
    def __init__(self, dataloaders: Mapping[str, torch.utils.data.DataLoader], loss_fn=nn.SmoothL1Loss(), experiment_dirpath: str = '../omtfNet', snapshot_frequency: int = 10):
        self._loss_fn = loss_fn
        self.dataloaders = dataloaders
        self.experiment_dirpath = experiment_dirpath
        self.snapshot_frequency = snapshot_frequency
        self.net = None

    def train(self, init_lr=1e-3, epochs=20):
        self.net = omtfNet()
        self.net.cuda()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=init_lr)

        preds_to_save = {}
        preds_to_save['TRAIN'] = {}
        preds_to_save['VALID'] = {}

        labels_to_save = {}
        labels_to_save['TRAIN'] = {}
        labels_to_save['VALID'] = {}

        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}')

            for phase in ['TRAIN', 'VALID']:
                if phase == 'TRAIN':
                    self.net.train()  # Set model to training mode
                else:
                    self.net.eval()   # Set model to evaluate mode

                phase_loss = 0
                gathered_labels = []
                gathered_preds = []
                for i, item in enumerate(self.dataloaders[phase]):
                    X, y_gt = item
                    X = X.cuda()
                    y_gt = y_gt.cuda()

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'TRAIN'):
                        y_pred = self.net(X)
                        loss = self._loss_fn(y_pred, y_gt)
                        phase_loss = phase_loss + loss.item()
                        gathered_labels = gathered_labels + \
                            y_gt.view(-1).tolist()
                        gathered_preds = gathered_preds + \
                            y_pred.view(-1).tolist()
                        if phase == 'TRAIN':
                            loss.backward()
                            optimizer.step()

                preds_to_save[phase][epoch] = gathered_preds
                labels_to_save[phase][epoch] = gathered_labels

                phase_loss /= len(self.dataloaders[phase])
                print(
                    f'{phase} LOSS {phase_loss}, r2 {self.get_statistics(gathered_labels, gathered_preds)}')

            if epoch % self.snapshot_frequency == 0:
                self.save_model()
                print('Snapshot successfully created!')
        save_dict = {'labels': labels_to_save,
                     'predictions': preds_to_save}
        self.save_model()
        np.save(os.path.join(self.experiment_dirpath,
                             'train_labels_and_preds.npy'), save_dict)

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
            f'TEST LOSS {phase_loss}, r2 {self.get_statistics(gathered_labels, gathered_preds)}')

    def save_model(self):
        os.makedirs(self.experiment_dirpath, exist_ok=True)
        torch.save(self.net, os.path.join(
            self.experiment_dirpath, 'model.pth'))

    def get_statistics(self, gathered_labels, gathered_preds):
        r2 = r2_score(gathered_labels, gathered_preds)
        return r2
