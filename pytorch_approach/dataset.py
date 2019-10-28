import numpy as np
import torch
from torch.utils.data import Dataset


class omtfDataset(Dataset):
    def __init__(self, data_path: str, mode: str, threshold: int = 100):
        """
        Args:
            data_path (string): path do input data
            mode (string): one of `TRAIN`, `VALID`, `TEST`
            threshold (int): upper bound for transverse momentum of a muon
        """

        assert mode in ['TRAIN', 'VALID',
                        'TEST'], 'Passed invalid mode type'
        self.mode = mode
        self.threshold = threshold
        data = np.load(data_path, 'r', allow_pickle=True)[
            self.mode][()]
        input = data['HITS']
        output = data['PT_VAL']
        input = input[output < self.threshold]
        output = output[output < self.threshold]
        input = torch.from_numpy(input).float().view(-1, 1, 18, 2)
        self.input = input
        self.output = torch.from_numpy(output).float().view(-1,1)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, index):
        X = self.input[index]
        y = self.output[index]
        return X, y
