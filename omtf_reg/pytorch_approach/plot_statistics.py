import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import argparse
import seaborn as sns

from tqdm import tqdm

from sklearn.metrics import r2_score

from omtf_reg.pytorch_approach.constants import pt_intervals


class omtfPlotter:

    def __init__(self, experiment_dirpath: str, original_data_path: str):
        self.experiment_dirpath = experiment_dirpath
        self.original_data_path = original_data_path
        self._extract_predictions_labels_and_test_data()

    def _extract_predictions_labels_and_test_data(self):
        prediction_paths = glob.glob(os.path.join(
            self.experiment_dirpath, 'predictions', '*'))
        label_paths = glob.glob(os.path.join(
            self.experiment_dirpath, 'labels', '*'))
        prediction_paths.sort(key=lambda x:
                              int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        label_paths.sort(key=lambda x:
                         int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

        train_stats = []
        valid_stats = []
        for i, (label_path, pred_path) in tqdm(enumerate(zip(label_paths, prediction_paths)), desc='Extracting data from npz files'):
            labels = np.load(label_path)['data']
            preds = np.load(pred_path)['data']
            if 'TRAIN' in label_path:
                train_stats.append((labels, preds))
            else:
                valid_stats.append((labels, preds))

        test_data = np.load(os.path.join(self.experiment_dirpath, 'test',
                                         'labels_and_preds.npz'), allow_pickle=True)['data'][()]
        test_npz = np.load(self.original_data_path, allow_pickle=True)['TEST'][()]

        with open(os.path.join(self.experiment_dirpath, 'losses.json'), 'r') as f:
            losses_dict = json.load(f)

        self.train_stats = train_stats
        self.valid_stats = valid_stats
        self.test_data = test_data
        self.test_npz = test_npz
        self.losses_dict = losses_dict

    def plot_stats(self, train_data, valid_data, title: str = '', xlabel: str = 'epochs',
                   ylabel: str = '', figsize: tuple = (8, 4), outpath=None):
        plt.figure(figsize=figsize)
        epochs = len(train_data)
        plt.plot(range(epochs), train_data, label='TRAIN', color='b')
        plt.plot(range(epochs), valid_data, label='VALID', color='y')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        if outpath is not None:
            plt.savefig(outpath)
        else:
            plt.show()
        plt.close()

    @staticmethod
    def get_pull(labels, preds):
        return np.mean((preds - labels) / labels)

    def draw_pull(self, figsize: tuple = (8, 4), outpath=None):
        pull_train = [self.get_pull(*s) for s in self.train_stats]
        pull_valid = [self.get_pull(*s) for s in self.valid_stats]
        kw = {'figsize': figsize,
              'title': f'Pull for training for {len(pull_train)} epochs',
              'xlabel': 'Epochs',
              'ylabel': 'Pull',
              'outpath': outpath}
        self.plot_stats(pull_train, pull_valid, **kw)

    def draw_losses(self, figsize: tuple = (8, 4), outpath=None):
        train_losses = list(self.losses_dict['loss']['TRAIN'].values())
        valid_losses = list(self.losses_dict['loss']['VALID'].values())

        kw = {'figsize': figsize,
              'title': f'Loss for training for {len(train_losses)} epochs',
              'xlabel': 'Epochs',
              'ylabel': 'Loss function',
              'outpath': outpath}
        self.plot_stats(train_losses, valid_losses, **kw)

    def draw_r2_scores(self, figsize: tuple = (8, 4), outpath=None):
        r2_train = [r2_score(*s) for s in self.train_stats]
        r2_valid = [r2_score(*s) for s in self.valid_stats]

        kw = {'figsize': figsize,
              'title': f'$r^2$ score for training for {len(r2_train)} epochs',
              'xlabel': 'Epochs',
              'ylabel': '$r^2$ score',
              'outpath': outpath}
        self.plot_stats(r2_train, r2_valid, **kw)

    def draw_effectivness_curve(self, pt_code_cut: float, pt_intervals: list = pt_intervals, figsize: tuple = (16, 8), outpath=None, mask_type='none'):
        labels = self.test_data['labels']
        preds = self.test_data['predictions']
        labels_omtf = self.test_npz['PT_CODE']
        preds_omtf = self.test_npz['OMTF_PT']
        omtf_quality = self.test_npz['OMTF_QUALITY']

        mask_omtf_upper = self.test_npz['PT_VAL'] < 100
        preds_omtf = preds_omtf[mask_omtf_upper]
        labels_omtf = labels_omtf[mask_omtf_upper]

        mask_omtf_pt = preds_omtf > 0
        mask_omtf_quality = omtf_quality[mask_omtf_upper] == 12

        mask_reg_pt = preds > 0
        mask_reg_quality = omtf_quality[mask_omtf_upper] == 12

        mask_dict = {
            'none': (np.full(preds.shape, True), np.full(preds_omtf.shape, True)),
            'quality_12': (mask_reg_quality, mask_omtf_quality),
            'pt': (mask_reg_pt, mask_omtf_pt),
            'full': (mask_reg_pt & mask_reg_quality, mask_omtf_quality & mask_omtf_pt)
        }

        preds = preds[mask_dict[mask_type][0]]
        labels = labels[mask_dict[mask_type][0]]

        preds_omtf = preds_omtf[mask_dict[mask_type][1]]
        labels_omtf = labels_omtf[mask_dict[mask_type][1]]

        # nn data
        pt_cut = pt_intervals[pt_code_cut - 1]
        h2 = np.histogram(labels[preds > pt_cut], pt_intervals)[0]
        h1 = np.histogram(labels, pt_intervals)[0]

        h2_omtf = np.histogram(labels_omtf[preds_omtf > pt_code_cut], bins=range(
            1, len(pt_intervals) + 1))[0]
        h1_omtf = np.histogram(
            labels_omtf, bins=range(1, len(pt_intervals) + 1))[0]

        eff_reg = np.nan_to_num(
            h2 / h1) if np.any(np.isnan(h2 / h1)) else h2 / h1
        eff_omtf = np.nan_to_num(
            h2_omtf / h1_omtf) if np.any(np.isnan(h2_omtf / h1_omtf)) else h2_omtf / h1_omtf

        sns.set()
        plt.figure(figsize=figsize)
        plt.plot(eff_reg, 'bo-', label='reg')
        plt.plot(eff_omtf, 'yo-', label='omtf')
        plt.axvline(pt_code_cut - 1, color='r')
        plt.xticks(range(h2.shape[0]), range(1, h2.shape[0] + 1))
        plt.title(
            f'Effectivness curve for $p_T \geq {pt_cut}$ GeV\nCut: code {pt_code_cut}, $p_T \in [{pt_cut}, {pt_intervals[pt_code_cut]}[$ GeV')
        plt.ylabel('Effectivness')
        plt.xlabel('Momentum code')
        plt.legend()
        plt.tight_layout()
        if outpath is not None:
            plt.savefig(outpath)
        else:
            plt.show()
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plotting tool for omtf_reg')
    parser.add_argument('--experiment_dirpath',
                        help='Location of predictions', required=True, type=str)
    parser.add_argument('--original_data_path',
                        help='Location of original data ', required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plots_path = os.path.join(args.experiment_dirpath, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    with open(os.path.join(args.experiment_dirpath, 'training_params.json'), 'r') as f:
        training_params = json.load(f)
    is_inverse = True if 'Inverse' in training_params['dataset_type'] else False
    plotter = omtfPlotter(args.experiment_dirpath, args.original_data_path)
    plotter.draw_losses(outpath=os.path.join(plots_path, 'losses.pdf'))
    plotter.draw_pull(outpath=os.path.join(plots_path, 'pulls.pdf'))
    plotter.draw_r2_scores(outpath=os.path.join(plots_path, 'r2_scores.pdf'))

    for cut in range(len(pt_intervals)):
        plotter.draw_effectivness_curve(pt_code_cut=cut, outpath=os.path.join(
            plots_path, f'effectivness_curve_cut_{pt_intervals[cut]}.pdf'), mask_type='full')


if __name__ == '__main__':
    main()
