import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import argparse

from tqdm import tqdm

from sklearn.metrics import r2_score

from omtf_reg.pytorch_approach.constants import pt_intervals


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plotting tool for omtf_reg')
    parser.add_argument('--experiment_dirpath',
                        help='Where to save the data', required=True, type=str)
    args = parser.parse_args()
    return args


def extract_predictions_and_labels(experiment_dirpath: str, fn_to_apply):
    prediction_paths = glob.glob(os.path.join(
        experiment_dirpath, 'predictions', '*'))
    label_paths = glob.glob(os.path.join(
        experiment_dirpath, 'labels', '*'))
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
            train_stats.append(fn_to_apply(labels, preds))
        else:
            valid_stats.append(fn_to_apply(labels, preds))
    return train_stats, valid_stats


def get_pull(labels, preds):
    return np.mean((preds - labels) / labels)





def draw_effectivness_curve(test_npz: dict, test_data: dict, pt_intervals: list, pt_code_cut: float, figsize: tuple = (16, 8), outpath=None):
    labels = test_data['labels']
    preds = test_data['predictions']
    labels_omtf = test_npz['PT_CODE']
    preds_omtf = test_npz['OMTF_PT']
    omtf_quality = test_npz['OMTF_QUALITY']

    #nn data
    pt_cut = pt_intervals[pt_code_cut-1]
    h2 = np.histogram(labels[preds > pt_cut], pt_intervals)[0]
    h1 = np.histogram(labels, pt_intervals)[0]

    mask = test_npz['PT_VAL'] < 100
    mask_omtf_pt = preds_omtf>0
    mask_omtf_quality = omtf_quality==12
    preds_omtf = preds_omtf[mask_omtf_quality & mask_omtf_pt]
    labels_omtf = labels_omtf[mask_omtf_quality & mask_omtf_pt]


    h2_omtf = np.histogram(labels_omtf[preds_omtf > pt_code_cut], bins=range(1, len(pt_intervals)+1))[0]
    h1_omtf = np.histogram(labels_omtf, bins=range(1, len(pt_intervals) + 1))[0]

    eff_reg = np.nan_to_num(h2/h1) if np.any(np.isnan(h2/h1)) else h2/h1
    eff_omtf = np.nan_to_num(h2_omtf/h1_omtf) if np.any(np.isnan(h2_omtf/h1_omtf)) else h2_omtf/h1_omtf


    sns.set()
    plt.figure(figsize=figsize)
    plt.plot(eff_reg, 'bo-', label='reg')
    plt.plot(eff_omtf, 'yo-', label='omtf')
    plt.axvline(pt_code_cut - 1, color='r')
    plt.xticks(range(h2.shape[0]), range(1, h2.shape[0] +1))
    plt.title(f'Effectivness curve for $p_T \geq {pt_cut}$ GeV\nCut: code {pt_code_cut}, $p_T \in [{pt_cut}, {pt_intervals[pt_code_cut]}[$ GeV')
    plt.ylabel('Effectivness')
    plt.xlabel('Momentum code')
    plt.legend()
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.close()


def draw_losses(json_data: dict, figsize: tuple = (8, 4), outpath=None):
    train_losses = list(json_data['loss']['TRAIN'].values())
    # valid_losses = list(json_data['loss']['VALID'].values())
    plt.figure(figsize=figsize)
    epochs = len(train_losses)
    plt.plot(range(epochs), train_losses, label='TRAIN', color='b')
    # plt.plot(range(epochs), valid_losses, label='VALID', color='y')
    plt.title(f'Loss for training for {epochs} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.close()


def draw_r2_scores(experiment_dirpath: str, figsize: tuple = (8, 4), outpath=None):
    r2_train, r2_valid = extract_predictions_and_labels(
        experiment_dirpath, fn_to_apply=r2_score)
    plt.figure(figsize=figsize)
    epochs = len(r2_train)
    plt.plot(range(epochs), r2_train, label='TRAIN', color='b')
    # plt.plot(range(epochs), r2_valid, label='VALID', color='y')
    plt.title(f'$r^2$ score for training for {epochs} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('r2_score')
    plt.legend()
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.close()


def draw_pull(experiment_dirpath: str, figsize: tuple = (8, 4), outpath=None):
    pull_train, pull_valid = extract_predictions_and_labels(
        experiment_dirpath, fn_to_apply=get_pull)
    plt.figure(figsize=figsize)
    epochs = len(pull_train)
    plt.plot(range(epochs), pull_train, label='TRAIN', color='b')
    # plt.plot(range(epochs), pull_valid, label='VALID', color='y')
    plt.title(f'Pull for training for {epochs} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Pull')
    plt.legend()
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    plots_path = os.path.join(args.experiment_dirpath, 'plots')
    os.makedirs(plots_path, exist_ok=True)
    with open(os.path.join(args.experiment_dirpath, 'training_params.json'), 'r') as f:
        training_params = json.load(f)
    is_inverse = True if 'Inverse' in training_params['dataset_type'] else False
    with open(os.path.join(args.experiment_dirpath, 'losses.json'), 'r') as f:
        json_data = json.load(f)
    draw_losses(json_data, outpath=os.path.join(plots_path, 'losses.pdf'))
    draw_r2_scores(args.experiment_dirpath, outpath=os.path.join(plots_path, 'r2_scores.pdf'))
    draw_pull(args.experiment_dirpath, outpath=os.path.join(plots_path, 'pulls.pdf'))
    test_data = np.load(os.path.join(args.experiment_dirpath, 'test', 'labels_and_preds.npz'), allow_pickle=True)['data'][()]
    for cut in pt_intervals:
        draw_effectivness_curve(test_data, pt_intervals, cut=cut, outpath=os.path.join(plots_path, f'effectivness_curve_cut_{cut}.pdf'), is_inverse=is_inverse)

if __name__ == '__main__':
    main()
