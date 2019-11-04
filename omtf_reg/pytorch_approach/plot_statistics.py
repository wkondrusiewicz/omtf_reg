import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import argparse

from sklearn.metrics import r2_score


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
    for i, (label_path, pred_path) in enumerate(zip(label_paths, prediction_paths)):
        labels = np.load(label_path)['data']
        preds = np.load(pred_path)['data']
        print(i)
        if 'TRAIN' in label_path:
            train_stats.append(fn_to_apply(labels, preds))
        else:
            valid_stats.append(fn_to_apply(labels, preds))
    return train_stats, valid_stats


def get_pull(labels, preds):
    return np.mean((preds - labels) / labels)


def draw_effectivness_curve(test_data: dict, pt_intervals: list, cut: int, figsize: tuple = (8, 4), outpath=None):
    labels = test_data['labels']
    preds = test_data['predictions']
    h2 = np.histogram(labels[preds > cut], pt_intervals)[0]
    h1 = np.histogram(labels, pt_intervals)[0]
    plt.figure(figsize=figsize)
    plt.plot(h2 / h1)
    plt.title(f'Effectivness curve for test data')
#     plt.xlabel('Epochs')
    plt.ylabel('Effectivness')
    plt.legend()
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.close()


def draw_losses(json_data: dict, figsize: tuple = (8, 4), outpath=None):
    train_losses = list(json_data['loss']['TRAIN'].values())
    valid_losses = list(json_data['loss']['VALID'].values())
    plt.figure(figsize=figsize)
    epochs = len(train_losses)
    plt.plot(range(epochs), train_losses, label='TRAIN', color='b')
    plt.plot(range(epochs), valid_losses, label='VALID', color='y')
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
    plt.plot(range(epochs), r2_valid, label='VALID', color='y')
    plt.title(f'Loss for training for {epochs} epochs')
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
    plt.plot(range(epochs), pull_valid, label='VALID', color='y')
    plt.title(f'Loss for training for {epochs} epochs')
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


if __name__ == '__main__':
    main()
