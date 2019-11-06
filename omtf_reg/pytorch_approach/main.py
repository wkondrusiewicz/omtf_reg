import argparse
import os
import json

from torch.utils.data import DataLoader

from omtf_reg.pytorch_approach.dataset import omtfDataset
from omtf_reg.pytorch_approach.net import omtfNet, omtfNetBig, omtfNetBigger
from omtf_reg.pytorch_approach.model import omtfModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use regression for OMTF')

    parser.add_argument('-e', '--epochs', help="Number of epochs",
                        type=int, default=10)
    parser.add_argument(
        '--init_lr', help='Initial learning_rate', type=float, default=0.001)
    parser.add_argument('-trb', '--train_batch_size',
                        help="Trian batch size", type=int, default=256)
    parser.add_argument('-teb', '--test_batch_size',
                        help="Test batch size", type=int, default=32)
    parser.add_argument('--experiment_dirpath',
                        help='Where to save the model', required=True, type=str)
    parser.add_argument(
        '--data_path', help='Path to data', required=True, type=str)
    parser.add_argument(
        '--net', choices=['omtfNet', 'omtfNetBig', 'omtfNetBigger'], default='omtfNet')

    args = parser.parse_args()
    return args


def get_net_architecture(name):
    return {'omtfNet': omtfNet, 'omtfNetBig': omtfNetBig, 'omtfNetBigger': omtfNetBigger}[name]


def main():
    args = parse_args()
    dataloaders = {'TRAIN': DataLoader(omtfDataset(data_path=args.data_path, mode='TRAIN'), batch_size=args.train_batch_size, shuffle=True),
                   'VALID': DataLoader(omtfDataset(data_path=args.data_path, mode='VALID'), batch_size=args.test_batch_size),
                   'TEST': DataLoader(omtfDataset(data_path=args.data_path, mode='TEST'), batch_size=args.test_batch_size)}
    net = get_net_architecture(args.net)()
    model = omtfModel(dataloaders=dataloaders,
                      experiment_dirpath=args.experiment_dirpath, net=net)
    model.train(epochs=args.epochs, init_lr=args.init_lr)
    model.predict()

    training_params = {'epochs': args.epochs, 'init_lr': args.init_lr, 'train_batch_size': args.train_batch_size,
                       'test_batch_size': args.test_batch_size, 'data_path': os.path.abspath(args.data_path)}

    with open(os.path.join(args.experiment_dirpath, 'training_params.json'), 'w') as f:
        json.dump(training_params, f)


if __name__ == '__main__':
    main()
