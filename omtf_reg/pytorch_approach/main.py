import argparse
import os
import json

from torch.utils.data import DataLoader

from omtf_reg.pytorch_approach.datasets import omtfDataset, omtfDatasetInverse, omtfDatasetMasked
from omtf_reg.pytorch_approach.net import omtfNet, omtfNetBig, omtfNetBigger, omtfResNet, omtfResNetBig, omtfHalfResNet, omtfNetNotSoDense
from omtf_reg.pytorch_approach.model import omtfModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use regression for OMTF')

    parser.add_argument('-e', '--epochs', help="Number of epochs",
                        type=int, default=10)
    parser.add_argument(
        '--init_lr', help='Initial learning_rate', type=float, default=0.001)
    parser.add_argument('-trb', '--train_batch_size',
                        help="Trian batch size", type=int, default=4096)
    parser.add_argument('-teb', '--test_batch_size',
                        help="Test batch size", type=int, default=256)
    parser.add_argument('--experiment_dirpath',
                        help='Where to save the model', required=True, type=str)
    parser.add_argument('--pretrained_model_path', help='Path to pretrained model',
                        type=str, default=None)
    parser.add_argument(
        '--data_path', help='Path to data', required=True, type=str)

    parser.add_argument(
        '--net', choices=['omtfNet', 'omtfNetBig', 'omtfNetBigger', 'omtfResNet', 'omtfResNetBig', "omtfHalfResNet", 'omtfNetNotSoDense'], default='omtfNet')
    parser.add_argument(
        '--dataset_type', choices=['omtfDataset', 'omtfDatasetInverse', 'omtfDatasetMasked'], default='omtfDataset')
    parser.add_argument('--mask_path', default=None)
    parser.add_argument('-lrd', '--lr_decay_rate',
                        default=None, type=float)
    parser.add_argument('-wd', '--weight_decay',
                        default=0.0, type=float)

    args = parser.parse_args()
    return args


def get_net_architecture(name):
    return {'omtfNet': omtfNet, 'omtfNetBig': omtfNetBig, 'omtfNetBigger': omtfNetBigger, 'omtfResNet': omtfResNet, 'omtfResNetBig': omtfResNetBig, "omtfHalfResNet": omtfHalfResNet, "omtfNetNotSoDense": omtfNetNotSoDense}[name]


def get_dataset_architecture(name):
    return {'omtfDataset': omtfDataset, 'omtfDatasetInverse': omtfDatasetInverse, 'omtfDatasetMasked': omtfDatasetMasked}[name]


def main():
    args = parse_args()
    kw = {'mask_path': args.mask_path} if args.mask_path is not None else {}
    dataloaders = {'TRAIN': DataLoader(get_dataset_architecture(args.dataset_type)(data_path=args.data_path, mode='TRAIN', **kw),
                                       batch_size=args.train_batch_size, shuffle=True),
                   'VALID': DataLoader(get_dataset_architecture(args.dataset_type)(data_path=args.data_path, mode='VALID', **kw),
                                       batch_size=args.test_batch_size),
                   'TEST': DataLoader(get_dataset_architecture(args.dataset_type)(data_path=args.data_path, mode='TEST', **kw),
                                      batch_size=args.test_batch_size)}

    net = get_net_architecture(args.net)()
    model = omtfModel(dataloaders=dataloaders,
                      experiment_dirpath=args.experiment_dirpath, net=net, init_lr=args.init_lr, weight_decay=args.weight_decay, lr_decay_rate=args.lr_decay_rate)
    if args.pretrained_model_path is not None:
        args.pretrained_model_path = os.path.abspath(args.pretrained_model_path)
        model.load_model(args.pretrained_model_path)

    model.train(epochs=args.epochs)
    model.predict()

    training_params = {'epochs': args.epochs,
                       'init_lr': args.init_lr,
                       'train_batch_size': args.train_batch_size,
                       'test_batch_size': args.test_batch_size,
                       'data_path': os.path.abspath(args.data_path), 'dataset_type': args.dataset_type,
                       'weight_decay': args.weight_decay,
                       'lr_decay_rate': args.lr_decay_rate,
                       'net': args.net,
                       'pretrained_model_path': args.pretrained_model_path or None}

    with open(os.path.join(args.experiment_dirpath, 'training_params.json'), 'w') as f:
        json.dump(training_params, f)


if __name__ == '__main__':
    main()
