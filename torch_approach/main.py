import argparse

from torch.utils.data import DataLoader

from dataset import omtfDataset
from net import omtfNet
from model import omtfModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use regression for OMTF')

    parser.add_argument('-e', '--epochs', help="Number of epochs",
                        type=int, default=10)
    parser.add_argument('--init_lr', help='Initial learning_rate', type=float, default=0.001)
    parser.add_argument('-trb', '--train_batch_size',
                        help="Trian batch size", type=int, default=256)
    parser.add_argument('-teb', '--test_batch_size',
                        help="Test batch size", type=int, default=32)
    parser.add_argument('--experiment_dirpath',
                        help='Where to save the model', required=True, type=str)
    parser.add_argument(
        '--data_path', help='Path to data', required=True, type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataloaders = {'TRAIN': DataLoader(omtfDataset(data_path=args.data_path, mode='TRAIN'), batch_size=args.train_batch_size, shuffle=True),
                    'VALID': DataLoader(omtfDataset(data_path=args.data_path, mode='VALID'), batch_size=args.test_batch_size),
                    'TEST': DataLoader(omtfDataset(data_path=args.data_path, mode='TEST'), batch_size=args.test_batch_size)}
    model = omtfModel(dataloaders=dataloaders, experiment_dirpath=args.experiment_dirpath)
    model.train(epochs=args.epochs, init_lr=args.init_lr)
    model.predict()

if __name__ == '__main__':
    main()
