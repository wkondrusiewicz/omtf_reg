import argparse
import json

def create_parser():
    parser = argparse.ArgumentParser(
        description='Simple attempt to use regression for OMTF')

    parser.add_argument('-e', '--epochs', help="Number of epochs",
                        type=int, default=10, required=True)

    parser.add_argument('-b', '--batch_size', help="Batch size",
                        type=int, default=10, required=True)

    parser.add_argument('-p', '--plot', help="Plot the result?",
                        type=bool, default=False, required=False)
    parser.add_argument('-s', '--save_loc', help='Saving location', type=str, required=True)
    parser.add_argument('-c', '--create_basic_info', help='Create basic info file', required=False, type=bool, default=False)
    parser.add_argument('-t', '--thresh', help='Data threshold', required=False, type=int, default=None)

    args = parser.parse_args()
    if args.create_basic_info:
        data = dict(zip(["epochs", "batch_size","plottable", "save_loc", "thresh"],[args.epochs, args.batch_size, args.plot, args.save_loc, args.thresh]))
        create_json(data=data)

    return args.epochs, args.batch_size, args.plot, args.save_loc, args.thresh

def create_json(data):
    with open('basic_info.json', 'w') as f:
        json.dump(data, f)
