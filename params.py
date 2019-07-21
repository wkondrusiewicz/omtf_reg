import argparse
import json
import os

def create_parser():
    parser = argparse.ArgumentParser(
        description='Simple attempt to use regression for OMTF')

    parser.add_argument('-e', '--epochs', help="Number of epochs",
                        type=int, default=10, required=True)

    parser.add_argument('-b', '--batch_size', help="Batch size",
                        type=int, default=10, required=True)
    parser.add_argument('-c', '--create_basic_info', help='Create basic info file', required=False, type=bool, default=False)
    parser.add_argument('-t', '--thresh', help='Data threshold', required=False, type=int, default=None)
    parser.add_argument('--path_model', help='Where to save the model', required=True, type=str)
    parser.add_argument('--path_data', help='Path to data', required=True, type=str)

    args = parser.parse_args()
    if args.create_basic_info:
        data = dict(zip(["epochs", "batch_size", "thresh", "path_model","path_data"],[args.epochs, args.batch_size, args.thresh, args.path_model, args.path_data]))
        create_json(data=data)

    return args.epochs, args.batch_size, args.thresh, args.path_model, args.path_data

def create_json(data):
    with open('basic_info.json', 'w') as f:
        json.dump(data, f)
