import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description='Simple attempt to use regression for OMTF')

    parser.add_argument('-e', '--epochs', help="Number of epochs",
                        type=int, default=10, required=True)

    parser.add_argument('-b', '--batch_size', help="Batch size",
                        type=int, default=10, required=True)

    parser.add_argument('-p', '--plot', help="Plot the result?",
                        type=bool, default=False, required=False)
    args = parser.parse_args()
    return args.epochs, args.batch_size, args.plot
