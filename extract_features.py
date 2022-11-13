import argparse
import os
from src.data_helper import preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder', type=str,
                        default='generated_data/max15/train',
                        help='path to training data')

    parser.add_argument('--val-folder', type=str,
                        default='generated_data/max20/val',

                        help='path to training data')
    parser.add_argument('--test-folder', type=str,
                        default='generated_data/max20/test',
                        help='path to training data')

    parser.add_argument('--original-label-data-path', type=str,
                        default='data/JKspeech/',
                        help='')

    parser.add_argument('--processed-data-path', type=str,
                        default='transformed/128_max20/',
                        help='')

    parser.add_argument('--seed', type=int, default=233,
                        help='seed for training')

    parser.add_argument('--load-model', action='store_true',
                        help='Retrainging with old model')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show validation results')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data_config = {
        'batch-length': 32768,
        'num_mfcc': 39,
        'num_chroma': 64,
        'n_fft': 2048,
        'hop_length': 128,
        'sr': 48000,
        'fixed-time': 1.0
    }
    data_config['timeseries_length'] = 2 + int(\
        (data_config['fixed-time'] * data_config['sr'] - 1) // data_config['hop_length'])

    preprocess(args.val_folder, os.path.join(
        args.processed_data_path, 'val'), data_config)
    preprocess(args.train_folder, os.path.join(
        args.processed_data_path, 'train'), data_config)
    preprocess(args.test_folder, os.path.join(
        args.processed_data_path, 'test'), data_config)


if __name__ == "__main__":
    main()
