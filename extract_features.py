import argparse
import os
import numpy as np
from src.data_helper import load_data, preprocess
from models.lstm import CLSTM
import torch
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder', type=str,
                        default='generated_data/train',
                        help='path to training data')
    
    parser.add_argument('--val-folder', type=str,
                        default='generated_data/val',
                        
                        help='path to training data')
    parser.add_argument('--test-folder', type=str,
                        default='generated_data/test',
                        help='path to training data')
    
    parser.add_argument('--original-label-data-path', type=str,
                        default='data/JKspeech/',
                        help='')
    
    parser.add_argument('--processed-data-path', type=str,
                        default='tmp',
                        help='')
    
    parser.add_argument('--model-path', type=str, default='./trainned_models"')
    
    parser.add_argument('-d', '--model-save-dir', type=str, 
                        default='trainned_models/',
                        help='directory to save model')
    
    # Model hyperparameters
    parser.add_argument('-l', '--loss', type=str, default='bce',
                        help='loss function to use')
    
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='optimizer to use')
    
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='number of epochs to train')
    
    parser.add_argument('--lr', type=float, default=0.0001,  
                        help='learning rate')
    
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size')
    
    parser.add_argument('--seed', type=int, default=233,
                        help='seed for training')
    
    parser.add_argument('--load-model', action='store_true',
                        help='Retrainging with old model')
    
    parser.add_argument('--preprocess', action='store_true', # default=True,
                        help='Show validation results')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show validation results')
    
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    
    
    data_config = {
        'batch-length': 32768,
        'num_mfcc': 64,
        'num_chroma': 64,
        'n_fft': 2048,
        'hop_length': 512,
        'timeseries_length': 469,
        'sr': 48000
    }
    
    preprocess(args.val_folder, 
                os.path.join(args.processed_data_path, 'val'), data_config)
    preprocess(args.train_folder, 
                os.path.join(args.processed_data_path, 'train'), data_config)
    preprocess(args.test_folder, 
                os.path.join(args.processed_data_path, 'test'), data_config)
    
if __name__ == "__main__":
    main()