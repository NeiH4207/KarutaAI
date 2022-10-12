import argparse
import os
import numpy as np
from src.data_helper import load_data, preprocess
from models.lstm import CLSTM, LSTM
import torch
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dataset', action='store_true',  default=True,
                        help='load_dataset model, preprocess data')
    
    parser.add_argument('--training', action='store_true',  default=True,
                        help='training model')
    # Dataset directory
    parser.add_argument('--train-folder', type=str,
                        default='/home/hienvq/Desktop/AI/KarutaAI/generated_data/train',
                        help='path to training data')
    
    parser.add_argument('--val-folder', type=str,
                        default='/home/hienvq/Desktop/AI/KarutaAI/generated_data/val',
                        
                        help='path to training data')
    parser.add_argument('--test-folder', type=str,
                        default='/home/hienvq/Desktop/AI/KarutaAI/generated_data/test',
                        help='path to training data')
    
    parser.add_argument('--processed-data-path', type=str,
                        default='/home/hienvq/Desktop/AI/KarutaAI/tmp',
                        help='')
    
    parser.add_argument('--accept-threshold', type=float, default=0.01,
                        help='threshold to accept a 2nd predicted label')
    
    parser.add_argument('--model-file-path', type=str, default='./trainned_models/model.pt')
    
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
    
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size')
    
    parser.add_argument('--seed', type=int, default=233,
                        help='seed for training')
    
    parser.add_argument('--preprocess', action='store_true',
                        help='Show validation results')
    
    parser.add_argument('--cpu', action='store_true',
                        help='Use cpu cores instead')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show validation results')
    
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    if args.cpu:
        device = 'cpu'
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data_config = {
        'num_mfcc': 13,
        'n_fft': 2048,
        'hop_length': 512,
        'timeseries_length': 128
    }
    
    model = CLSTM(
        input_size=33,
        hidden_size=256,
        num_layers=2,
        num_classes=88, 
        device=device
    )
    audio_file_path = 'generated_data/train/data/ej_combined_sample_337.wav'
    # audio_file_path = 'data/Q_12.wav'
    audio_file_path = 'data/Q_12.wav'
    audio_label_path = 'generated_data/train/label/ej_combined_sample_337.txt'
    trainer = Trainer(model, save_dir=args.model_save_dir, 
                      save_name="model.pt", device=device, verbose=True)
    trainer.load_model_from_path(args.model_file_path)
    trainer.test(audio_file_path, audio_label_path, data_config)
    
if __name__ == "__main__":
    main()