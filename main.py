import argparse
import os
import numpy as np
from src.data_helper import load_data, preprocess, preprocess2, preprocess3
from models.lstm import CLSTM, CLSTM2, LSTM
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
    
    parser.add_argument('--original-label-data-path', type=str,
                        default='/home/hienvq/Desktop/AI/KarutaAI/data/JKspeech/',
                        help='')
    
    parser.add_argument('--processed-data-path', type=str,
                        default='/home/hienvq/Desktop/AI/KarutaAI/tmp',
                        help='')
    
    parser.add_argument('--accept-threshold', type=float, default=0.01,
                        help='threshold to accept a 2nd predicted label')
    
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
        'num_mfcc': 64,
        'num_chroma': 64,
        'n_fft': 2048,
        'hop_length': 512,
        'timeseries_length': 216,
        'sr': 22050
    }
    
    if args.preprocess:
        preprocess(args.train_folder, 
                    os.path.join(args.processed_data_path, 'train'), data_config)
        preprocess(args.val_folder, 
                    os.path.join(args.processed_data_path, 'val'), data_config)
        preprocess(args.test_folder, 
                    os.path.join(args.processed_data_path, 'test'), data_config)
        
        
    x_train, y_train = load_data(os.path.join(args.processed_data_path, 'train'))
    x_val, y_val = load_data(os.path.join(args.processed_data_path, 'test'))
    
    print("Number of training examples: %d" % x_train.shape[0])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CLSTM(
        input_size=x_train[0].shape[1],
        hidden_size=512,
        num_layers=2,
        num_classes=88, 
        device=device
    )
    
    training_params = {
        'loss_function': args.loss,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
                    
    trainer = Trainer(model, save_dir=args.model_save_dir, 
                      save_name="model.pt", device=device, verbose=True)
    if args.load_model:
        trainer.load_model_from_path(os.path.join(args.model_save_dir, "model.pt"))
        
    trainer.set_data(x_train, y_train, x_val, y_val)
    trainer.train(optimizer=args.optimizer, training_params=training_params, )
    
if __name__ == "__main__":
    main()