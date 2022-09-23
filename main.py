import argparse
import numpy as np
from models.VGG import VGG
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
    parser.add_argument('--label-train-path', type=str, 
                        default='input/DGV4VN/DGV4VN_1015.HISAT_result.csv',
                        help='path to label file')
    parser.add_argument('--label-val-path', type=str, 
                        default='input/DGV4VN/DGV4VN_1015.HISAT_result.csv',
                        help='path to label file')
    parser.add_argument('--label-test-path', type=str, 
                        default='input/DGV4VN/DGV4VN_1015.HISAT_result.csv',
                        help='path to label file')
    parser.add_argument('--sample-train-path', type=str,
                        default='input/consensus23/consensus23.phased.HLA.sample.list',
                        help='Sample list for training data')
    parser.add_argument('--sample-val-path', type=str,
                        default='input/GSAv4_27/Batch1.sample_name.list',
                        help='Sample list for validation data')
    parser.add_argument('--sample-test-path', type=str,
                        default='input/GSAv4_27/Batch1.sample_name.list',
                        help='Sample list for test data')
    parser.add_argument('--ref-position-path', type=str,
                        default='input/GSAv3_24/GSAv3_24.GRCh38.rename.HLAregion.position.list',
                        help='Only using position in this file for training')
    
    parser.add_argument('--entropy-threshold', type=float, default=None,
                        help='threshold to accept for extract single column data')
    
    parser.add_argument('--accept-threshold', type=float, default=0.01,
                        help='threshold to accept a 2nd predicted label')
    
    parser.add_argument('--output-path', type=str, default='output/Training_Results/1KVG_GSAv4_27')
    parser.add_argument('--model-path', type=str, default='/home/hienvq/Desktop/AI/KarutaAI/trained_models/vae.pt')
    parser.add_argument('-d', '--model-save-dir', type=str, 
                        default='trainned_models/1KVG_GSAv4_27/',
                        help='directory to save model')
    
    parser.add_argument('-g', '--group', type=str, default='2',
                        help='HLA groups to use, include: [HLA-A], [HLA-B, HLA-C], \
                            [HLA-DRB1, HLA-DQA1, HLA-DQB1], [DPB1]')
    
    parser.add_argument('--use-cross-validation', action='store_true')
    parser.add_argument('--not-use-collapse', action='store_true')
    
    # Model hyperparameters
    parser.add_argument('-l', '--loss', type=str, default='bce',
                        help='loss function to use')
    parser.add_argument('-o', '--optimizer', type=str, default='nadam',
                        help='optimizer to use')
    parser.add_argument('-k', '--k-fold', type=int, default=10,
                        help='number of folds to use for cross validation')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005,  
                        help='learning rate')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=233,
                        help='seed for training')
    parser.add_argument('--split-ratio', type=float, default=0.9,
                        help='ratio of training data')

    parser.add_argument('--n-digits', type=int, default=2,
                        help='number of digits to be considered, etc: 10:12:13:14 -> 10:12')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show validation results')
    
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    trainer = Trainer(model=VGG('VGG7'), 
                      lr=0.003, 
                      loss='bce', 
                      optimizer='adas', 
                      batch_size=256, 
                      n_repeats=1,
                      save_every=500
                      )
    trainer.load_extract_model(args.model_path)
    
    train_folder = args.train_folder
    val_folder = args.val_folder
    test_folder = args.test_folder
    trainer.load_data(train_folder, val_folder, test_folder)
    trainer.fit()
    
if __name__ == "__main__":
    main()