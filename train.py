import argparse
import os
import numpy as np
from src.data_helper import load_data, preprocess
from models.lstm import CLSTM, CNN
from models.rcnn import RCNN
from models.arcnn import ARCNN
import torch
from src.trainer import Trainer
from src.utils import gather_files_from_folder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetpath', type=str,
                        default='transformed/max10/')
    parser.add_argument('--model-path', type=str, default='./trained_models"')

    parser.add_argument('-d', '--model-save-dir', type=str,
                        default='trained_models/CNN/',
                        help='directory to save model')

    # Model hyper-parameters
    parser.add_argument('-l', '--loss', type=str, default='bce',
                        help='loss function to use')

    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='optimizer to use')

    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size')

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
    x_val, y_val = load_data(os.path.join(
        args.datasetpath, 'val', 'dataset_batch_0.pickle'))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = CLSTM(
    #     input_size=x_val[0].shape[1],
    #     hidden_size=512,
    #     num_layers=2,
    #     num_classes=88,
    #     device=device
    # )
    
    # model = CNN(
    #     input_size=x_val[0].shape[1],
    #     embed_size=1024,
    #     num_classes=88,
    #     device=device
    # )
    model = ARCNN(
        input_shape=x_val[0].shape,
        num_chunks= 4,
        in_channels=1,
        rnn_hidden_size=512,
        rnn_num_layers=2,
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
        trainer.load_model_from_path(
            os.path.join(args.model_save_dir, "model.pt"))

    batch_names = gather_files_from_folder(os.path.join(
        args.datasetpath, 'train'), _extension='.pickle')

    for _ in range(1000):
        for trainpath in batch_names:
            x_train, y_train = load_data(trainpath)
            trainer.set_data(x_train, y_train, x_val, y_val)
            trainer.train(optimizer=args.optimizer,
                          training_params=training_params, )


if __name__ == "__main__":
    main()
