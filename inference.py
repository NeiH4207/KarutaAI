import argparse
from models.arcnn import ARCNN
from models.lstm import CLSTM, CNN
import torch
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset directory
    parser.add_argument('--audio-file-path', type=str, 
                        default='data/Q03.wav',
                        help='audio file to training data')
    
    parser.add_argument('--model-file-path', type=str, default='trainned_models/RCNN/model.pt')
    
    parser.add_argument('-d', '--model-save-dir', type=str, 
                        default='trained_models/',
                        help='directory to save model')
    
    parser.add_argument('-k', type=int, default=3,
                        help='Num mixed readers')
    
    parser.add_argument('--cpu', action='store_true',
                        help='Use cpu cores instead')
    
    parser.add_argument('--plot', action='store_true',
                        help='Plot probability distribution')
    
    
    args = parser.parse_args() 
    return args

def main():
    args = parse_args()
    if args.cpu:
        device = 'cpu'
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_config = {
        'num_mfcc': 39,
        'num_chroma': 64,
        'n_fft': 2048,
        'hop_length': 512,
        'sr': 48000,
        'fixed-time': 2.5
    }
    
    data_config['timeseries_length'] = int(1 + \
        (data_config['fixed-time'] * data_config['sr'] - 1) // data_config['hop_length'])

    
    # model = CLSTM(
    #     input_size=136,
    #     hidden_size=512,
    #     num_layers=2,
    #     num_classes=88, 
    #     device=device
    # )
    
    # model = CNN(
    #     input_size=281,
    #     embbed_size=1024,
    #     num_classes=88, 
    #     device=device
    # )
    
    model = ARCNN(
        input_shape=(data_config['timeseries_length'], 128),
        num_chunks= 4,
        in_channels=1,
        rnn_hidden_size=512,
        rnn_num_layers=2,
        num_classes=88, 
        device=device
    )
    # audio_file_path = 'data/sample_Q_202205/sample_Q_M01/problem1.wav'
    trainer = Trainer(model, save_dir=args.model_save_dir, 
                      save_name="model.pt", device=device, verbose=True)
    trainer.load_model_from_path(args.model_file_path)
    trainer.test(args.audio_file_path, 
                 data_config=data_config, 
                 k=args.k, plot=args.plot)
    
if __name__ == "__main__":
    main()