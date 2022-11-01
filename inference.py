import argparse
from models.arcnn import ARCNN
from models.lstm import CLSTM, CNN
import torch
from src.predictor import Predictor
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset directory
    parser.add_argument('-a', '--audio-file-path', type=str, 
                        default='data/Q03.wav',
                        help='audio file to training data')
    
    parser.add_argument('--save-file-path', type=str, 
                        default=None,
                        help='audio file to training data')
    
    parser.add_argument('--model-file-path', type=str, 
                        default='trained_models/RCNN2/model.pt')
    
    parser.add_argument('-k', type=int, default=20,
                        help='Num mixed readers')
    
    parser.add_argument('--cpu', action='store_true', default=True,
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
    #     input_shape=(data_config['timeseries_length'], 128),
    #     hidden_size=512,
    #     num_layers=2,
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

    predictor = Predictor(model, device=device)
    predictor.load_model_from_path(args.model_file_path)
    prob_out, labels = predictor.predict(args.audio_file_path, data_config,
        k=88, plot=True, 
        question_id=0,
        save_path=args.save_file_path)
    print(labels[prob_out.argsort()[::-1]][:args.k])
    
if __name__ == "__main__":
    main()