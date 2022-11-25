import argparse
from models.arcnn import ARCNN
from models.lstm import CLSTM, CLSTM2
import numpy as np
from tqdm import tqdm
import torch
import librosa
from src.predictor import Predictor
from configs.conf import data_config

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset directory
    parser.add_argument('-a', '--audio-file-path', type=str, 
                        default='data/Q_20_4_0.wav',
                        help='audio file to training data')
    
    parser.add_argument('--save-file-path', type=str, 
                        default=None,
                        help='audio file to training data')
    
    parser.add_argument('--model-file-path', type=str, 
                        default='trainned_models/LSTM5/model.pt')
    
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cpu:
        device = 'cpu'
        
    model = CLSTM2(
        input_size=128,
        hidden_size=512,
        num_layers=2,
        num_classes=88,
        device=device
    )

    predictor = Predictor(model, data_config, fixed_length=False, device=device)
    predictor.load_model_from_path(args.model_file_path)
    
    prob_out, labels = predictor.predict(args.audio_file_path, 
        plot=True, 
        return_label=True,
        save_path=None)
    orders = prob_out.argsort()[::-1][:args.k]
    
    print(str(labels[orders][:args.k]).replace('\'', '\"'))
    
    old_prob_out = prob_out
    prob_audio, sr = librosa.load(args.audio_file_path, 
                                    sr=data_config['sr'])
    
    selected_audio = []
    selected_audio_idx = []
    
    for j, audio_name in enumerate(labels[orders[:-1]]):
        file_path = 'data/JKspeech/{}.wav'.format(audio_name)
        original_audio, sr = librosa.load(file_path, 
                                      sr=data_config['sr'])
        offset = 5
        print(j, audio_name)
        for i in range(min(1, len(original_audio)//offset)):
            tmp_audio = original_audio[i*offset:i*offset+len(prob_audio)]
            if i*offset + len(prob_audio) >= len(original_audio):
                padding_size = i*offset+len(prob_audio)-len(original_audio)
                tmp_audio = np.concatenate([tmp_audio, [0] * padding_size])
            tmp_audio = prob_audio - tmp_audio
            prob_out, labels = predictor.predict_by_audio(
                tmp_audio[:48000], 
                plot=False, 
                return_label=True,
                save_path=None)
            print('{}/{}:'.format(i, len(original_audio)//offset), 
                  np.round(prob_out[orders], 2))
            if old_prob_out[orders[j]] - prob_out[orders[j]] > 0.5:
                print('Offset {} in audio {}'.format(i, audio_name))
                selected_audio.append(audio_name)
                selected_audio_idx.append(j)
                prob_audio = tmp_audio
                predictor.plot_prob(prob_out, labels)
                print(str(labels[orders]).replace('\'', '\"'))
                break
            
    prob_out, labels = predictor.predict_by_audio(
                prob_audio, 
                plot=False, 
                return_label=True,
                save_path=None)
    prob_out[np.array(selected_audio_idx)] = 0
    orders = prob_out.argsort()[::-1][:args.k]
    predictor.plot_prob(prob_out[orders], labels[orders])
    selected_audio += labels[orders[:args.k - len(selected_audio_idx)]].tolist()
    print(str(sorted(selected_audio)).replace('\'', '\"'))
if __name__ == "__main__":
    main()