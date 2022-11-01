import argparse
import json
import os
import numpy as np
from src.request import Socket
from src.karuta import Karuta
from models.arcnn import ARCNN
import torch
import warnings
import librosa
from configs import wavConfig
from libraries.voice import save_wave, get_wav_channel
from src.predictor import Predictor
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol-path", type=str, default="output/solutions")
    parser.add_argument("--output-path", type=str,
                        default="./output/recovered_images/")
    parser.add_argument("--token", type=str,
                        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTQsIm5hbWUiOiLEkOG6oWkgaOG7jWMgQsOhY2gga2hvYSBIw6AgTuG7mWkiLCJpc19hZG1pbiI6ZmFsc2UsImlhdCI6MTY2NjUxODM5Nn0.WJTZuzGlphQzudXQLTM4Pz6SQnoQWgdZC-E_ADCB_yw"
                        )
    parser.add_argument("-s", "--tournament_name",
                        type=str, default='Procon2022')
    parser.add_argument("-r", "--round_name", type=str, default='Round1')
    parser.add_argument("-m", "--match_name", type=str, default='Tran1')
    parser.add_argument("-q", "--question_name", type=str, default='Q_20')
    parser.add_argument("-n", "--n-parts", type=int, default=4)
    parser.add_argument("-a", "--answer_id", type=str, default=None)
    parser.add_argument("-p", "--mode", type=str, default='r')
    
    parser.add_argument('--model-file-path', type=str, 
                        default='trained_models/RCNN2/model.pt')
    parser.add_argument('--cpu', action='store_true',
                        help='Use cpu cores instead')
    args = parser.parse_args()
    return args


def read(socket: Socket, tournament_name, round_name,
         match_name, question_name, n_parts):
    tournaments = socket.get_tournament()
    tournament_id = None
    round_id = None
    match_id = None
    question_id = None
    game_info = Karuta()

    for tournament in tournaments['data']:
        if tournament['name'] == tournament_name:
            tournament_id = tournament['id']
            break
    if tournament_id is None:
        print("Tournament {} not found on server.".format(tournament_name))
        return None
    tournament_info = socket.get_tournament(tournament_id)
    rounds = socket.get_round()

    for round in rounds['data']:
        if round['name'] == round_name:
            round_id = round['id']
            break

    if round_id is None:
        print("Round {} not found on server.".format(round_name))
        return None
    round_info = socket.get_round(round_id)

    matches = socket.get_match()

    for match in matches['data']:
        if match['name'] == match_name:
            match_id = match['id']
            break

    if match_id is None:
        print("match {} not found on server.".format(match_name))
        return None
    match_info = socket.get_match(match_id)

    questions = socket.get_question()

    for question in questions['data']:
        if question['name'] == question_name:
            question_id = question['id']
            break

    if question_id is None:
        print("question {} not found on server.".format(question_name))
        return None

    question_info = socket.get_question(question_id)
    durations, indexes = socket.post_div_audio(question_id, n_parts)
    print("Numparts: {}".format(len(durations)))
    question_info['durations'] = durations
    question_info['indexes'] = indexes
    game_info.tournament = tournament_info
    game_info.round = round_info
    game_info.question = question_info
    game_info.question['question_data'] = json.loads(game_info.question['question_data'])
    return game_info


def main():
    args = parse_args()
    tournament_name = args.tournament_name
    round_name = args.round_name
    match_name = args.match_name
    question_name = args.question_name
    answer_id = args.answer_id
    args = parse_args()
    socket = Socket(args.token)
    game_info = read(socket, tournament_name, round_name, 
                     match_name, question_name, args.n_parts)
    _, _, params = get_wav_channel('data/sample_Q_202205/sample_Q_E01/problem1.wav', 0)
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
    
    probs = []
    
    for id in np.argsort(game_info.question['durations'])[::-1]:
        part = game_info.question['indexes'][id]
        audio = socket.get_div_audio(game_info.question['id'], part)
        # audio = audio[1000:]
        audio_save_path = 'audio/question_{}_{}.wav'.\
                                    format(game_info.question['id'], part)
        # save_wave(audio, params, 1, audio_save_path)
        save_fig_path = audio_save_path.replace('.wav', '.png')
        prob_out, labels = predictor.predict(audio_save_path, data_config,
                k=88, plot=False, 
                question_id=game_info.question['id'],
                save_path=save_fig_path)
        probs.append(np.log(prob_out))
        
        prob_mean = np.mean(probs, axis=0)
        # predictor.plot_prob(np.exp(prob_mean), labels, 'audio/question_{}.png'.\
        #                             format(game_info.question['id']))
        print(labels[np.argsort(np.exp(prob_mean))[::-1]][:game_info.question['question_data']['n_cards']])

if __name__ == '__main__':
    main()
