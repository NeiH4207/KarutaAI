import argparse
import json
import os
import numpy as np
from models.lstm import CLSTM
from src.request import Socket
from src.karuta import Karuta
from models.arcnn import ARCNN
import torch
import warnings
from libraries.voice import save_wave, get_wav_channel
from src.predictor import Predictor
from copy import deepcopy as copy
from scipy.stats import entropy
from configs.conf import wav_params, data_config

from src.utils import intersection
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
    parser.add_argument("-q", "--question_name", type=str, default='Q_15')
    parser.add_argument("--ID", type=int, default=None)
    parser.add_argument("-c", "--account", type=str, default='BK.PuzzleGod')
    parser.add_argument("-n", "--n-parts", type=int, default=4)
    parser.add_argument("-a", "--answer_id", type=str, default=None)
    parser.add_argument("--download-all-answers", action='store_true')
    
    parser.add_argument('--model-file-path', type=str, 
                        default='trainned_models/LSTM5/model.pt')
    parser.add_argument('--model-file-path2', type=str, 
                        default='trainned_models/LSTM/backup_LSTM3.pt')
    parser.add_argument('--cpu', action='store_true',
                        help='Use cpu cores instead')
    args = parser.parse_args()
    return args


def read(socket: Socket, tournament_name, round_name,
         match_name, account, question_name, n_parts):
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
    
    tournament_info = socket.get_tournament(tournament_id)
    rounds = socket.get_round()

    for round in rounds['data']:
        if round['name'] == round_name:
            round_id = round['id']
            break

    if round_id is None:
        print("Round {} not found on server.".format(round_name))
    
    round_info = socket.get_round(round_id)

    matches = socket.get_match()

    for match in matches['data']:
        if match['name'] == match_name:
            match_id = match['id']
            break

    if match_id is None:
        print("match {} not found on server.".format(match_name))
    
    match_info = socket.get_match(match_id)
    teams = match_info['teams']
    
    for team in teams:
        if team['account'] == account:
            team_id = team['id']
            break

    if team_id is None:
        print("team {} not found on server.".format(team_id))
        return None
    
    questions = socket.get_question()

    for question in questions['data']:
        if question['name'] == question_name and question['match']['id'] == match_id:
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
    game_info.tournament_id = tournament_id
    game_info.round_id = round_id
    game_info.team_id = team_id
    game_info.match_id = match_id
    game_info.question = question_info
    game_info.question['question_data'] = json.loads(game_info.question['question_data'])
    return game_info

def read_by_id(socket: Socket, question_id, n_parts):
    game_info = Karuta()

    question_info = socket.get_question(question_id)
    durations, indexes = socket.post_div_audio(question_id, n_parts)
    print("Numparts: {}".format(len(durations)))
    question_info['durations'] = durations
    question_info['indexes'] = indexes
    game_info.round_id = question_info['match']['round_id']
    game_info.match_id = question_info['match']['id']
    game_info.question = question_info
    game_info.question['question_data'] = json.loads(game_info.question['question_data'])
    return game_info

def get_scale_prob(chunk_probs):
    prob_mean = np.mean(np.log(chunk_probs), axis=0)
    return np.exp(prob_mean)

def get_probs_from_audio(audio, predictor):
    chunk_size = 24000
    chunk_stride = 12000
    chunk_probs = []
    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')
    audio_save_path = 'tmp/part_tmp.wav'
    
    for i_chunk_part in range(audio.shape[0]//chunk_stride):
        if min(chunk_stride * i_chunk_part + chunk_size, audio.shape[0]) \
                    - chunk_stride * i_chunk_part < 24000:
            break
        _audio = audio[chunk_stride * i_chunk_part:chunk_stride * i_chunk_part + chunk_size]
        save_wave(_audio, wav_params.values(), 1, audio_save_path)
        prob_out = predictor.predict(audio_save_path, data_config,
                k=88, plot=False, 
                question_id=None,
                save_path=None,
                return_label=False)
        chunk_probs.append(prob_out)
        
    os.remove(audio_save_path)
    return get_scale_prob(chunk_probs)

def main():
    args = parse_args()
    socket = Socket(args.token)
    
    if args.download_all_answers:
        socket.download_all_answers('tmp/answers/')
        return
    
    if args.ID:
        game_info = read_by_id(socket, args.ID, args.n_parts)
    else:
        tournament_name = args.tournament_name
        round_name = args.round_name
        match_name = args.match_name
        question_name = args.question_name
        answer_id = args.answer_id
        game_info = read(socket, tournament_name, round_name, 
                        match_name, args.account, question_name, args.n_parts)
    if not game_info:
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cpu:
        device = 'cpu'
    
    model = CLSTM(
        input_size=128,
        hidden_size=512,
        num_layers=2,
        num_classes=88,
        device=device
    )    
    # model2 = CLSTM2(
    #     input_size=128,
    #     hidden_size=512,
    #     num_layers=2,
    #     num_classes=88,
    #     device=device
    # )    
    predictor = Predictor(model, device=device)
    # predictor2 = Predictor(model2, device=device)
    
    predictor.load_model_from_path(args.model_file_path)
    # predictor2.load_model_from_path(args.model_file_path2)
    
    all_probs = []
    labels = []
    total_shape = 0
    answers = []
    n_cards = game_info.question['question_data']['n_cards']
    
    for id in np.argsort(game_info.question['durations'])[::-1]:
        part = game_info.question['indexes'][id]
        audio = socket.get_div_audio(game_info.question['id'], part)
        print('Audio length part {}:'.format(part), audio.shape)
        total_shape += audio.shape[0]
        probs = get_probs_from_audio(audio, predictor)
        labels = predictor.get_labels()
        best_cards = labels[np.argsort(probs)[::-1][:n_cards]]
        all_probs.append(probs)
        answers.append(best_cards)
    prob_sum = np.mean(all_probs, axis=0)  
    ans_out = labels[np.argsort(prob_sum)[::-1]][:n_cards].tolist()
    answer = ans_out
    probs = prob_sum
    print('Submit:', sorted(ans_out))
    predictor.plot_prob(probs, labels, 'audio/question_{}.png'.\
                                format(game_info.question['id']))
    team_id, match_id, question_id = game_info.team_id, game_info.match_id, game_info.question['id']
    old_corrects, old_changes = socket.submit(team_id, match_id, question_id, ans_out)
    if old_corrects == -1:
        for ans in answers:
            print("Num correct intersect {}/{}".format(len(intersection(ans, answer)), n_cards))
        return
    print(old_corrects, old_changes)
    new_corrects = old_corrects
    new_changes = old_changes
    orders = np.argsort(probs)[::-1]
    bad_card = labels[orders[-1]]
    skipping_cards = np.zeros(88, dtype=bool)
    while new_corrects != n_cards:
        wrong_cards = []
        for i in range(n_cards):
            if skipping_cards[orders[n_cards-i-1]]:
                continue
            answer = copy(ans_out)
            answer[n_cards-i-1] = bad_card
            answer = sorted(answer)
            corrects, _ = socket.submit(team_id, match_id, question_id, answer)
            if corrects == old_corrects:
                wrong_cards.append(orders[n_cards-i-1])
            if len(wrong_cards) == n_cards - new_corrects:
                break
        for i in range(n_cards):
            if orders[i] not in wrong_cards:
                skipping_cards[orders[i]] = True
        probs[np.array(wrong_cards)] = 0       
        print('Change:', labels[wrong_cards]) 
        orders = np.argsort(probs)[::-1]
        ans_out = labels[orders][:n_cards].tolist()
        answer = sorted(ans_out)
        new_corrects, new_changes = socket.submit(team_id, match_id, question_id, answer)
        old_corrects = new_corrects
        print('Submit:', answer)
        print('Results:', new_corrects, new_changes)     
    print('Num changed used:', new_changes - old_changes)  
    
    for ans in answers:
        print("Num correct intersect {}/{}".format(len(intersection(ans, answer)), n_cards))
        
if __name__ == '__main__':
    main()
