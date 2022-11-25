import argparse
import numpy as np
from models.lstm import CLSTM, CLSTM2
from src.karuta import Karuta
import torch
import warnings
from src.predictor import Predictor
from copy import deepcopy as copy
from configs.conf import data_config, data_config_2
from src.request import Socket

from src.utils import intersection
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol-path", type=str, default="output/solutions")
    parser.add_argument("--output-path", type=str,
                        default="./output/recovered_images/")
    parser.add_argument("--token", type=str,
                        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTQsIm5hbWUiOiLEkOG6oWkgaOG7jWMgQsOhY2gga2hvYSBIw6AgTuG7mWkiLCJpc19hZG1pbiI6ZmFsc2UsImlhdCI6MTY2OTI3NjU1NX0.Lo5zx4bWpP9itBiHOcFExlLwGDnH8Pa-jxSi19X0VSY"
                        )
    parser.add_argument("-s", "--tournament_name",
                        type=str, default='Procon2022')
    parser.add_argument("-r", "--round_name", type=str, default='Round1')
    parser.add_argument("-m", "--match_name", type=str, default='Tran1')
    parser.add_argument("-q", "--question_name", type=str, default='Q_18')
    parser.add_argument("--ID", type=int, default=None)
    parser.add_argument("-c", "--account", type=str, default='BK.PuzzleGod')
    parser.add_argument("-n", "--new", action='store_true')
    parser.add_argument("--part-id", type=int, default=0)
    parser.add_argument("--save-audio-part", action='store_true')
    parser.add_argument("-a", "--answer_id", type=str, default=None)
    parser.add_argument("--download-all-answers", action='store_true')
    
    parser.add_argument('--model-file-path', type=str, 
                        default='trainned_models/LSTM7/model.pt')
    parser.add_argument('--model-file-path2', type=str, 
                        default='trainned_models/LSTM5/model.pt')
    parser.add_argument('--cpu', action='store_true',
                        help='Use cpu cores instead')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    socket = Socket(args.token)
    
    karuta = Karuta()
    
    if args.download_all_answers:
        socket.download_all_answers('tmp/answers/')
        return
    
    try:
        if args.ID:
            karuta.read_by_id(socket, args.ID)
        else:
            tournament_name = args.tournament_name
            round_name = args.round_name
            match_name = args.match_name
            question_name = args.question_name
            answer_id = args.answer_id
            karuta.read(socket, tournament_name, round_name, 
                            match_name, args.account, question_name)
    except Exception as e:
        print(e)
        return None
    
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
    predictor1 = Predictor(model, data_config, fixed_length=False, device=device)
    predictor1.load_model_from_path(args.model_file_path)
    
    model2 = CLSTM2(
        input_size=128,
        hidden_size=512,
        num_layers=2,
        num_classes=88,
        device=device
    )    
    
    predictor2 = Predictor(model2, data_config, fixed_length=True, device=device)
    predictor2.load_model_from_path(args.model_file_path2)
    
    all_probs = []
    answers = []
    labels = []
    total_shape = 0
    n_cards = karuta.num_cards
    labels = predictor1.get_labels()
    
    new = False
    while True:
        part_ids = socket.get_info_audio_part(karuta.question_id, new=new)
        print(part_ids)
        all_probs = []
        answers = []
        for id in part_ids:
            save_path = 'audio/question_{}_{}.wav'.\
                                format(karuta.question_id, id)
            audio = socket.get_div_audio(karuta.question_id, id)
            print('Audio length part {}:'.format(id), audio.shape)
            total_shape += audio.shape[0]
            probs1 = karuta.get_probs_from_audio(audio, predictor1, 
                                        args.save_audio_part, save_path)
            probs2 = karuta.get_probs_from_audio(audio, predictor2, 
                                        args.save_audio_part, save_path)
            beta = 0.5
            probs = np.mean([beta*probs1, (1-beta)*probs2], axis=0)
            best_cards = labels[np.argsort(probs)[::-1][:n_cards]]
            all_probs.append(probs)
            answers.append(best_cards)
        
        if len(part_ids) > 0:
            prob_sum = np.mean(all_probs, axis=0) 
            orders = np.argsort(prob_sum)[::-1][:n_cards+10]
            predictor1.plot_prob(prob_sum[orders], 
                                labels[orders].tolist(), 'audio/question_{}.png'.\
                                format(karuta.question_id))
        q = input('Get new part? yes/no (y/n)')
        if 'y' in q.lower():
            new = True
        else:
            if len(part_ids) > 0:
                break
            else:
                print("No data for predict, choose yes to get a new data")
                return
        
    prob_sum = np.mean(all_probs, axis=0)  
    orders = np.argsort(prob_sum)[::-1][:n_cards+10]
    ans_out = labels[np.argsort(prob_sum)[::-1]][:n_cards].tolist()
    answer = ans_out
    probs = prob_sum
    print('Submit:', sorted(ans_out))
    # predictor1.plot_prob(probs[orders], labels[orders], 'audio/question_{}.png'.\
    #                             format(karuta.question_id))
    old_corrects, old_changes = socket.submit(karuta.team_id, karuta.match_id, 
                      karuta.question_id, answer)
    if old_corrects == -1:
        for ans in answers:
            print("Num correct intersect {}/{}".format(len(intersection(ans, answer)), n_cards))
        return
    """
        Reader cards selected by K best probs from LSTM trained model.
    """
    print(old_corrects, old_changes)
    new_corrects = old_corrects
    new_changes = old_changes
    orders = np.argsort(probs)[::-1]
    bad_card = labels[orders[-1]]
    skipping_cards = np.zeros(88, dtype=bool)
    
    q = input('Change answer? yes/no (y/n)')
    if 'y' in q.lower():
        while new_corrects != n_cards:
            wrong_cards = []
            for i in range(n_cards):
                if skipping_cards[orders[n_cards-i-1]]:
                    continue
                answer = copy(ans_out)
                answer[n_cards-i-1] = bad_card
                answer = sorted(answer)
                corrects, _ = socket.submit(karuta.team_id, karuta.match_id, 
                        karuta.question_id, answer)
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
            new_corrects, new_changes = socket.submit(karuta.team_id, karuta.match_id, 
                        karuta.question_id, answer)
            old_corrects = new_corrects
            print('Submit:', answer)
            print('Results:', new_corrects, new_changes)  
            print('----------------------------------------------------------------') 
            # orders = np.argsort(probs)[::-1][:n_cards+10]
            # predictor1.plot_prob(probs[orders], 
            #                     labels[orders].tolist(), 'audio/question_{}.png'.\
            #                     format(karuta.question_id))    
            
    print('Num changed used:', new_changes - old_changes)  
    
    for ans in answers:
        print("Num correct intersect {}/{}".format(len(intersection(ans, answer)), n_cards))
        
if __name__ == '__main__':
    main()
