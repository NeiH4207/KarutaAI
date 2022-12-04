import argparse
import numpy as np
from models.lstm import CLSTM
from src.karuta import Karuta
import torch
import warnings
from src.predictor import Predictor
from copy import deepcopy as copy
from configs.conf import data_config
from src.request import Socket

from src.utils import intersection
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str,
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTQsIm5hbWUiOiLEkOG6oWkgaOG7jWMgQsOhY2gga2hvYSBIw6AgTuG7mWkiLCJpc19hZG1pbiI6ZmFsc2UsImlhdCI6MTY3MDE0MTA2M30.jQg5OascEeYAK1RVbYVEwALunilNJ8XlTzNtrYh0MOE"
    )
    parser.add_argument("-s", "--tournament_name",
                        type=str, default='')
    parser.add_argument("-r", "--round_name", type=str, default='')
    parser.add_argument("-m", "--match_name", type=str, default='Tran1')
    parser.add_argument("-q", "--question_name", type=str, default='Q_18')
    parser.add_argument("--qid", type=int, default=52)
    parser.add_argument("-c", "--account", type=str, default='BK.PuzzleGod')
    parser.add_argument("-n", "--new", action='store_true')
    parser.add_argument("--part-id", type=int, default=0)
    parser.add_argument("--save-audio-part", action='store_true')
    parser.add_argument("-a", "--answer_id", type=str, default=None)
    parser.add_argument("--download-all-answers", action='store_true')
    
    parser.add_argument('--model-file-path', type=str, 
                        default='trainned_models/LSTM2/model.pt')
    parser.add_argument('--model-file-path2', type=str, 
                        default='trainned_models/LSTM1/model.pt')
    parser.add_argument('--cpu', action='store_true',
                        help='Use cpu cores instead')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    socket = Socket(args.token)
    
    karuta = Karuta(socket)
    try:
        karuta.read_by_id(args.qid)
    except Exception as e:
        print(e)
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
    predictor1 = Predictor(model, data_config, fixed_length=False, device=device)
    predictor1.load_model_from_path(args.model_file_path)
    
    model2 = CLSTM(
        input_size=128,
        hidden_size=512,
        num_layers=2,
        num_classes=88,
        device=device
    )    
    
    predictor2 = Predictor(model2, data_config, fixed_length=True, device=device)
    predictor2.load_model_from_path(args.model_file_path2)
    karuta.set_predictor(predictors=[predictor1, predictor2])
    labels = []
    n_cards = karuta.num_cards
    labels = predictor1.get_labels()
    
    probs, answers = karuta.request_audio(return_probs=True, save_audio_part=args.save_audio_part)
    # orders = np.argsort(prob_sum)[::-1][:n_cards+10]
    ans_out = labels[np.argsort(probs)[::-1]][:n_cards].tolist()
    answer = ans_out
    old_corrects, old_changes = karuta.submit(answer)
    if old_corrects == -1:
        for ans in answers:
            print("Num correct intersect {}/{}".format(len(intersection(ans, answer)), n_cards))
        return
    """
        Reader cards selected by K best probs from LSTM trained model.
    """
    new_corrects = old_corrects
    new_changes = old_changes
    orders = np.argsort(probs)[::-1]
    bad_card = labels[orders[-1]]
    skipping_cards = np.zeros(88, dtype=bool)
    karuta.predictors[0].plot_prob(probs[orders][:n_cards+10], labels[orders][:n_cards+10], 
                                    'tmp/question_{}.png'.format(karuta.question_id))
    
    q = input('Change answer? yes/no (y/n): ')
    if 'y' in q.lower():
        while new_corrects != n_cards:
            wrong_card = None
            wrong_id = None
            for i in range(n_cards):
                if skipping_cards[orders[n_cards-i-1]]:
                    continue
                answer = copy(ans_out)
                answer[n_cards-i-1] = bad_card
                answer = sorted(answer)
                corrects, _ = karuta.submit(answer)
                if corrects == old_corrects:
                    wrong_card = orders[n_cards-i-1]
                    wrong_id = n_cards-i-1
                    break
                else:
                    skipping_cards[orders[n_cards-i-1]] = True
                    
            probs[wrong_card] = 0      
            
            for change_card_id in orders[n_cards:]:
                print('Change {} to {}'.format(labels[wrong_card], labels[change_card_id]))
                answer = copy(ans_out)
                answer[wrong_id] = labels[change_card_id]
                answer = sorted(answer)
                corrects, changes = karuta.submit(answer)
                if corrects > new_corrects:
                    new_corrects, new_changes = corrects, changes
                    break
                else:
                    probs[change_card_id] = 0
                
            orders = np.argsort(probs)[::-1]
            ans_out = labels[orders][:n_cards].tolist()
            answer = sorted(ans_out)
            old_corrects = new_corrects
            
            karuta.predictors[0].plot_prob(probs[orders][:n_cards+10], labels[orders][:n_cards+10], 
                                            'tmp/question_{}.png'.format(karuta.question_id))
            if new_corrects != n_cards:
                q = input('Change answer? yes/no (y/n): ')
                if 'n' in q.lower():
                    break
            
    print('Num changed used:', new_changes - old_changes)  
    
    for i, ans in enumerate(answers):
        print("Num correct intersect {}/{} in part {}".format(len(intersection(ans, answer)), n_cards, i))
        
if __name__ == '__main__':
    main()
