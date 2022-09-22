import json
import os
from src.request import Socket
from src.karuta import Karuta
import warnings
warnings.filterwarnings("ignore")
import argparse

'''
python3 interaction.py \
    -s 'Computer_Tour' -r Computer_Round -p 'r' \
         --token '' \
              -m Natural_8
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol-path", type=str, default="output/solutions")
    parser.add_argument("--output-path", type=str, default="./output/recovered_images/")
    parser.add_argument( "--token", type=str, 
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTQsIm5hbWUiOiLEkOG6oWkgaOG7jWMgQsOhY2gga2hvYSBIw6AgTuG7mWkiLCJpc19hZG1pbiI6ZmFsc2UsImlhdCI6MTY2MzgyOTAxNn0.ENFdEtxlGAVAxSAdpcer9LhJFrOqHZTOKwbzxeORSDA"
    )
    parser.add_argument("-s", "--tournament_name", type=str, default='Procon2022')
    parser.add_argument("-r", "--round_name", type=str, default='Round1')
    parser.add_argument("-m", "--match_name", type=str, default='Tran1')
    parser.add_argument("-q", "--question_name", type=str, default='Q_3')
    parser.add_argument("-a", "--answer_id", type=str, default=None)
    parser.add_argument("-p", "--mode", type=str, default='r')
    args = parser.parse_args()
    return args


def read(socket: Socket, tournament_name, round_name, match_name, question_name, get_img_info=False):
    tournaments = socket.get_tournament()
    tournament_id = None
    round_id = None
    match_id = None
    question_id = None
    
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
    audio = socket.get_div_audio(question_id, 1)
    print('Read Match sucessful')
    
    game_info = Karuta()
    game_info.name = match_name
    if not get_img_info:
        return game_info
    game_info.mode = 'rgb'
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
    if args.mode == 'r':
        game_info = read(socket, tournament_name, round_name, match_name, question_name, get_img_info=True)
    else:
        game_info = read(socket, tournament_name, round_name, match_name, question_name)
    if args.mode == 'r':
        game_info = read(socket, tournament_name, round_name, match_name, question_name, get_img_info=True)
        game_info.save_to_json()
    elif args.mode == 'w':
        file_path = os.path.join(args.sol_path, args.match_name + '.txt')
        f = open(file_path, 'r')
        lines = f.readlines()
        lines = [line.replace('\n', '') for line in lines]
        data_text = '\n'.join(lines)
        data_text  = data_text.encode('utf-8')
        res = socket.send(challenge_id=game_info.challenge_id, data_text=data_text)
        print(res)
    elif args.mode == 'show':
        solution = socket.get_answer(id=answer_id)
        print(json.dumps(solution, indent = 1))
    elif args.mode == 'del':
        res = socket.del_all_answer(challenge_id=game_info.challenge_id)
        print(res)
    else:
        print('mode invalid, please use -m w or -m r')
    
if __name__ == '__main__':
    main()