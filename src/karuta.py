import json
from src.request import Socket
import numpy as np
from libraries.voice import save_wave, get_wav_channel
from configs.conf import wav_params, data_config
import os

class Karuta():

    def __init__(self):
        self.tournament_id = None
        self.round_id = None
        self.team_id = None
        self.match_id = None
        self.question = None
        self.num_cards = 0
        self.tournament_info = None
        self.round_info = None
        
    def play(self):
        return
    
    def read(self, socket: Socket, tournament_name, round_name,
            match_name, account, question_name):
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
        
        self.tournament_info = socket.get_tournament(tournament_id)
        rounds = socket.get_round()

        for round in rounds['data']:
            if round['name'] == round_name:
                round_id = round['id']
                break

        if round_id is None:
            print("Round {} not found on server.".format(round_name))
        
        self.round_info = socket.get_round(round_id)

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
        self.tournament_id = tournament_id
        self.round_id = round_id
        self.team_id = team_id
        self.match_id = match_id
        self.question = question_info
        self.question['question_data'] = json.loads(self.question['question_data'])
        self.num_cards = self.question['question_data']['n_cards']
        self.question_id = self.question['id']
        
    def read_by_id(self, socket: Socket, question_id, n_parts):

        question_info = socket.get_question(question_id)
        self.round_id = question_info['match']['round_id']
        self.match_id = question_info['match']['id']
        self.question = question_info
        self.question['question_data'] = json.loads(self.question['question_data'])

    def get_scale_prob(self, chunk_probs):
        prob_mean = np.mean(np.log(chunk_probs), axis=0)
        return np.exp(prob_mean)

    def get_probs_from_audio(self, audio, predictor, save=False,
                            save_path='tmp/part_tmp.wav'):
        chunk_size = 30000
        chunk_stride = 12000
        chunk_probs = []
        par_path = save_path.replace(os.path.basename(save_path), '')
        if not os.path.exists(par_path):
            os.makedirs(par_path)
        audio_save_path = save_path
        
        for i_chunk_part in range(audio.shape[0]//chunk_stride):
            if min(chunk_stride * i_chunk_part + chunk_size, audio.shape[0]) \
                        - chunk_stride * i_chunk_part < 24000:
                break
            _audio = audio[chunk_stride * i_chunk_part:chunk_stride * i_chunk_part + chunk_size]
            save_wave(_audio, wav_params.values(), 1, audio_save_path)
            prob_out = predictor.predict(audio_save_path,
                                        plot=False, save_path=None, return_label=False)
            chunk_probs.append(prob_out)
        
        if not save:
            os.remove(audio_save_path)
        return self.get_scale_prob(chunk_probs)
