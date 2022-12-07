import json
from random import shuffle
import sys
from src.request import Socket
import numpy as np
from libraries.voice import save_wave
from configs.conf import wav_params
import os

from src.utils import bcolors

class Karuta():

    def __init__(self, socket: Socket):
        self.tournament_id = None
        self.round_id = None
        self.team_id = None
        self.match_id = None
        self.question = None
        self.num_cards = 0
        self.tournament_info = None
        self.round_info = None
        self.socket = socket
        self.submision_history = []
        self.part_ids = []
        
    def set_predictor(self, predictors:list):
        self.predictors = predictors
        self.labels = predictors[0].get_labels()
        
    def get_probs(self, audio):
        probs = []
        for predictor in self.predictors:
            probs.append(self.get_probs_from_audio(audio, predictor))
        return np.mean(probs, axis=0)
        
    def play(self):
        return
        
    def read_by_id(self, question_id):
        self.question_id = question_id
        question_info = self.socket.get_question(question_id)
        self.round_id = question_info['match']['round_id']
        self.match_id = question_info['match']['id']
        self.question = question_info
        self.question['question_data'] = json.loads(self.question['question_data'])
        self.num_cards = self.question['question_data']['n_cards']
        self.question_id = self.question['id']
        
    def get_final_score(self, score_data):
        num_changes = score_data['changed'] - self.first_summited_change
        raw_score = score_data['score']['raw_score']
        num_correct = int(raw_score / self.question['question_data']['point_per_correct'])
        final_penalties = num_changes * self.question['question_data']['penalty_per_change']
        final_bonus_factor = (self.question['question_data']['n_parts'] - len(self.part_ids)) \
            / self.question['question_data']['n_parts'] * self.question['question_data']['bonus_factor']
        max_score = self.num_cards * self.question['question_data']['point_per_correct'] \
            *  (1 + self.question['question_data']['bonus_factor'] * \
                (self.question['question_data']['n_parts'] - 1) / self.question['question_data']['n_parts'])
        final_score = (raw_score - final_penalties) * (1 + final_bonus_factor)
        return num_correct, num_changes, final_score, max_score
        
    def submit(self, answer, verbose=True):
        # print('Submit:', answer) 
        self.submision_history.append(answer)
        score_data = self.socket.submit(self.question_id, answer)
        if len(self.submision_history) == 1:
            self.first_summited_change = score_data['changed']
        num_corrects, num_changes, final_score, max_score = self.get_final_score(score_data)
        if verbose:
            print(bcolors.OKBLUE + "Score: {}/{} ({}), change: {}, final score: {}/{}".format(
                num_corrects, self.num_cards, score_data['score']['raw_score'],
                num_changes, final_score, max_score))
        return num_corrects
    
    def request_audio(self, save_audio_part=False, return_probs=True, num_parts=1):    
        new = False
        all_probs = []
        answers = []
        print(bcolors.OKBLUE + "Number of parts required: {}".format(num_parts))
        while True:
            self.part_ids = self.socket.get_info_audio_part(self.question_id, new=new)
            # shuffle(self.part_ids)
            self.part_ids = self.part_ids[1:1+num_parts]
            for id in self.part_ids:
                # save_path = 'audio/question_{}_{}.wav'.\
                #                     format(self.question_id, id)
                audio = self.socket.get_div_audio(self.question_id, id)
                print('Audio length part {}:'.format(id), audio.shape)
                
                probs = self.get_probs(audio)
                best_cards = self.labels[np.argsort(probs)[::-1][:self.num_cards]]
                all_probs.append(probs)
                answers.append(best_cards)
            
            if len(self.part_ids) > 0:
                prob_sum = np.mean(all_probs, axis=0) 
                orders = np.argsort(prob_sum)[::-1][:self.num_cards+10]
                if not os.path.exists('tmp'):
                    os.makedirs('tmp')
                # self.predictors[0].plot_prob(prob_sum[orders], 
                #                     self.labels[orders].tolist(), 'tmp/question_{}.png'.\
                #                     format(self.question_id))
            if len(self.part_ids) >= num_parts:
                break
            new = True
            
        if return_probs:
            return np.mean(all_probs, axis=0), answers
        
    def get_scale_prob(self, chunk_probs):
        prob_mean = np.mean(np.log(chunk_probs), axis=0)
        return np.exp(prob_mean)

    def get_probs_from_audio(self, audio, predictor, save=False,
                            save_path='tmp/part_tmp.wav'):
        chunk_size = 32000
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
