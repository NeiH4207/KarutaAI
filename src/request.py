import json
import sys
import requests
import base64
import numpy as np
# from src.data_helper import DataProcessor

# TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwiaWF0IjoxNjQ1OTU2MzYyLCJleHAiOjE2NDU5NzQzNjJ9.a_O5bxFBlPZZRamp5XaiLlxcH7dLbWoHlhP7cZGehGA'
END_POINT_API = 'https://procon2022.duckdns.org'

class Socket:
    def __init__(self, token):
        self.token = token
        self.headers = {
            'Authorization': '{}'.format(self.token)
        }

    def get_tournament(self, id=None):
        url = END_POINT_API + '/tournament'
        if id is not None:
            url = url + '/' + str(id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def get_round(self, id=None):
        url = END_POINT_API + '/round'
        if id is not None:
            url = url + '/' + str(id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def get_match(self, id=None):
        url = END_POINT_API + '/match'
        if id is not None:
            url = url + '/' + str(id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def get_question(self, id=None):
        url = END_POINT_API + '/question'
        if id is not None:
            url = url + '/' + str(id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def get_answer(self, id=None):
        url = END_POINT_API + '/answer'
        if id is not None:
            url = url + '/' + str(id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def get_audio(self, id):
        url = END_POINT_API + '/question/{}/audio/problemdata'.format(id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response

    def post_div_audio(self, id, n_parts):
        url = END_POINT_API + '/question/{}/divided-data'.format(id)
        body = {
            "n_divided": n_parts
        }
        response = requests.post(url, headers=self.headers, 
                                 json=body, verify=False).json()
        if 'data' not in response:
            print ("Question not found")
            sys.exit(0)
        sdata = response['data']
        durations = [x['duration'] for x in sdata]
        indexes = [x['index'] for x in sdata]
        return [durations, indexes]
    

    def get_info_audio_part(self, id, new=False):
        url = END_POINT_API + '/question/{}/divided-data'.format(id)
        body = {
            "new": new
        }
        response = requests.post(url, headers=self.headers, json=body,
                                verify=False).json()
        return response['data']
    
    def get_div_audio(self, id, part):
        url = END_POINT_API + '/question/{}/audio/divided-data?uuid={}'.format(id, part)
        response = requests.get(url, headers=self.headers, verify=False)
        sdata = response.content
        typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(2)
        print ("Extracting channel {} out of {} channels, {}-bit depth".format(0+1, 1, 2*8))
        data = np.frombuffer(sdata.split(b'data')[1][4:], dtype=typ)
        return data
    
    def get_audio_data(self, id=None):
        url = END_POINT_API + '/question/download/raw-challenge/resource'
        if id is not None:
            url = url + '/' + str(id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response
    
    def get_all_answer_info(self, question_id=None):
        url = END_POINT_API + '/answer'
        response = requests.get(url, headers=self.headers, verify=False)
        if question_id is not None:
            for answer in response.json()['data']:
                if answer['question_id'] == question_id:
                    return answer
        return None
    
    def post_answer(self, team_id, match_id, question_id, answer_data):
        url = END_POINT_API + '/answer'
        body = {
            "answer_data": answer_data,
            "question_id": question_id,
            "team_id": team_id,
            "match_id": match_id
        }
        response = requests.post(url, headers=self.headers, 
                                 json=body, verify=False)
        return response.json()
    
    def edit_answer(self, ans_id, answer_data):
        url = END_POINT_API + '/answer/{}'.format(ans_id)
        body = {
            "answer_data": answer_data
        }
        response = requests.put(url, headers=self.headers, 
                                 json=body, verify=False)
        return response.json()
    
    def submit(self, team_id, match_id, question_id, answer_data, not_check=False):    
        answer_info = self.get_all_answer_info(question_id)
        if answer_info == None:
            self.post_answer(team_id, match_id, question_id, answer_data)
        else:
            score_data = json.loads(answer_info['score_data'])['score_data']
            self.edit_answer(answer_info['id'], answer_data)
        answer_info = self.get_all_answer_info(question_id)
        if answer_info is not None:
            score_data = json.loads(answer_info['score_data'])['score_data']
            return score_data['correct'], score_data['changed']
        else:
            print('May be not found this question or team not added in this round')
            return -1, -1
        
    def del_all_answer(self, challenge_id=None):
        url = END_POINT_API + '/solution/delete/{}'.format(challenge_id)
        response = requests.delete(url, headers=self.headers, verify=False)
        return response.json()
    
    def download_all_answers(self, path):
        for id in range(1000):
            url = END_POINT_API + '/answer/{}'.format(id)
            response = requests.get(url, headers=self.headers, verify=False)
            
            if response.status_code == 200:
                response = response.json()
                response['answer_data'] = json.loads(response['answer_data'])
                response['question']['question_data'] = json.loads(response['question']['question_data'])
                response['score_data'] = json.loads(response['score_data'])
                save_path = '{}/answer_{}_{}_{}_{}.json'.format(
                                                    path, 
                                                    response['question']['name'],
                                                    response['match']['id'],
                                                    response['team']['account'], 
                                                    response['id'])
                json.dump(response, open(save_path,  'w'), indent=2)
                print(save_path)
                
    
    def send(self, challenge_id, data_text):
        url = END_POINT_API + '/solution/submit/{}'.format(challenge_id)
        header = self.headers.copy()
        header['Content-Type'] = 'text/plain'
        response = requests.post(url, headers=header, data=data_text, verify=False)
        return response.json()
