import requests
import base64
import numpy as np
# from src.data_helper import DataProcessor

# TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwiaWF0IjoxNjQ1OTU2MzYyLCJleHAiOjE2NDU5NzQzNjJ9.a_O5bxFBlPZZRamp5XaiLlxcH7dLbWoHlhP7cZGehGA'
END_POINT_API = 'http://procon2022.duckdns.org'

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
    
    def get_div_audio(self, id, part):
        url = END_POINT_API + '/question/{}/audio/divided-data?index={}'.format(id, part)
        response = requests.get(url, headers=self.headers, verify=False)
        sdata = response.content
        typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(4)
        print ("Extracting channel {} out of {} channels, {}-bit depth".format(0+1, 1, 2*8))
        data = np.frombuffer(sdata, dtype=typ)
        return data
    
    def get_audio_data(self, id=None):
        url = END_POINT_API + '/question/download/raw-challenge/resource'
        if id is not None:
            url = url + '/' + str(id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response
    
    def get_all_answer_info(self, challenge_id=None):
        url = END_POINT_API + '/solution/team/{}'.format(challenge_id)
        response = requests.get(url, headers=self.headers, verify=False)
        return response.json()
    
    def del_all_answer(self, challenge_id=None):
        url = END_POINT_API + '/solution/delete/{}'.format(challenge_id)
        response = requests.delete(url, headers=self.headers, verify=False)
        return response.json()
    
    def send(self, challenge_id, data_text):
        url = END_POINT_API + '/solution/submit/{}'.format(challenge_id)
        header = self.headers.copy()
        header['Content-Type'] = 'text/plain'
        response = requests.post(url, headers=header, data=data_text, verify=False)
        return response.json()
    