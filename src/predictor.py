import torch
import torch as T
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
import os
import seaborn as sns
from src.data_helper import *
import pandas as pd
from src.evaluator import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import *


class Predictor:
    def __init__(self, model, device=T.device("cpu")):
        self.model = model
        self.device = device
    
    def test(self, audio_file_path=None, 
             label_file_path=None, data_config=None, 
            k=1, plot=False):
        self.model.to(self.device)
        self.model.eval()
        #Going through each data_filename within a label
        audio, sr = librosa.load(audio_file_path, sr=data_config['sr'])
        data = audio_to_tensor(audio, sr, data_config, data_config['fixed-time'])
    
        inp = Variable(torch.FloatTensor(np.array([data])).to(self.device), requires_grad=False)
        prob_out = self.model(inp).detach().cpu().numpy()[0]
        labels = np.array(['E' + ('0' if (i % 44)+1 < 10 else '') + str(i + 1) for i in range(44)]
                          + ['J' + ('0' if (i % 44)+1 < 10 else '') + str(i + 1) for i in range(44)])
        if plot:
            df = pd.DataFrame({'probability': prob_out, 'labels': labels})
            ax = sns.barplot(x='labels', y='probability',
                             data=df, errwidth=0)
            plt.xticks(color='w')
            plt.savefig('images/' + 
                        os.path.basename(audio_file_path).replace('wav', 'png'))
            plt.show()
        ans = labels[np.argsort(prob_out)[::-1]][:k]
        strans = '[' + ", ".join(['"%s"' % x for x in ans]) + ']'
        print(strans)
        
    def plot_prob(self, probs, labels, save_path):
        df = pd.DataFrame({'probability': probs, 'labels': labels})
        ax = sns.barplot(x='labels', y='probability',
                            data=df, errwidth=0)
        plt.xticks(color='w')
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def predict(self, audio_file_path, data_config,
                k=20, plot=False, question_id=0, save_path=None):
        self.model.to(self.device)
        self.model.eval()
        audio, sr = librosa.load(audio_file_path, sr=data_config['sr'])
        data = audio_to_tensor(audio, data_config)
        inp = Variable(torch.FloatTensor(np.array([data])).to(self.device), requires_grad=False)
        prob_out = self.model(inp).detach().cpu().numpy()[0]
        labels = np.array(['E' + ('0' if (i % 44)+1 < 10 else '') + str(i + 1) for i in range(44)]
                          + ['J' + ('0' if (i % 44)+1 < 10 else '') + str(i + 1) for i in range(44)])
        if plot:
            self.plot_prob(prob_out, labels, save_path)
            
        return prob_out, labels
        
    def load_model_from_path(self, path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(T.load(path, map_location=device))
        print("Model loaded sucessful!")
