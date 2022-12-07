import torch
import torch as T
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import seaborn as sns
from src.data_helper import *
import pandas as pd
from src.utils import *
import matplotlib
matplotlib.style.use('ggplot')


class Predictor:
    def __init__(self, model, data_config, fixed_length=False, device=T.device("cpu")):
        self.model = model
        self.device = device
        self.data_config = data_config
        self.fixed_length = fixed_length
        self.labels = np.array(['E' + ('0' if (i % 44)+1 < 10 else '') + str(i + 1) for i in range(44)]
                          + ['J' + ('0' if (i % 44)+1 < 10 else '') + str(i + 1) for i in range(44)])
    
    def get_labels(self):
        return self.labels

    def plot_prob(self, probs, labels, save_path=None, show=True):
        df = pd.DataFrame({'probability': probs, 'labels': labels})
        summited_columns = np.array([True] * len(labels))
        summited_columns[-10:] = False
        df['summitted'] = summited_columns
        ax = sns.barplot(x='labels', y='probability',
                         hue='summitted',
                        data=df, errwidth=0)
        # plt.xticks(color='w')
        plt.xticks(rotation='vertical')
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        
    def predict(self, audio_file_path, plot=False, save_path=None, return_label=False):
        self.model.to(self.device)
        self.model.eval()
        audio, sr = librosa.load(audio_file_path, sr=self.data_config['sr'])
        data = audio_to_tensor(audio, self.data_config, self.fixed_length)
        rleng = data.shape[0]
        if not self.fixed_length:
            data = np.concatenate([data, np.zeros((max(1, self.data_config['timeseries_length'] - data.shape[0]), 
                                                data.shape[1]), dtype=np.float32)], axis=0)
        inp = Variable(torch.FloatTensor(np.array([data])).to(self.device), requires_grad=False)
        prob_out = self.model(inp, [rleng]).detach().cpu().numpy()[0]
        if plot:
            self.plot_prob(prob_out, self.labels, save_path)
        
        if return_label:
            return prob_out, self.labels
        else:
            return prob_out
    
    def predict_by_audio(self, audio, plot=False, save_path=None, return_label=False):
        self.model.to(self.device)
        self.model.eval()
        data = audio_to_tensor(audio, self.data_config)
        inp = Variable(torch.FloatTensor(np.array([data])).to(self.device), requires_grad=False)
        prob_out = self.model(inp).detach().cpu().numpy()[0]
        if plot:
            self.plot_prob(prob_out, self.labels, save_path)
        
        if return_label:
            return prob_out, self.labels
        else:
            return prob_out
        
    def load_model_from_path(self, path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(T.load(path, map_location=device))
        print("Model loaded sucessful!")
