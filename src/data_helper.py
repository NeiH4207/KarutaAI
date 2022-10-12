import pickle
import numpy as np
import math
import os
import librosa
import json
from tqdm import tqdm

from src.utils import *


def onehot_encode(label, num_labels=88, multilabel=True):
    encoded_label = np.zeros(num_labels, dtype=np.int8)
    if multilabel:
        for i in range(len(label)):
            encoded_label[int(label[i][1:]) - 1 + (44 if label[i][0] == 'J' else 0)] = 1
            
    return encoded_label

def preprocess(datasetpath, dumppath, data_config):
    datapath = os.path.join(datasetpath, 'data')
    labelpath = os.path.join(datasetpath, 'label')
    
    data_filenames = sorted(gather_files_from_folder(datasetpath, '.wav'))
    label_filenames = sorted(gather_files_from_folder(datasetpath, '.txt'))
    
    data = np.zeros(
        (len(data_filenames), data_config['timeseries_length'], 33), dtype=np.float64
    )
    target = []
    
    for i, data_filename in tqdm(enumerate(data_filenames)):
        sample_name = get_filename(data_filename)
        label_filename = os.path.join(labelpath, sample_name + '.txt')
        if label_filename not in label_filenames:
            continue
        with open(label_filename, 'r') as f:
            label = onehot_encode(f.read().split('\t'))
            f.close()
        target.append(label)
        #Going through each data_filename within a label
        y, sr = librosa.load(data_filename)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_fft = data_config['n_fft'],
            hop_length=data_config['hop_length'], n_mfcc=data_config['num_mfcc']
        )
        spectral_center = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=data_config['hop_length']
        )
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=data_config['hop_length'])
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=data_config['hop_length']
        )


        data[i, :, 0:13] = mfcc.T[0:data_config['timeseries_length'], :]
        data[i, :, 13:14] = spectral_center.T[0:data_config['timeseries_length'], :]
        data[i, :, 14:26] = chroma.T[0:data_config['timeseries_length'], :]
        data[i, :, 26:33] = spectral_contrast.T[0:data_config['timeseries_length'], :]
    
    dataset = {
        'data': data,
        'target': target
    }
    
    if dumppath:
        if not os.path.exists(dumppath):
            os.makedirs(dumppath)
        dumppath = os.path.join(dumppath, 'dataset.pickle')
        with open(dumppath, 'wb') as fp:
            pickle.dump(dataset, fp)
        
def load_data(data_path):
    dumppath = os.path.join(data_path, 'dataset.pickle')
    if not os.path.exists(dumppath):
        print('Not found dataset in', dumppath)
        return None, None
        
    print("Data loading ...\n")
    with open(dumppath, "rb") as fp:
        data = pickle.load(fp)
    x = np.array(data["data"])
    y = np.array(data["target"])
    print("Loaded Data")
    return x, y

if __name__ == "__main__":
    preprocess('/home/hienvq/Desktop/AI/KarutaAI/generated_data/test', None)