import pickle
from random import random
import numpy as np
import math
import os
import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import *


def onehot_encode(label, num_labels=88, multilabel=True):
    encoded_label = np.zeros(num_labels, dtype=np.int8)
    if multilabel:
        for i in range(len(label)):
            encoded_label[int(label[i][1:]) - 1 + (44 if label[i][0] == 'J' else 0)] = 1
            
    return encoded_label

def plot(y):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(y, fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.tight_layout()
    plt.show()

def audio_to_tensor(audio, data_config):
    audio = audio.astype(np.float32)
    sr = data_config['sr']
    fixed_size = int(sr * data_config['fixed-time'])
    y = librosa.util.fix_length(audio, size=fixed_size)
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_fft = data_config['n_fft'],
        hop_length=data_config['hop_length'], 
        n_mfcc=data_config['num_mfcc']
    )

    spectral_center = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=data_config['hop_length']
    )

    chroma = librosa.feature.chroma_stft(y=y,
                                         sr=sr,  
                                         n_chroma=data_config['num_chroma'], 
                                         hop_length=data_config['hop_length'])
    
    oenv = librosa.onset.onset_strength(y=y, 
                                        sr=sr, 
                                        hop_length=data_config['hop_length'])
    
    tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr,
                                                    win_length=32,
                                                    hop_length=data_config['hop_length'])
    tempogram = np.abs(tempogram[:, :-1])
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=data_config['hop_length']
    )
    data = np.zeros(
        (data_config['timeseries_length'],
         17 + 8 + data_config['num_chroma'] + data_config['num_mfcc']), dtype=np.float64
    )
    order_sizes = [0, data_config['num_mfcc'], 7, 1, 17,  data_config['num_chroma']]
    cumsum_sizes = np.cumsum(order_sizes)
    order_features = [mfcc, spectral_contrast, spectral_center, tempogram, chroma]
    
    for i in range(len(order_features)):
        data[:, cumsum_sizes[i]:cumsum_sizes[i+1]] = order_features[i].T
    
    return data

def preprocess(datasetpath, dumppath, data_config, n_skips=0):
    # datapath = os.path.join(datasetpath, 'data')
    labelpath = os.path.join(datasetpath, 'label')
    data_filenames = sorted(gather_files_from_folder(datasetpath, '.wav'))
    label_filenames = sorted(gather_files_from_folder(datasetpath, '.txt'))
    
    data = np.zeros(
        (data_config['batch-length'], data_config['timeseries_length'],
         8 + 17 + data_config['num_chroma'] + data_config['num_mfcc']), dtype=np.float64
    )
    target = []
    
    for i, data_filename in tqdm(enumerate(
                data_filenames[n_skips * data_config['batch-length']:])):
        sample_name = get_filename(data_filename)
        label_filename = os.path.join(labelpath, sample_name + '.txt')
        if label_filename not in label_filenames:
            continue
        with open(label_filename, 'r') as f:
            label = onehot_encode(f.read().split('\t'))
            f.close()
        target.append(label)
        #Going through each data_filename within a label
        audio, sr = librosa.load(data_filename, sr=data_config['sr'])
        idx = i % data_config['batch-length']
        data[idx, :] = audio_to_tensor(audio, data_config)[:]
        if (((i + 1) % data_config['batch-length']) == 0) or (i + 1) == len(data_filenames):
            dataset = {
                'data': data[:idx+1],
                'target': target
            }
            if dumppath:
                if not os.path.exists(dumppath):
                    os.makedirs(dumppath)
                _dumppath = os.path.join(dumppath, 'dataset_batch_{}.pickle'.\
                    format(int((i + 1 + n_skips) / data_config['batch-length'])))
                with open(_dumppath, 'wb') as fp:
                    pickle.dump(dataset, fp)
            data = np.zeros(
                (data_config['batch-length'], data_config['timeseries_length'],
                8 + 17 + data_config['num_chroma'] + data_config['num_mfcc']), dtype=np.float64
            )
            target = []
        
def load_data(data_path):
    if not os.path.exists(data_path):
        print('Not found dataset in', data_path)
        return None, None
        
    print("Data loading ...")
    with open(data_path, "rb") as fp:
        data = pickle.load(fp)
        fp.close()
    x = np.array(data["data"])
    y = np.array(data["target"])
    print("Loaded Data\n")
    return x, y

if __name__ == "__main__":
    preprocess('/home/hienvq/Desktop/AI/KarutaAI/generated_data/test', None)
