import pickle
from random import random
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


def preprocess2(datasetpath, dumppath, data_config):
    datapath = os.path.join(datasetpath, 'data')
    labelpath = os.path.join(datasetpath, 'label')
    
    data_filenames = sorted(gather_files_from_folder(datasetpath, '.wav'))
    label_filenames = sorted(gather_files_from_folder(datasetpath, '.txt'))
    
    data = []
    target = []
    min_len = 65536
    
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

        min_len = min(min_len , len(y))
        data.append(y)

        if i == 1000:
            break
    dataset = {
        'data': np.array(data),
        'target': target
    }
    
    if dumppath:
        if not os.path.exists(dumppath):
            os.makedirs(dumppath)
        dumppath = os.path.join(dumppath, 'dataset.pickle')
        with open(dumppath, 'wb') as fp:
            pickle.dump(dataset, fp)


def preprocess3(datasetpath, original_label_file_path, dumppath, data_config):
    # datapath = os.path.join(datasetpath, 'data')
    labelpath = os.path.join(datasetpath, 'label')
    
    data_filenames = sorted(gather_files_from_folder(datasetpath, '.wav'))[:1000]
    label_filenames = sorted(gather_files_from_folder(datasetpath, '.txt'))
    original_label_file_names = sorted(gather_files_from_folder(original_label_file_path, '.wav'))
    
    original_data = {}
    for j, data_filename in tqdm(enumerate(original_label_file_names)):
        label = get_filename(data_filename)
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
        tmp_data = np.zeros((data_config['timeseries_length'], 20 + data_config['num_mfcc']), dtype=np.float64)
        tmp_data[:, 0:data_config['num_mfcc']] = \
            mfcc.T[0:data_config['timeseries_length'], :]
        tmp_data[:, data_config['num_mfcc']:data_config['num_mfcc']+1] = \
            spectral_center.T[0:data_config['timeseries_length'], :]
        tmp_data[:, data_config['num_mfcc']+1:data_config['num_mfcc']+8] =\
            spectral_contrast.T[0:data_config['timeseries_length'], :]
        tmp_data[:, data_config['num_mfcc']+8:data_config['num_mfcc']+20] = \
            chroma.T[0:data_config['timeseries_length'], :] 
        original_data[label] = tmp_data
        
    data = []
    target = []
    
    
    for i, data_filename in tqdm(enumerate(data_filenames)):
        sample_name = get_filename(data_filename)
        label_filename = os.path.join(labelpath, sample_name + '.txt')
        if label_filename not in label_filenames:
            continue
        with open(label_filename, 'r') as f:
            label = f.read().split('\t')
            f.close()
        
        false_ratio = 1.0 / len(label)
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

        tmp_data = np.zeros((2, data_config['timeseries_length'], 20 + data_config['num_mfcc']), dtype=np.float64)
        tmp_data[0, :, 0:data_config['num_mfcc']] = \
            mfcc.T[0:data_config['timeseries_length'], :]
        tmp_data[0, :, data_config['num_mfcc']:data_config['num_mfcc']+1] = \
            spectral_center.T[0:data_config['timeseries_length'], :]
        tmp_data[0, :, data_config['num_mfcc']+1:data_config['num_mfcc']+8] =\
            spectral_contrast.T[0:data_config['timeseries_length'], :]
        tmp_data[0, :, data_config['num_mfcc']+8:data_config['num_mfcc']+20] = \
            chroma.T[0:data_config['timeseries_length'], :]

        for j, data_filename in enumerate(original_label_file_names):
            orin_label = get_filename(data_filename)
            if orin_label in label:
                tmp_data[1, :] = original_data[orin_label][:]
                data.append(tmp_data.copy())
                target.append(1)
            else:
                if random() > false_ratio:
                    tmp_data[1, :] = original_data[orin_label][:]
                    data.append(tmp_data.copy())
                    target.append(0)
                    
    
    dataset = {
        'data': np.array(data),
        'target': target
    }
    
    if dumppath:
        if not os.path.exists(dumppath):
            os.makedirs(dumppath)
        dumppath = os.path.join(dumppath, 'dataset.pickle')
        with open(dumppath, 'wb') as fp:
            pickle.dump(dataset, fp)

def audio_to_tensor(audio, sr, data_config, required_audio_size=5):
    y = librosa.util.fix_length(audio, size=required_audio_size  * sr)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_fft = data_config['n_fft'],
        hop_length=data_config['hop_length'], n_mfcc=data_config['num_mfcc']
    )

    spectral_center = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=data_config['hop_length']
    )

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, 
                                            n_chroma=data_config['num_chroma'],
                                            hop_length=data_config['hop_length'])

    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=data_config['hop_length']
    )
    data = np.zeros(
        (data_config['timeseries_length'],
         8 + data_config['num_chroma'] + data_config['num_mfcc']), dtype=np.float64
    )
    
    data[:, 0:data_config['num_mfcc']] = \
        mfcc.T[0:data_config['timeseries_length'], :]
        
    data[:, data_config['num_mfcc']:data_config['num_mfcc']+1] = \
        spectral_center.T[0:data_config['timeseries_length'], :]
        
    data[:, data_config['num_mfcc'] + 1:data_config['num_mfcc']+data_config['num_chroma']+1] = \
        chroma.T[0:data_config['timeseries_length']+data_config['num_chroma']+1, :]
        
    data[:, data_config['num_mfcc']+data_config['num_chroma']+1:] = \
        spectral_contrast.T[0:data_config['timeseries_length'], :]
    return data

def preprocess(datasetpath, dumppath, data_config):
    # datapath = os.path.join(datasetpath, 'data')
    labelpath = os.path.join(datasetpath, 'label')
    required_audio_size = 5
    data_filenames = sorted(gather_files_from_folder(datasetpath, '.wav'))
    label_filenames = sorted(gather_files_from_folder(datasetpath, '.txt'))
    
    data = np.zeros(
        (len(data_filenames), data_config['timeseries_length'],
         8 + data_config['num_chroma'] + data_config['num_mfcc']), dtype=np.float64
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
        audio, sr = librosa.load(data_filename)
        data[i, :] = audio_to_tensor(audio, sr, data_config)[:]
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