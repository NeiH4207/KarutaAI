import wave
import os
import sys
import numpy as np
from pydub import AudioSegment

def get_wav_channel( fn, channel):
    '''
    take filename , get specific channel of wav file
    '''
    wav = wave.open(fn)
    # Read data
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    # print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.frombuffer(sdata, dtype=typ)
    ch_data = data[channel::nch]

    return ch_data, typ, wav.getparams()

def save(output_data, params, outputchannels, ofn):
    outwav = wave.open(ofn,'w')
    outwav.setparams(params)
    outwav.setnchannels(1)
    outwav.writeframes(output_data.astype('<i2').tobytes())
    outwav.close()

def combine_waves(frames, typ, params, output_fn=None, outputchannels=None):
    if output_fn is not None:
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    samples = [np.frombuffer(f, dtype='<i2') for f in frames]
    samples = [samp.astype(np.float64) for samp in samples]
    # mix as much as possible
    min_len = min(map(len, samples))
    mix = samples[0][:min_len]
    for i in range(1, len(samples)):
        mix += samples[i][:min_len]
        
    save(mix, params, 1, output_fn)
    
def split_wav_by_time(wav_data, params, time_interval=1.0, num_samples=1):
    wav_len = wav_data.shape[0]
    seconds = wav_len / params.framerate
    sample_len = int(time_interval / seconds * wav_len)
    random_start_idx = [np.random.randint(0,wav_len - sample_len) for _ in range(num_samples)]
    samples = [wav_data[start_idx:start_idx+sample_len] for start_idx in random_start_idx]
    return samples

if __name__ == '__main__':
    datapath = '/home/hienvq/Desktop/AI/KarutaAI/data/'
    result_path = '/Users/mac/Desktop/AI/KaturaAI/result/'
    datafile = ['Q_12.wav']
    num_data = 10
    min_v = 2
    max_v = 10
    datafile = [os.path.join(datapath, _file) for _file in datafile]
    chns = [[0]] * len(datafile)
    wav_data = []
    channels = []
    all_samples = []
    for idx, fn in enumerate(datafile):
        data, typ, params = get_wav_channel(fn, chns[idx][0])
        wav_data.append(data)
        channels.append(params.nchannels)
        
        samples = split_wav_by_time(data, params, time_interval=1.5, num_samples=3)
        all_samples.extend(samples)
        
    # shuffle samples
    np.random.shuffle(all_samples)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    for idx in range(num_data):
        # get min_v -> max_v random samples not same
        ids = np.random.randint(0, len(all_samples), size=num_data)
        samples = [all_samples[idx] for idx in ids]
        combined_sample = combine_waves(samples, typ, params, 
                                        output_fn=os.path.join(result_path, 'combined_sample_{}.wav'.format(idx)))