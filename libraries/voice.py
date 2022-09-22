import wave
import os
import sys
import numpy as np

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

def combinechannels(chdatas0, typ):
    maxdatalen = max(chdata.shape[0] for chdata in chdatas0)
    chdatas = [np.concatenate((chdata0, [0]*(maxdatalen-len(chdata0)))) for chdata0 in chdatas0]
    outputchannels = len(chdatas)
    output_data = np.zeros(outputchannels*chdatas[0].shape[0]).astype(typ)
    for ch,chdata in enumerate(chdatas):
        output_data[ch::outputchannels] = chdata
    
    return output_data, outputchannels
    
def save(output_data, params, outputchannels, ofn):
    outwav = wave.open(ofn,'w')
    outwav.setparams(params)
    outwav.setnchannels(outputchannels)
    outwav.writeframes(output_data.tobytes())
    outwav.close()

def combineMultFns(fns,chns,output_fn, outchannels=None):
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    ch_datas = [[]for _ in range(len(fns))]
    for idx,fn in enumerate(fns):
        for ch in chns[idx]:
            data, typ, params = get_wav_channel(fn,ch)
            ch_datas[idx].append(data)
    output_data = []
    for ch_data in ch_datas:
        output_data += ch_data
    merged_output_data, outputchannels = combinechannels(output_data, typ)
    if outchannels is not None:
        outputchannels = outchannels
    save(merged_output_data, params, outputchannels, output_fn)

def combine_waves(wav_data_list, typ, params, output_fn=None, outputchannels=None):
    merged_output_data, out_channels = combinechannels(wav_data_list, typ)
    if outputchannels is not None:
        out_channels = outputchannels
    if output_fn is not None:
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        save(merged_output_data, params, out_channels, output_fn)
    return merged_output_data, out_channels
    
def split_wav_by_time(wav_data, params, time_interval=1.0, num_samples=1):
    wav_len = wav_data.shape[0]
    seconds = wav_len / params.framerate
    sample_len = int(time_interval / seconds * wav_len)
    random_start_idx = [np.random.randint(0,wav_len-sample_len) for _ in range(num_samples)]
    samples = [wav_data[start_idx:start_idx+sample_len] for start_idx in random_start_idx]
    return samples

def split_wav_random(wav_data, n_samples=1, min_time=0.5):
    return wav_data

if __name__ == '__main__':
    datapath = '/Users/mac/Desktop/AI/KaturaAI/data/JKspeech/'
    result_path = '/Users/mac/Desktop/AI/KaturaAI/result/'
    datafile = ['E01.wav', 'E02.wav', 'E03.wav']
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