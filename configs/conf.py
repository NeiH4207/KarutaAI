data_config = {
    'num_mfcc': 39,
    'num_chroma': 64,
    'n_fft': 2048,
    'hop_length': 128,
    'sr': 48000,
    'fixed-time': 1
}
data_config['timeseries_length'] = 1 + int(1 + \
    (data_config['fixed-time'] * data_config['sr'] - 1) // data_config['hop_length'])

# data_config_2= {
#     'num_mfcc': 39,
#     'num_chroma': 64,
#     'n_fft': 2048,
#     'hop_length': 512,
#     'sr': 48000,
#     'fixed-time': 1
# }
# data_config_2['timeseries_length'] =  int(1 + \
#     (data_config['fixed-time'] * data_config['sr'] - 1) // data_config['hop_length'])


wav_params = {
    'nchannels': 1, 
    'sampwidth': 2, 
    'framerate': 48000, 
    'nframes': 96000, 
    'comptype': 'NONE', 
    'compname': 'not compressed'
}