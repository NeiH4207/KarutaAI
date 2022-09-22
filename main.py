from libraries.utils import *
import argparse
from libraries.voice import *


def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--datapath', 
                        help='the path of the data', 
                        default='/Users/mac/Desktop/AI/KaturaAI/data/JKspeech')
    parser.add_argument('--language', 
                        help='the language of the data', 
                        default='ej')
    parser.add_argument('--num_data',
                        help='the number of generated data',
                        default=10) 
    parser.add_argument('--min_time',
                        help='the minimum time audio',
                        default=2)
    parser.add_argument('--max_time',
                        help='the maximum time audio',
                        default=10)
    parser.add_argument('--result_path',
                        help='the path of the result',
                        default='/Users/mac/Desktop/AI/KaturaAI/result')
    return parser.parse_args()

def main():
    args = parser_args()
    all_files = gather_file(args.datapath, ext='.wav')
    if args.language == 'en':
        datafile =  [os.path.join(args.datapath, _file) 
                     for _file in all_files if _file.startswith('E')]
    elif args.language == 'jv':
        datafile =  [os.path.join(args.datapath, _file) 
                     for _file in all_files if _file.startswith('J')]
    elif args.language == 'ej':
        datafile =  [os.path.join(args.datapath, _file) 
                     for _file in all_files if _file.startswith('E')] + \
                    [os.path.join(args.datapath, _file) 
                     for _file in all_files if _file.startswith('J')]
        
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
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    for idx in range(args.num_data):
        ids = np.random.randint(0, len(all_samples), size=args.num_data)
        samples = [all_samples[idx] for idx in ids]
        file_path = os.path.join(args.result_path, '{}_combined_sample_{}.wav'.format(args.language, idx))
        combine_waves(samples, typ, params, output_fn=file_path)

if __name__ == '__main__':
    main()