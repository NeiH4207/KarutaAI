from libraries.utils import *
import argparse
from libraries.voice import *
from tqdm import tqdm
from random import random, randint


def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--datapath',
                        help='the path of the data',
                        default='./data/JKspeech')
    parser.add_argument('--language',
                        help='the language of the data',
                        default='ej')
    parser.add_argument('--num-gen-data', type=int,
                        help='the number of generated data',
                        default=16384)
    parser.add_argument('--num_mixed', type=int,
                        help='the max number of mixed readers',
                        default=10)
    parser.add_argument('--num_segs', type=int,
                        help='the random segments readers',
                        default=30)
    parser.add_argument('--min_time',
                        help='the minimum time audio', type=float,
                        default=0.5)
    parser.add_argument('--max_time',
                        help='the maximum time audio', type=float,
                        default=2.5)
    parser.add_argument('--gen-data-path',
                        help='the path of the result',
                        default='generated_data/max10/test')
    return parser.parse_args()


def main():
    args = parser_args()
    all_files = gather_file(args.datapath, ext='.wav')
    if args.language == 'en':
        datafile = [os.path.join(args.datapath, _file)
                    for _file in all_files if _file.startswith('E')]
    elif args.language == 'jv':
        datafile = [os.path.join(args.datapath, _file)
                    for _file in all_files if _file.startswith('J')]
    elif args.language == 'ej':
        datafile = [os.path.join(args.datapath, _file)
                    for _file in all_files if _file.startswith('E')] + \
            [os.path.join(args.datapath, _file)
             for _file in all_files if _file.startswith('J')]

    chns = [[0]] * len(datafile)
    wav_data = []
    channels = []
    all_samples = []
    all_labels = []

    for idx, fn in enumerate(datafile):
        data, typ, params = get_wav_channel(fn, chns[idx][0])
        wav_data.append(data)
        channels.append(params.nchannels)
        time_interval = random()*(args.max_time - args.min_time) / \
            args.min_time + args.min_time
        samples = split_wav_by_time(
            data, params, time_interval=time_interval, num_samples=args.num_segs)
        all_samples.append(samples)
        all_labels.append([fn.split('/')[-1].split('.')[0]] * len(samples))

    all_labels = np.array(all_labels, dtype=str)
    all_samples = np.array(all_samples, dtype=object)
    # shuffle samples
    gen_data_path = args.gen_data_path
    data_path = os.path.join(gen_data_path, 'data')
    label_path = os.path.join(gen_data_path, 'label')
    if not os.path.exists(gen_data_path):
        os.makedirs(gen_data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    for idx in tqdm(range(args.num_gen_data)):
        num_mixed = 1 + np.random.randint(1, max(2, args.num_mixed))
        ids = np.random.randint(0, len(all_samples), size=num_mixed)
        labels = []
        samples = []
        selected_sample_ids = np.array([np.random.choice(args.num_segs, 1)[0] 
                            for _ in range(len(ids))])
        for i, id in enumerate(ids):
            samples.append(all_samples[id][i])
            labels.append(all_labels[id][i])
        audio_file_path = os.path.join(
            data_path, '{}_combined_sample_{}.wav'.format(args.language, idx))
        label_file_path = os.path.join(
            label_path, '{}_combined_sample_{}.txt'.format(args.language, idx))
        combine_waves(samples, typ, params, output_fn=audio_file_path)
        with open(label_file_path, 'w') as f:
            f.write('\t'.join(labels))
            f.close()


if __name__ == '__main__':
    main()
