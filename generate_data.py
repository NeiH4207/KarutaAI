from libraries.utils import *
import argparse
from libraries.voice import *
from tqdm import tqdm

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
                        default=327680)
    parser.add_argument('--num_mixed', type=int,
                        help='the max number of mixed readers',
                        default=15)
    parser.add_argument('--min_time',
                        help='the minimum time audio', type=float,
                        default=0.5)
    parser.add_argument('--max_time',
                        help='the maximum time audio', type=float,
                        default=1.0)
    parser.add_argument('--gen-data-path',
                        help='the path of the result',
                        default='generated_data/max20_3/train/')
    return parser.parse_args()


def main():
    args = parser_args()
    all_files = gather_file(args.datapath, ext='.wav')
    datafile = []
    
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
        all_labels.append(fn.split('/')[-1].split('.')[0])
        
    all_labels = np.array(all_labels)
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
        num_mixed = 1 + np.random.randint(2, max(3, args.num_mixed))
        labels = []
        samples = []
        selected_sample_ids = np.random.choice(88, num_mixed, replace=False)
        time_interval = random() * (args.max_time - args.min_time) + args.min_time
        for id in selected_sample_ids:
            samples.append(get_wav_by_time(wav_data[id], params, time_interval))
        labels = all_labels[selected_sample_ids]
        audio_file_path = os.path.join(
            data_path, '{}_combined_sample_{}.wav'.format(args.language, idx))
        label_file_path = os.path.join(
            label_path, '{}_combined_sample_{}.txt'.format(args.language, idx))
        combine_waves(samples, typ, params, output_fn=audio_file_path)
        with open(label_file_path, 'w') as f:
            f.write('\t'.join(sorted(labels)))
            f.close()


if __name__ == '__main__':
    main()
