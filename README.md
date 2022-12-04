# KaturaAI

### Folder tree

```
.
├── build
│   ├── bdist.linux-x86_64
│   └── lib
├── configs
│   ├── conf.py
│   ├── __init__.py
│   ├── __pycache__
│   └── wav_config.json
├── configs.py
├── data
│   ├── JKspeech
│   ├── problem
│   └── sample_Q_202205
├── inference.py
├── interaction.py
├── libraries
│   ├── __pycache__
│   ├── utils.py
│   └── voice.py
├── models
│   ├── lstm.py
│   └── nnet.py
├── README.md
├── setup.py
├── src
│   ├── data_helper.py
│   ├── encoder.py
│   ├── evaluator.py
│   ├── karuta.py
│   ├── predictor.py
│   ├── request.py
│   └── utils.py
└── trainned_models
    ├── LSTM1
    └── LSTM2
```

### Installation

```
pip install -e .
```

### How to generate data

```
python3 generate_data.py --datapath ./data/JKspeech/ --language ej --num-gen-data 5000 --num_merge_data 5 --gen-data-path generated_data/train/
```


```
python3 generate_data.py --datapath ./data/JKspeech/ --language ej --num-gen-data 500 --num_merge_data 5 --gen-data-path generated_data/val/
```

```
python3 generate_data.py --datapath ./data/JKspeech/ --language ej --num-gen-data 500 --num_merge_data 5 --gen-data-path generated_data/test/
```

```
usage: generate_data.py [-h] [--datapath DATAPATH] [--language LANGUAGE] [--num-gen-data NUM_GEN_DATA]
                        [--num_merge_data NUM_MERGE_DATA] [--min_time MIN_TIME] [--max_time MAX_TIME]
                        [--gen-data-path GEN_DATA_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --datapath DATAPATH   the path of the data
  --language LANGUAGE   the language of the data
  --num-gen-data NUM_GEN_DATA
                        the number of generated data
  --num_merge_data NUM_MERGE_DATA
                        the max number of mixed readers
  --min_time MIN_TIME   the minimum time audio
  --max_time MAX_TIME   the maximum time audio
  --gen-data-path GEN_DATA_PATH
                        the path of the result

```

### How to training model

```
python3 main.py --preprocess \
    --model-save-dir trained_models/ \
    --train-folder generated_data/train \
    --val-folder generated_data/val \
    --test-folder generated_data/test \
    --original-label-data-path data/JKspeech/ \
    --processed-data-path tmp/ \
    -e 5 -v
```

```
usage: main.py [-h] [--train-folder TRAIN_FOLDER] [--val-folder VAL_FOLDER] [--test-folder TEST_FOLDER]
               [--original-label-data-path ORIGINAL_LABEL_DATA_PATH]
               [--processed-data-path PROCESSED_DATA_PATH] [--model-path MODEL_PATH] [-d MODEL_SAVE_DIR]
               [-l LOSS] [-o OPTIMIZER] [-e EPOCHS] [--lr LR] [-b BATCH_SIZE] [--seed SEED] [--load-model]
               [--preprocess] [-v]

optional arguments:
  -h, --help            show this help message and exit
  --train-folder TRAIN_FOLDER
                        path to training data
  --val-folder VAL_FOLDER
                        path to training data
  --test-folder TEST_FOLDER
                        path to training data
  --original-label-data-path ORIGINAL_LABEL_DATA_PATH
  --processed-data-path PROCESSED_DATA_PATH
  --model-path MODEL_PATH
  -d MODEL_SAVE_DIR, --model-save-dir MODEL_SAVE_DIR
                        directory to save model
  -l LOSS, --loss LOSS  loss function to use
  -o OPTIMIZER, --optimizer OPTIMIZER
                        optimizer to use
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train
  --lr LR               learning rate
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size
  --seed SEED           seed for training
  --load-model          Retrainging with old model
  --preprocess          Show validation results
  -v, --verbose         Show validation results

```

### How to inference an audio file with trained model

```
python3 inference.py --audio-file-path data/sample_Q_202205/sample_Q_E01/problem.wav --model-file-path ./trained_models/model.pt -k 3 
```

```
usage: inference.py [-h] [--audio-file-path AUDIO_FILE_PATH] [--model-file-path MODEL_FILE_PATH]
                    [-d MODEL_SAVE_DIR] [-k K] [--cpu]

optional arguments:
  -h, --help            show this help message and exit
  --audio-file-path AUDIO_FILE_PATH
                        audio file to training data
  --model-file-path MODEL_FILE_PATH
  -d MODEL_SAVE_DIR, --model-save-dir MODEL_SAVE_DIR
                        directory to save model
  -k K                  Num mixed readers
  --cpu                 Use cpu cores instead
```
