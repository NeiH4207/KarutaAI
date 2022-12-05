# KaturaAI

### Folder tree

```
.
├── configs
│   ├── conf.py
│   ├── __init__.py
│   └── wav_config.json
├── data
│   ├── JKspeech
│   ├── problem
│   └── sample_Q_202205
├── interaction.py
├── libraries
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

### Setup Environment

```
conda create -n karuta python=3.8

conda activate karuta

pip install . --no-cache-dir --upgrade
```

### Play

```
karuta --token <TOKEN> -q <question id> --num_parts <num of required parts>
```
