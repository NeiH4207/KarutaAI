import os
import pickle
import torch
import torch as T
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm

from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset
from deep_audio_features.models.cnn import load_cnn
from deep_audio_features.models.convAE import load_convAE
from deep_audio_features.utils.model_editing import drop_layers
from deep_audio_features.lib.training import test
from src.encoder import Encoder

from src.utils import gather_files_from_folder

class Trainer:
    def __init__(self, model, loss, optimizer, train_loader=None, test_loader=None,
                 device=T.device("cpu"), lr=0.001, epochs=1000, batch_size=64,
                 n_repeats = 2, print_every=1, save_every=500, 
                 save_dir="./trainned_models",
                 save_name="model.pt", verbose=True):
        self.model = model
        self.model.set_loss_function(loss)
        self.model.set_optimizer(optimizer, lr)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_repeats = n_repeats
        self.print_every = print_every
        self.save_every = save_every
        self.save_dir = save_dir
        self.save_name = save_name
        self.verbose = verbose
        self.encoder = Encoder()
        self.train_losses = []

        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
    
    def load_extract_model(self, modelpath=None, layers_dropped=0):
        """Loads a model and predicts each classes probability

    Arguments:

            modelpath {str} : A path where the model was stored.

            ifile {str} : A path of a given wav file,
                        which will be tested.
            test_segmentation {bool}: If True extracts segment level
                            predictions of a sequence
            verbose {bool}: If True prints the predictions

    Returns:

            y_pred {np.array} : An array with the probability of each class
                                that the model predicts.
            posteriors {np.array}: An array containing the unormalized
                                posteriors of each class.

        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Restore model
        with open(modelpath, "rb") as input_file:
            model_params = pickle.load(input_file)
        if "classes_mapping" in model_params:
            self.task = "classification"
            model, hop_length, window_length = load_cnn(modelpath)
            self.class_names = model.classes_mapping
            # Apply layer drop
            model = drop_layers(model, layers_dropped)
        else:
            self.task = "representation"
            model, hop_length, window_length = load_convAE(modelpath)
        
        model.layers_dropped = layers_dropped
        
        model.hop_length = hop_length
        model.window_length = window_length
        model = model.to(device)
        max_seq_length = model.max_sequence_length

        model.max_sequence_length = max_seq_length
        self.extract_model = model
    
    def extract_features_from_wav_files(self, wav_files, label_files, test_segmentation=False):
        
        labels = [open(wav_file, 'r').read().split('\t') for wav_file in label_files]
        
        onehot_labels = [self.encoder.onehot_encode(label) for label in labels]
        
        labels = [label[0] for label in labels]
        
        feature_data = FeatureExtractorDataset( X=wav_files,
                                                # Random class -- does not matter at all
                                                y=onehot_labels,
                                                fe_method="",
                                                oversampling=False,
                                                max_sequence_length=self.extract_model.max_sequence_length,
                                                zero_pad=self.extract_model.zero_pad,
                                                forced_size=self.extract_model.spec_size,
                                                fuse=self.extract_model.fuse, show_hist=False,
                                                test_segmentation=test_segmentation,
                                                hop_length=self.extract_model.hop_length, 
                                                window_length=self.extract_model.window_length)

        # Create test dataloader
        feature_loader = DataLoader(dataset=feature_data, batch_size=32,
                                num_workers=4, drop_last=False,
                                shuffle=False)

        # Forward a sample
        _, preds, _ = test(model=self.extract_model, dataloader=feature_loader,
                                    cnn=True, task=self.task,
                                    classifier=False if self.extract_model.layers_dropped == 0
                                    else False)
        
        return preds
    
    def load_data(self, train_folder, val_folder, test_folder):
        train_data_files = gather_files_from_folder(os.path.join('data', train_folder), _extension='.wav')[:32]
        val_data_files = gather_files_from_folder(os.path.join('data', val_folder), _extension='.wav')[:32]
        test_data_files = gather_files_from_folder(os.path.join('data', test_folder), _extension='.wav')[:32]
        
        train_label_files = gather_files_from_folder(os.path.join('label', train_folder), _extension='.txt')[:32]
        val_label_files = gather_files_from_folder(os.path.join('label', val_folder), _extension='.txt')[:32]
        test_label_files = gather_files_from_folder(os.path.join('label', test_folder), _extension='.txt')[:32]
        
        train_data = self.extract_features_from_wav_files(train_data_files, train_label_files)
        val_data = self.extract_features_from_wav_files(val_data_files, val_label_files)
        test_data = self.extract_features_from_wav_files(test_data_files, test_label_files)
        
    def split_batch(self, dataset, batch_size, shuffle=True):
        """
        Split dataset into batches
        :param dataset: dataset
        :param batch_size: batch size
        :return: batches
        """
        batches = []
        if shuffle:
            indices = np.random.permutation(len(dataset['data']))
            data_set = [dataset['data'][i] for i in indices]
            target_set = [dataset['target'][i] for i in indices]
        for i in range(0, len(data_set), batch_size):
            batches.append((data_set[i:i + batch_size], target_set[i:i + batch_size]))
        return batches
    
    def train(self):
        self.model.to(self.device)

            
    def load_model_from_path(self, path):
        self.model.load_state_dict(T.load(path))
    
    def save_train_losses(self):
        plt.plot(self.train_losses)
        out_dir = 'output/train_losses'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig("{}/{}_{}".format(out_dir, self.model.name, 'train_losses.png'))
    
    def test(self):
        if len(self.test_loader['data']) == 0:
            print('Skipping test')
        self.model.eval()