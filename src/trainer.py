import torch
import torch as T
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
import os
import seaborn as sns
from src.data_helper import *
import pandas as pd
from src.evaluator import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import *

class Trainer:
    def __init__(self, model, device=T.device("cpu"), 
                 save_dir="./trainned_models",
                 save_name="model.pt",
                 verbose=True):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.save_name = save_name
        self.verbose = verbose
        self.device=device
        self.evaluater = Evaluator()
        self.train_losses = []

        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
        
        self.set_model_path(save_dir, save_name)
        
    def set_model_path(self, save_dir, save_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.model_path = os.path.join(save_dir, save_name)
        
    def set_data(self, train_x, train_y, val_x, val_y):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
    
    def split_batch(self, x, y, batch_size, shuffle=True):
        batches = []
        if shuffle:
            indices = np.random.permutation(len(y))
            dataset = [x[i] for i in indices]
            labelset = [y[i] for i in indices]
        else:
            dataset = x
            labelset = y
        for i in range(0, len(dataset), batch_size):
            batches.append((dataset[i:i + batch_size], labelset[i:i + batch_size]))
            
        return batches
    
    def eval(self, val_loader):
        val_losses = []
        val_accuracies = []
        
        self.model.eval()
        for inp, lab in val_loader:
            inp = Variable(torch.FloatTensor(inp), requires_grad=False)
            lab = Variable(torch.FloatTensor(lab), requires_grad=False)
            inp, lab = inp.to(self.device).detach(), lab.to(self.device).detach()
            out = self.model(inp).detach()
            loss = self.model.loss(out.detach(), lab.detach())
            out = out.cpu().numpy()
            out = [x[::-1][:int(sum(lab[i]).item())] for i, x in enumerate(out.argsort())]
            for i, x in enumerate(out):
                out_vec = np.zeros(self.model.num_classes)
                out_vec[x] = 1
                out[i] = out_vec
                
            acc = self.evaluater.multilabel_accuracy(out, lab.cpu().numpy())
            val_losses.append(loss.item())
            val_accuracies.append(acc)
            
        self.model.train()
        
        return np.mean(val_losses), np.mean(val_accuracies)
    
    def test(self, audio_file_path=None, label_file_path=None, data_config=None):
        self.model.to(self.device)
        self.model.eval()
        with open(label_file_path, 'r') as f:
            target = onehot_encode(f.read().split('\t')).astype(np.bool)
            f.close()
        #Going through each data_filename within a label
        audio, sr = librosa.load(audio_file_path)
        data = audio_to_tensor(audio, sr, data_config)
    
        
        inp = Variable(torch.FloatTensor([data]).to(self.device), requires_grad=False)
        prob_out = self.model(inp).detach().cpu().numpy()[0]
        labels = np.array(['E' + str(i + 1) for i in range(44)] + ['J' + str(i + 1) for i in range(44)] )
        df = pd.DataFrame({'probability':prob_out,'target': target, 'labels': labels})
        ax = sns.barplot(x='labels', y='probability', hue='target',
                 data=df, errwidth=0)
        plt.xticks(color = 'w')
        plt.show()
        print(prob_out[np.argsort(prob_out)[::-1]])
        print(labels[np.argsort(prob_out)[::-1][:sum(target)]])
        print(labels[np.argsort(prob_out)][::-1])
        print(np.argsort(prob_out)[::-1])
        
    def train(self, optimizer='adam', training_params=None):
        # utility for running the training process
        self.model.to(self.device)
        epochs = training_params['epochs']
        batch_size = training_params['batch_size']
        self.model.set_loss_function(training_params['loss_function'])
        self.model.set_optimizer(optimizer, lr=training_params['learning_rate'])
        # print options
        counter = 0
        print_every = 5
        clip = 5
        valid_loss_min = np.Inf
        valid_acc_max = - np.Inf
        
        self.model.train()
        # scheduler = ReduceLROnPlateau(self.model.optimizer, factor=0.5, patience=0, verbose=True)
        train_loader = self.split_batch(self.train_x, self.train_y, batch_size=batch_size, shuffle=True)
        val_loader = self.split_batch(self.val_x, self.val_y, batch_size=batch_size, shuffle=False)
        
        for i in range(epochs):
            t = tqdm(train_loader)
            tot_loss = 0
            counter = 0
            for inputs, labels in t:
                counter += 1
                inputs = Variable(torch.FloatTensor(inputs), requires_grad=True)
                labels = Variable(torch.FloatTensor(labels))
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.model.zero_grad()
                output = self.model(inputs)
                loss = self.model.loss(output, labels.detach())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.model.step()
                tot_loss += loss.item()
                if counter % print_every == 0:
                    t.set_postfix(loss=tot_loss / counter)
                    
            val_loss, val_acc = self.eval(val_loader)
            
            if self.verbose:
                print('Epoch: {}/{}'.format(i + 1, epochs))
                print('Val accuracy:', val_acc)
                print('Val loss:', val_loss)
            if val_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.\
                    format(valid_loss_min, val_loss))
                torch.save(self.model.state_dict(), self.model_path)
                valid_loss_min = val_loss
            elif val_acc > valid_acc_max:
                valid_acc_max = val_acc
                torch.save(self.model.state_dict(), self.model_path)
            # scheduler.step(val_loss)
                    
        return (self.model)

            
    def load_model_from_path(self, path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(T.load(path, map_location=device))
        print("Model loaded sucessful!")
    
    def save_train_losses(self):
        plt.plot(self.train_losses)
        out_dir = 'output/train_losses'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig("{}/{}_{}".format(out_dir, self.model.name, 'train_losses.png'))
    