# class definition
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.nnet import NNet

class LSTM(NNet):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(NNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional= True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.device = device

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device) 

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
        out = torch.cat([out[:, -1, :], h[0]], dim=1)
        # Decode the hidden state of the last time step
        out = self.fc1(out)
        out = self.fc2(out)
        return torch.sigmoid(out)