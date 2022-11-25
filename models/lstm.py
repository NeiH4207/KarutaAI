# class definition
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from configs.conf import data_config
from models.nnet import NNet

class CLSTM(NNet):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(NNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTM(input_size, hidden_size,
                             num_layers, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size,
                             num_layers, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 3, 512)
        self.bn1 = nn.BatchNorm1d(512).to(device)
        self.fc2 = nn.Linear(hidden_size * 3, 512)
        self.bn2 = nn.BatchNorm1d(512).to(device)
        self.fc3 = nn.Linear(512, num_classes//2)
        self.fc4 = nn.Linear(512, num_classes//2)
        self.device = device

    def forward(self, x, rlengths=None):
        # Set initial hidden and cell states
        x = x[:, :data_config['timeseries_length']]
        batch_size = x.size(0)
        if rlengths is None:
            rlengths = [-1] * batch_size
            
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # shape = (batch_size, seq_length, hidden_size)
        out1, (h1, c1) = self.lstm1(x, (h0, c0))
        out1 = torch.stack([out1[i][rlengths[i]-5:rlengths[i]-1].mean(axis=0)
                            for i in range(batch_size)], axis=0)
        # shape = (batch_size, seq_length, hidden_size)
        out2, (h2, c2) = self.lstm2(x, (h1, c1))
        out2 = torch.stack([out2[i][rlengths[i]-5:rlengths[i]-1].mean(axis=0)
                            for i in range(batch_size)], axis=0)
        out1 = torch.cat([out1, h1[0], h1[-1]], dim=1)
        out2 = torch.cat([out2, h2[0], h2[-1]], dim=1)
        
        # Decode the hidden state of the last time step
        out1 = F.dropout(F.elu(self.fc1(out1)), p=0.3, training=self.training)
        out2 = F.dropout(F.elu(self.fc2(out2)), p=0.3, training=self.training)
        out = torch.cat([self.fc3(out1), self.fc4(out2)], dim=1)
        return torch.sigmoid(out)
    
class CLSTM2(NNet):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(NNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTM(input_size, hidden_size,
                             num_layers, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size,
                             num_layers, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 6, 512)
        self.bn1 = nn.BatchNorm1d(512).to(device)
        self.fc2 = nn.Linear(hidden_size * 6, 512)
        self.bn2 = nn.BatchNorm1d(512).to(device)
        self.fc3 = nn.Linear(512, num_classes//2)
        self.fc4 = nn.Linear(512, num_classes//2)
        self.device = device

    def forward(self, x, rlengths=None):
        # Set initial hidden and cell states
        x = x[:, :data_config['timeseries_length']]
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size,
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size,
                         self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # shape = (batch_size, seq_length, hidden_size)
        out1, (h1, c1) = self.lstm1(x, (h0, c0))
        # shape = (batch_size, seq_length, hidden_size)
        out2, (h2, c2) = self.lstm2(x, (h1, c1))
        out1 = torch.cat([out2[:, 0, :], out1[:, -1, :], h1[0], h1[-1]], dim=1)
        out2 = torch.cat([out2[:, 0, :], out2[:, -1, :], h2[0], h2[-1]], dim=1)
        # Decode the hidden state of the last time step
        out1 = F.dropout(F.elu(self.fc1(out1)), p=0.3, training=self.training)
        out2 = F.dropout(F.elu(self.fc2(out2)), p=0.3, training=self.training)
        out = torch.cat([self.fc3(out1), self.fc4(out2)], dim=1)
        return torch.sigmoid(out)