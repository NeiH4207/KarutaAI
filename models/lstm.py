# class definition
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.nnet import NNet

class Encoder(nn.Module):
    def __init__(self, input_size, embbed_size, device='cuda'):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.embbed_size = embbed_size
        
        self.conv1  = nn.Conv2d(in_channels=1, out_channels=64,
                                kernel_size=3,  stride=2, padding=0).to(device)
        self.bn1    = nn.BatchNorm2d(64).to(device)
        self.pool1  = nn.MaxPool2d(kernel_size=2).to(device)
        self.conv1_out_len = (int((input_size - 3) / 2 + 1)) // 2

        self.conv2  = nn.Conv2d(in_channels=64, out_channels=128, 
                                kernel_size=3,  stride=1, padding=0).to(device)
        self.bn2    = nn.BatchNorm2d(128).to(device)
        self.pool2  = nn.MaxPool2d(kernel_size=2).to(device)
        self.conv2_out_len = (int((self.conv1_out_len - 3) / 2 + 1)) // 2

        self.conv3  = nn.Conv2d(in_channels=128, out_channels=256, 
                                kernel_size=3,  stride=1, padding=0).to(device)
        self.bn3   = nn.BatchNorm2d(256).to(device)
        self.pool3  = nn.MaxPool2d(kernel_size=2).to(device)
        self.conv3_out_len = (int((self.conv2_out_len - 3) / 2 + 1)) // 2

        self.conv4  = nn.Conv2d(in_channels=256, out_channels=512, 
                                kernel_size=3,  stride=1, padding=0).to(device)
        self.bn4  = nn.BatchNorm2d(512).to(device)
        self.pool4  = nn.MaxPool2d(kernel_size=2).to(device)
        self.conv4_out_len = (int((self.conv3_out_len - 3) / 2 + 1)) // 2

        self.linear_input = 12288
        self.fc = nn.Linear(self.linear_input, embbed_size).to(device)
        self.fc_bn = nn.BatchNorm1d(embbed_size).to(device)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x_w = F.relu(self.pool1(self.bn1(self.conv1(x))))
        x_w = F.relu(self.pool2(self.bn2(self.conv2(x_w))))
        x_w = F.relu(self.pool3(self.bn3(self.conv3(x_w))))
        x_w = F.relu(self.pool4(self.bn4(self.conv4(x_w))))
        x_w = x_w.view(x_w.shape[0], -1)
        x_embbed = F.dropout(F.elu(self.fc(x_w)), p=0.3, training=self.training)
        return x_embbed

class CLSTM(NNet):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(NNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional= True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional= True, batch_first=True)
        # self.encoder1 = Encoder(input_size, self.hidden_size, device)
        # self.encoder2 = Encoder(input_size, self.hidden_size, device)
        self.fc1 = nn.Linear(hidden_size * 4, 512)
        self.bn1    = nn.BatchNorm1d(512).to(device)
        self.fc2 = nn.Linear(hidden_size * 4, 512)
        self.bn2    = nn.BatchNorm1d(512).to(device)
        self.fc3 = nn.Linear(512, num_classes//2)
        self.fc4 = nn.Linear(512, num_classes//2)
        self.device = device

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device) 

        # Forward propagate LSTM
        out1, (h1, c1) = self.lstm1(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
        out2, (h2, c2) = self.lstm2(x, (h1, c1))  # shape = (batch_size, seq_length, hidden_size)
        out1 = torch.cat([out1[:, -1, :], h1[0], h1[-1]], dim=1)
        out2 = torch.cat([out2[:, -1, :], h2[0], h2[-1]], dim=1)
        # Decode the hidden state of the last time step
        out1 = F.dropout(F.elu(self.fc1(out1)), p=0.3, training=self.training)
        out2 = F.dropout(F.elu(self.fc2(out2)), p=0.3, training=self.training)
        out = torch.cat([self.fc3(out1), self.fc4(out2)], dim=1)
        return torch.sigmoid(out)
    
class CNN(NNet):
    def __init__(self, input_size, embbed_size, num_classes, device):
        super(NNet, self).__init__()
        self.embbed_size = embbed_size
        self.num_classes = num_classes
        # self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional= True, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional= True, batch_first=True)
        self.encoder1 = Encoder(input_size, self.embbed_size, device)
        self.encoder2 = Encoder(input_size, self.embbed_size, device)
        self.fc1 = nn.Linear(embbed_size, 512)
        self.bn1    = nn.BatchNorm1d(512).to(device)
        self.fc2 = nn.Linear(embbed_size, 512)
        self.bn2    = nn.BatchNorm1d(512).to(device)
        self.fc3 = nn.Linear(512, num_classes//2)
        self.fc4 = nn.Linear(512, num_classes//2)
        self.device = device

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        embbed_1 = self.encoder1(x)
        embbed_2 = self.encoder2(x)
        # Decode the hidden state of the last time step
        out1 = F.dropout(F.elu(self.fc1(embbed_1)), p=0.3, training=self.training)
        out2 = F.dropout(F.elu(self.fc2(embbed_2)), p=0.3, training=self.training)
        out = torch.cat([self.fc3(out1), self.fc4(out2)], dim=1)
        return torch.sigmoid(out)