import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.nnet import NNet

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device='cpu'):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.lstm1 = nn.LSTM(input_size, hidden_size,
                             num_layers, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size,
                             num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        # Set initial hidden and cell states
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
        return torch.cat([out1[:, -1, :], h1[0], h1[-1]], dim=1),\
            torch.cat([out2[:, -1, :], h2[0], h2[-1]], dim=1)


class CNN(nn.Module):
    def __init__(
        self,
        conv_kernels,
        paddings,
        strides,
        channels,
        pool_kernels,
        in_channels=1,
        device='cpu'
    ):
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.conv_kernels = conv_kernels
        self.paddings = paddings
        self.strides = strides
        self.channels = channels
        self.pool_kernels = pool_kernels

        cnn = nn.Sequential()

        def add_block(i, batch_normalization=False, leaky_relu=False):
            n_in = self.in_channels if i == 0 else self.channels[i - 1]
            n_out = self.channels[i]

            cnn.add_module(
                f'conv2D_{i}',
                nn.Conv2d(
                    n_in, n_out, self.conv_kernels[i], self.strides[i], self.paddings[i])
            )
            if batch_normalization:
                cnn.add_module(
                    f'batch_normalization_{i}', nn.BatchNorm2d(n_out))
            if leaky_relu:
                cnn.add_module(f'activation_{i}',
                               nn.LeakyReLU(0.2, in_place=True))
            else:
                cnn.add_module(f'activation_{i}', nn.ReLU(True))

            cnn.add_module(
                f'pooling_{i}',
                (nn.MaxPool2d(kernel_size=self.pool_kernels[i]) 
                    if i%2==0 else nn.AvgPool2d(kernel_size=self.pool_kernels[i])
                )
            )

        # input_shape = (batch, 1, 235, 136)
        add_block(0, batch_normalization=True)  # (batch, 64, _, _)
        cnn.add_module('dropout_0', nn.Dropout2d(0.7))
        add_block(1, batch_normalization=True)  # (batch, 128, _, _)
        cnn.add_module('dropout_1', nn.Dropout2d(0.5))
        add_block(2, batch_normalization=True)  # (batch, 128, _, _)
        cnn.add_module('dropout_2', nn.Dropout2d(0.5))
        add_block(3, batch_normalization=True)  # (batch, 128, _, _)
        cnn.add_module('dropout_3', nn.Dropout2d(0.3))

        self.cnn = cnn
        self.device = device

    def forward(self, x):
        # input (batch, channels, frequency, time)
        x = torch.unsqueeze(x, 1)
        conv_output = self.cnn(x)
        batch_size, channels, frequency, time = conv_output.size()
        # (batch, channels, frequency, time) -> (batch, time, frequency, channels)
        conv_output = conv_output.permute(0, 3, 2, 1)
        batch_size, time, frequency, channels = conv_output.size()
        # (batch, time, frequency, channels) -> (batch, time, frequency * channels)
        conv_output = conv_output.reshape(batch_size, time, frequency * channels)
        return conv_output


class ALSTM(NNet):
    '''
    Adaptive LSTM
    '''
    def __init__(self, input_shape, num_chunks, rnn_hidden_size, rnn_num_layers,
                 num_classes, in_channels=1, dropout=0.0, device='cpu'):
        super(NNet, self).__init__()

        self.input_shape = input_shape
        self.num_chunks = num_chunks
        self.in_channels = in_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.num_classes = num_classes
        self.n_cnn_layers = 4
        self.device = device
        self.window_length = ((( self.input_shape[0] - 1) // self.num_chunks) + 1)

        self.rnn = BiLSTM(
            input_size=self.input_shape[1],
            hidden_size=self.rnn_hidden_size // 2,
            num_layers=self.rnn_num_layers,
            device = self.device
        )
        self.fc1 = nn.Linear(self.rnn_hidden_size * 2, 1024)
        self.fc2 = nn.Linear(self.rnn_hidden_size * 2, 1024)
        self.fc3 = nn.Linear(1024 * self.num_chunks, num_classes // 2)
        self.fc4 = nn.Linear(1024 * self.num_chunks, num_classes // 2)

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        # input shape = (batch, channels, frequency, time)
        # split to chunks
        x = x.permute(0, 2, 1)
        padding_size = self.num_chunks * self.window_length - x.shape[-1]
        padding_part = Variable(torch.zeros((x.shape[0], x.shape[1], padding_size)), 
                                requires_grad=True).to(self.device)
        x = torch.cat([x, padding_part], axis=-1)
        chunks = torch.chunk(x, self.num_chunks, dim=-1)
        lstm_out1 = None
        lstm_out2 = None
        
        for chunk in chunks:
            lstm_chunk = chunk.permute(0, 2, 1)
            lstm_chunk_out1,  lstm_chunk_out2 = self.rnn(lstm_chunk)
            lstm_chunk_out1 = lstm_chunk_out1.unsqueeze(1)
            lstm_chunk_out2 = lstm_chunk_out2.unsqueeze(1)
            
            if lstm_out1 is None:
                lstm_out1 = lstm_chunk_out1
                lstm_out2 = lstm_chunk_out2
            else:
                lstm_out1 = torch.cat([lstm_out1, lstm_chunk_out1, lstm_chunk_out2], dim=1)
                lstm_out2 = torch.cat([lstm_out2, lstm_chunk_out2], dim=1)

        out1 = self.dropout(F.elu(self.fc1(lstm_out1)))
        out2 = self.dropout(F.elu(self.fc2(lstm_out2)))
        out = torch.cat([self.fc3(out1.view(out1.shape[0], -1)), 
                         self.fc4(out2.view(out2.shape[0], -1))], dim=1)
        return self.sigmoid(out)
