import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.nnet import NNet


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
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
        return out1, out2


class CNN(nn.Module):
    def __init__(
        self,
        conv_kernels,
        paddings,
        strides,
        channels,
        pool_kernels,
        in_channels=1,
        device=device
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
                nn.MaxPool2d(kernel_size=self.pool_kernels[i])
            )

        # input_shape = (batch, 1, 235, 136)
        add_block(0, batch_normalization=True)  # (batch, 64, _, _)
        cnn.add_module('dropout_0', nn.Dropout2d(0.3))
        add_block(1, batch_normalization=True)  # (batch, 128, _, _)
        cnn.add_module('dropout_1', nn.Dropout2d(0.3))
        add_block(2, batch_normalization=True)  # (batch, 256, _, _)
        cnn.add_module('dropout_2', nn.Dropout2d(0.3))
        add_block(3, batch_normalization=True)  # (batch, 512, _, _)
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
        conv_output = conv_output.view(batch_size, time, frequency * channels)
        return conv_output


class CRNN(NNet):
    def __init__(self, input_shape, rnn_hidden_size, rnn_num_layers, num_classes, in_channels=1, dropout=0.0, device=device):
        super(NNet, self).__init__()

        self.input_shape = input_shape
        self.in_channels = in_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.num_classes = num_classes

        conv_kernels = [3, 3, 3, 3]
        paddings = [1, 1, 1, 1]
        strides = [1, 1, 1, 1]
        channels = [64, 128, 256, 512]
        pool_kernels = [2, 2, 2, 2]

        def cal_cnn_out(h, w):
            for i in range(len(channels)):
                padding = paddings[i] if isinstance(
                    paddings[i], tuple) else (paddings[i], paddings[i])
                stride = strides[i] if isinstance(
                    strides[i], tuple) else (strides[i], strides[i])
                kernel = conv_kernels[i] if isinstance(
                    conv_kernels[i], tuple) else (conv_kernels[i], conv_kernels[i])
                pooling = pool_kernels[i] if isinstance(
                    pool_kernels[i], tuple) else (pool_kernels[i], pool_kernels[i])
                h = (h + 2 * padding[0] - kernel[0] - 2) / stride[0] + 1
                w = (w + 2 * padding[1] - kernel[1] - 2) / stride[1] + 1
                h = h // pooling[0]
                w = w // pooling[1]
            return int(h) * channels[-1]

        self.cnn = CNN(
            in_channels=in_channels,
            conv_kernels=conv_kernels,
            paddings=paddings,
            strides=strides,
            channels=channels,
            pool_kernels=pool_kernels,
            device=device
        )
        frequency, time = self.input_shape
        cnn_out = cal_cnn_out(h=frequency, w=time)
        self.rnn = BiLSTM(
            input_size=cnn_out,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers
        )
        self.fc1 = nn.Linear(rnn_hidden_size * 4, 512)
        self.fc2 = nn.Linear(rnn_hidden_size * 4, 512)
        self.fc3 = nn.Linear(512, num_classes//2)
        self.fc4 = nn.Linear(512, num_classes//2)

        self.dropout = nn.Dropout(dropout, training=self.training)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        x = self.cnn(x)
        out1, out2 = self.rnn(x)
        out1 = self.dropout(F.elu(self.fc1(out1)))
        out2 = self.dropout(F.elu(self.fc2(out2)))
        out = torch.cat(self.fc3(out1), self.fc4(out2), dim=1)
        return self.sigmoid(out)
