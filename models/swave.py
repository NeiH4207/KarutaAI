# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.nnet import NNet

from src.utils import overlap_and_add
from src.utils import capture_init


class MulCatBlock(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0, bidirectional=False):
        super(MulCatBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = nn.LSTM(input_size, hidden_size, 1, dropout=dropout,
                           batch_first=True, bidirectional=bidirectional)
        self.rnn_proj = nn.Linear(hidden_size * self.num_direction, input_size)

        self.gate_rnn = nn.LSTM(input_size, hidden_size, num_layers=1,
                                batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.gate_rnn_proj = nn.Linear(
            hidden_size * self.num_direction, input_size)

        self.block_projection = nn.Linear(input_size * 2, input_size)

    def forward(self, input):
        output = input
        # run rnn module
        rnn_output, _ = self.rnn(output)
        rnn_output = self.rnn_proj(rnn_output.contiguous(
        ).view(-1, rnn_output.shape[2])).view(output.shape).contiguous()
        # run gate rnn module
        gate_rnn_output, _ = self.gate_rnn(output)
        gate_rnn_output = self.gate_rnn_proj(gate_rnn_output.contiguous(
        ).view(-1, gate_rnn_output.shape[2])).view(output.shape).contiguous()
        # apply gated rnn
        gated_output = torch.mul(rnn_output, gate_rnn_output)
        gated_output = torch.cat([gated_output, output], 2)
        gated_output = self.block_projection(
            gated_output.contiguous().view(-1, gated_output.shape[2])).view(output.shape)
        return gated_output


class ByPass(nn.Module):
    def __init__(self):
        super(ByPass, self).__init__()

    def forward(self, input):
        return input


class DPMulCat(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 dropout=0, num_layers=1, bidirectional=True, input_normalize=False):
        super(DPMulCat, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.in_norm = input_normalize
        self.num_layers = num_layers

        self.rows_grnn = nn.ModuleList([])
        self.cols_grnn = nn.ModuleList([])
        self.rows_normalization = nn.ModuleList([])
        self.cols_normalization = nn.ModuleList([])

        # create the dual path pipeline
        self.rows_grnn = MulCatBlock(
            input_size, hidden_size, dropout, bidirectional=bidirectional)
        self.cols_grnn = MulCatBlock(
            input_size, hidden_size, dropout, bidirectional=bidirectional)
        if self.in_norm:
            self.rows_normalization = nn.GroupNor(1, input_size, eps=1e-8)
            self.cols_normalization = nn.GroupNor(1, input_size, eps=1e-8)
        else:
            # used to disable normalization
            self.rows_normalization = ByPass()
            self.cols_normalization = ByPass()

        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        batch_size, _, d1, d2 = input.shape
        output = input
        row_input = output.permute(0, 3, 2, 1).contiguous().view(
            batch_size * d2, d1, -1)
        row_output = self.rows_grnn(row_input)
        row_output = row_output.view(
            batch_size, d2, d1, -1).permute(0, 3, 2, 1).contiguous()
        row_output = self.rows_normalization(row_output)
        col_input = output.permute(0, 2, 3, 1).contiguous().view(
            batch_size * d1, d2, -1)
        col_output = self.cols_grnn(col_input)
        col_output = col_output.view(
            batch_size, d1, d2, -1).permute(0, 3, 1, 2).contiguous()
        col_output = self.cols_normalization(col_output).contiguous()
        return self.output(output)


class Separator(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, 
                 layer=4, segment_size=100, input_normalize=False, bidirectional=True):
        super(Separator, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.input_normalize = input_normalize

        self.rnn_model = DPMulCat(self.feature_dim, self.hidden_dim,
                                  output_dim, num_layers=layer, 
                                  bidirectional=bidirectional, 
                                  input_normalize=input_normalize)

    # ======================================= #
    # The following code block was borrowed and modified from https://github.com/yluo42/TAC
    # ================ BEGIN ================ #
    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        rest = segment_size - (segment_stride + seq_len %
                               segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)
                           ).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(
            batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def create_chuncks(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size,
                                                                    dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(
            batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(
            batch_size, dim, -1, segment_size).transpose(2, 3)
        return segments.contiguous(), rest

    def merge_chuncks(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(
            batch_size, dim, -1, segment_size*2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(
            batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(
            batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]
        return output.contiguous()  # B, N, T
    # ================= END ================= #

    def forward(self, input):
        # create chunks
        enc_segments, enc_rest = self.create_chuncks(
            input, self.segment_size)
        return self.rnn_model(enc_segments)


class SWave(NNet):
    def __init__(self, n_classes):
        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.pooling = nn.AdaptiveAvgPool2d((8, 8)) # extended
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        # setting 50% overlap
        self.conv1 = nn.Conv1d(
            1, N, kernel_size=L, stride=L // 2, bias=False)
        self.maxpool1 = nn.MaxPool1d(3, stride=2)
        self.conv2 = nn.Conv1d(
            N, N * 2, kernel_size=L, stride=L // 2, bias=False)
        self.maxpool2 = nn.MaxPool1d(3, stride=2)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.maxpool1(self.conv1(mixture)))
        mixture_w = F.relu(self.maxpool2(self.conv2(mixture_w)))
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, L):
        super(Decoder, self).__init__()
        self.L = L

    def forward(self, est_source):
        est_source = torch.transpose(est_source, 2, 3)
        est_source = nn.AvgPool2d((1, self.L))(est_source)
        est_source = overlap_and_add(est_source, self.L//2)

        return est_source