'''DLA in PyTorch.
Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import numpy as np
import torch
from torch import optim
import torch.nn as nn

from AdasOptimizer.adasopt_pytorch import Adas


class NNet(nn.Module):
    def __init__(self):
        self.name = 'NNet'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print('Using GPU')
        else:
            print('Using CPU')
        self.train_losses = []

    def forward(self, x1, x2):
        pass

    def predict(self, input_1, input_2):
        input_1 = torch.FloatTensor(np.array(input_1)).to(self.device).detach()
        input_2 = torch.FloatTensor(np.array(input_2)).to(self.device).detach()
        # input_1 = input_1.view(-1, input_1.shape[0], input_1.shape[1], input_1.shape[2])
        # input_2 = input_2.view(-1, 8)
        output = self.forward(input_1, input_2)
        return output.cpu().data.numpy().flatten()

    def set_loss_function(self, loss):
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "bce":
            self.loss = nn.BCELoss()
        elif loss == "bce_logits":
            self.loss = nn.BCEWithLogitsLoss()
        elif loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "smooth_l1":
            self.loss = nn.SmoothL1Loss()
        elif loss == "soft_margin":
            self.loss = nn.SoftMarginLoss()
        elif loss == "mlsm":
            self.loss = nn.MultiLabelSoftMarginLoss()
        else:
            raise ValueError("Loss function not found")

    def set_optimizer(self, optimizer, lr):
        self.lr = lr
        if optimizer == "sgd":
            # Tối ưu theo gradient descent thuần túy
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adamax":
            self.optimizer = optim.Adamax(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            # Phương pháp Adadelta có lr update
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)
        elif optimizer == "adagrad":
            # Phương pháp Adagrad chỉ cập nhật lr ko nhớ
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == "nadam":
            self.optimizer = optim.NAdam(self.parameters(), lr=lr)

    def reset_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()