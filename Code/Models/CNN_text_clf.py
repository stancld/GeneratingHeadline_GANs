import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import torch.optim as optim
import pandas as pd
import sklearn
import random


class _CNN_text_clf(nn.Module):
    def __init__(self, **kwargs):
        super(_CNN_text_clf, self).__init__()
        '''
        A CNN to do text classification, paper: Convolutional Neural Networks for Sentence Classification
                          implementation on https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        '''

        # kwargs:
        self.seq_len = kwargs['seq_len']
        self.batch_size = kwargs['batch_size']
        self.embed_dim = kwargs['embed_dim']
        self.C_out = kwargs['kernel_num']
        self.C_in = kwargs['in_channel']
        # number of parallel layer in this network
        self.parallel_layer = kwargs['parallel_layer']

        self.conv1 = nn.Conv2d(in_channels=self.C_in,
                               out_channels=self.C_out,
                               kernel_size=(3, self.embed_dim)
                               )
        self.conv2 = nn.Conv2d(in_channels=self.C_in,
                               out_channels=self.C_out,
                               kernel_size=(4, self.embed_dim)
                               )
        self.conv3 = nn.Conv2d(in_channels=self.C_in,
                               out_channels=self.C_out,
                               kernel_size=(5, self.embed_dim)
                               )
        self.fc = nn.Linear(in_features=self.parallel_layer * self.C_out,
                            out_features=2)
        self.drop_out = nn.Dropout(kwargs['drop_out'])
        self.device = kwargs['device']

    def forward(self, embedded):
        '''
        Args:
          embedded type -> torch.Tensor

          embedded dim -> [batch_size,seq_len,embed_dim] 

          desirable input to CNN: (N, C_in, H, W) = [batch_size,in_channel,seq_len,embed_dim]

          output -> [batch_size,] boolean judgement
        '''

        x = embedded.unsqueeze(1)
        # -> [batch_size,in_channel,seq_len,embed_dim]

        x1 = F.relu(self.conv1(x)).squeeze(3)
        # -> [batch_size,out_channel,seq_len]

        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(
            2)  # [batch_size,out_channel]

        # [batch_size,out_channel,seq_len]
        x2 = F.relu(self.conv1(x)).squeeze(3)

        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(
            2)  # [batch_size,out_channel]

        # [batch_size,out_channel,seq_len]
        x3 = F.relu(self.conv1(x)).squeeze(3)

        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(
            2)  # [batch_size,out_channel]

        # concat the parallel layers together!
        x = self.drop_out(torch.cat((x1, x2, x3), dim=1))

        # nn.BCEWithLogitsLoss will apply sigmoid activation internally
        x = self.fc(x)
        return x


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # toy example
    all_grid = {'max_epochs': 64,
                'seq_len': 10,
                'learning_rate': 1e-3,
                'batch_size': 20,
                'embed_dim': 100,
                'drop_out': 0,
                'kernel_num': 5,
                'in_channel': 1,  # for text this should be one
                'parallel_layer': 3,
                'device': device
                }
    # grid, model, optimiser, lossfunction = Discriminator_utility.instan_things(
    #     **all_grid)

    # to make a desirable toy input
    # x = torch.rand(grid['seq_len'],
    #                grid['batch_size'], grid['embed_dim']).unsqueeze(0)

    # y = torch.ones(grid['batch_size'], 1).unsqueeze(0)
    # generator = zip(x, y)
    # for i, j in generator:
    #     print(i.size())
    #     print(j.size())

    # print(Discriminator_utility.training(
    #     model, generator, optimiser, lossfunction))