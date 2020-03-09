"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London

File: CNN_discriminator.py

Description of our model:

Collaborators:
    - Daniel Stancl
    - Dorota Jagnesakova
    - Guoliang HE
    - Zakhar Borok`
"""
# ----- Settings -----
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


# ----- Settings -----



class _Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(_Discriminator, self).__init__()
        '''
        A CNN to do text classification, paper: Convolutional Neural Networks for Sentence Classification
                          implementation on https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        Args:
          input:
            dis_input [summary sequence length, batch size, embedding dimension]  i.e. the summary
          output -> 
            judgement [batch size,2] ; 2 represent probability
        '''
        #self.seq_len = kwargs['seq_len']
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

    def forward(self, x):
        '''
        Args:
          raw input: x [seq_len, batch_size, embed_dim] 

          desirable input to CNN: (N, C_in, H, W) = [batch_size,in_channel,seq_len,embed_dim]

          output -> [batch_size,] boolean judgement
        '''
        x = x.transpose(0, 1).unsqueeze(1)  # format input to CNN

        # [batch_size,out_channel,seq_len]

        x1 = F.relu(self.conv1(x)).squeeze(3)

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
    grid, model, optimiser, lossfunction = Discriminator_utility.instan_things(
        **all_grid)

    # to make a desirable toy input
    x = torch.rand(grid['seq_len'],
                   grid['batch_size'], grid['embed_dim']).unsqueeze(0)
    y = torch.ones(grid['batch_size'], 1).unsqueeze(0)
    generator = zip(x, y)
    # for i, j in generator:
    #     print(i.size())
    #     print(j.size())

    # 64 classes, batch size = 10
    # target = torch.ones([10, 64], dtype=torch.float32)
    # output = torch.full([10, 64], 0.999)  # A prediction (logit)
    # print('out')
    # print(output.size())
    # print(target.size())
    # criterion = torch.nn.BCEWithLogitsLoss()
    # loss = criterion(output, target)  # -log(sigmoid(0.999))
    # print(loss.item())

    print(Discriminator_utility.training(
        model, generator, optimiser, lossfunction))
