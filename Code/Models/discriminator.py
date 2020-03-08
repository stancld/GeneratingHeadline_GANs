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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Discriminator_utility():
    @staticmethod
    def show_parameter():
        print(
            '''
            This is a discriminator model for text classification:

            1. CNN is used

            2. Embedding should be done before input into this model

            3. default loss function is BCEWithLogitsLoss()

            Instatiate:
            Discriminator_utility.instan_things
                    in which you should define the following dictionary parameters
            e.g.
            param = {'max_epochs':64,
                    'learning_rate':1e-3,
                    'batch_size':1,
                    'seq_len': 20,                   # length of your summary
                    'embed_dim': 100,
                    'drop_out': 0,
                    'kernel_num': 5,                 # number of your feature map
                    'in_channel': 1,                 # for text classification should be one
                    # how many conv net are used in parallel in text classification
                    'parallel_layer':3,
                    'device':device}

            Training:
            Discriminator_utility.training

            Evaluation:

            Prediction:

          '''
        )

    @staticmethod
    def instan_things(**kwargs):
        grid = {'max_epochs': kwargs['max_epochs'],
                'learning_rate': kwargs['learning_rate'],
                'batch_size': kwargs['batch_size'],
                'seq_len': kwargs['seq_len'],
                'embed_dim': kwargs['embed_dim'],
                'drop_out': kwargs['drop_out'],
                'kernel_num': kwargs['kernel_num'],
                # for text this should be one
                'in_channel': kwargs['in_channel'],
                # the conv net are used in parallel in text classification
                'parallel_layer': kwargs['parallel_layer']
                }
        device = kwargs['device']

        model = _Discriminator(**grid).to(device)

        optimiser = optim.Adam(model.parameters(), lr=grid['learning_rate'])

        # pos_weight = kwargs['pos_weight']
        # All weights are equal to 1 and we have 2 classes
        lossfunction = nn.CrossEntropyLoss().to(device)

        return (grid, model, optimiser, lossfunction)

    @staticmethod
    def training(model, training_generator, optimiser, lossfunction):
        '''
        Args:
          train_generator: iterable object creating X_train, y_train
          X_train [seq_len, batch size, embedding dimension]
          y_train [batch size,] ; boolean, long type, 1d tensor
        '''
        model.train()
        epoch_loss = 0
        for local_batch, local_labels in training_generator:

            # print('input are:')
            # print(local_batch.size())
            # print(local_labels.size())

            local_batch, local_labels = local_batch.to(
                device), local_labels.flatten().long().to(device)

            optimiser.zero_grad()
            local_output = model(local_batch)

            # print('output are:')
            # print(local_output.size())
            # print(local_labels.size())

            loss = lossfunction(local_output, local_labels)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        return epoch_loss

    @staticmethod
    def evaluation(model, test_generator, lossfunction):
        '''
        Args:
          train_generator: iterable object creating X_train, y_train
          X_train [seq_len, batch size, embedding dimension]
          y_train [batch size,] ; boolean, long type, 1d tensor
        '''
        model.eval()
        epoch_loss = 0
        for local_batch, local_labels in test_generator:

            # print('input are:')
            # print(local_batch.size())
            # print(local_labels.size())

            local_batch, local_labels = local_batch.to(
                device), local_labels.flatten().long().to(device)

            local_output = model(local_batch)

            # print('output are:')
            # print(local_output.size())
            # print(local_labels.size())

            loss = lossfunction(local_output, local_labels)
            epoch_loss += loss.item()
        return epoch_loss

    @staticmethod
    def run_epochs(grid, model):
        '''
        to make a epoch train evaluate circle
        '''
        pass


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
