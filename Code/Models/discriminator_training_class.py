"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London

File: CNN_text_clf.py

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

exec(open('Code/Models/CNN_text_clf.py').read())


class Discriminator_utility():

    def __init__(self, embedding, **kwargs):
        self.grid = {'max_epochs': kwargs['max_epochs'],
                     'learning_rate': kwargs['learning_rate'],
                     'batch_size': kwargs['batch_size'],
                     'embed_dim': kwargs['embed_dim'],
                     'drop_out': kwargs['drop_out'],
                     'kernel_num': kwargs['kernel_num'],
                     # for text this should be one
                     'in_channel': kwargs['in_channel'],
                     # the conv net are used in parallel in text classification
                     'parallel_layer': kwargs['parallel_layer'],
                     'model_name': kwargs['model_name'],
                     'device': kwargs['device']
                     }
        self.device = kwargs['device']

        self.model = _CNN_text_clf(**self.grid).to(self.device)

        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding), freeze=True)

        self.optimiser = optim.Adam(
            self.model.parameters(), lr=self.grid['learning_rate'])

        # pos_weight = kwargs['pos_weight']
        # All weights are equal to 1 and we have 2 classes
        self.lossfunction = nn.CrossEntropyLoss().to(self.device)

    def run_epochs(self, X_train, y_train, X_test, y_test):
        '''
        Args:
            input:
                X_train [N_samples,seq_len] word indices type long()
                y_train [N_samples,] -> boolean tensor 

                X_test [N_samples,seq_len] word indices type long()
                y_test [N_samples,] -> boolean tensor 
        '''
        best_valid_loss = float('inf')

        for epoch in range(self.grid['max_epochs']):

            train_loss = self.training(X_train, y_train)
            valid_loss = self.evaluation(X_test, y_test)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                m = copy.deepcopy(self.model)
                print(f'Epoch: {epoch+1}:')
                print(f'Train Loss: {train_loss:.3f}')
                print(f'Validation Loss: {valid_loss:.3f}')

        return best_valid_loss, m

    def training(self, X_train, y_train):
        '''
        Args:
            X_train -> [N_samples,seq_len]; word index array
            y_train -> [N_samples,]; boolean, long type, 1d tensor

            output from _generate_batches:
                local_batch  -> [batch_size, seq_len]
                local_labels -> [batch_size,]; boolean
        '''
        self.model.train()
        epoch_loss = 0
        for local_batch, local_labels in self._generate_batches(X_train, y_train):

            # pass through embedding layer
            local_batch_embedded = self._embedding_layer(local_batch)
            # -> [batch_size,seq_len,emb_dim]

            local_batch, local_labels = local_batch.to(
                self.device), local_labels.flatten().to(self.device)

            # print('input are:')
            # print(local_batch.size())
            # print(local_labels.size())

            self.optimiser.zero_grad()
            local_output = self.model(local_batch_embedded)

            # print('output are:')
            # print(local_output.size())
            # print(local_labels.size())

            loss = self.lossfunction(local_output, local_labels)
            loss.backward()
            self.optimiser.step()
            epoch_loss += loss.item()
        return epoch_loss

    def evaluation(self, X_test, y_test):
        '''
        Args:
            X_test -> [N_samples,seq_len]; word index array
            y_test -> [N_samples,]; boolean, long type, 1d tensor

            output batches:
                local_batch  -> [batch_size, seq_len]
                local_labels -> [batch_size,] boolean
        '''
        self.model.eval()
        epoch_loss = 0
        for local_batch, local_labels in self._generate_batches(X_test, y_test):

            # pass through embedding layer
            local_batch_embedded = self._embedding_layer(local_batch)
            # -> [batch_size,seq_len,emb_dim]

            local_batch, local_labels = local_batch.to(
                self.device), local_labels.flatten().to(self.device)

            local_output = self.model(local_batch_embedded)

            loss = self.lossfunction(local_output, local_labels)
            epoch_loss += loss.item()
        return epoch_loss

    def _embedding_layer(self, x):
        '''
        To embedd x 

        Args: 
            input x -> [N_samples,seq_len]
            output -> [N_samples,seq_len,embedding_dimension]
        '''

        return self.embedding_layer(x).float()

    def _generate_batches(self, summary_bag, label):
        """
        This function to generate batch for training and evaluation 

        Args:
            data type numpy.py or torch.Tensor

            input:
                bag of summary: [N_summary, seq_len]  seq_len is a list of token index
                summary label: [N_summary,] boolean; 

            output ->:
                dataloader -> (x,y) each iteration will give one local batch 
                where:
                x ->[batch_size, seq_len]
                y ->[batch_size,] boolean
        """
        if torch.is_tensor(summary_bag):
            pass
        else:
            summary_bag = torch.from_numpy(summary_bag).long()
            label = torch.from_numpy(label).long()

        label = label.view(-1, 1)  # to 1d column
        l = len(summary_bag)
        for batch in range(0, l, self.grid['batch_size']):
            yield (summary_bag[batch:min(batch + self.grid['batch_size'], l)], label[batch:min(batch + self.grid['batch_size'], l)])

    def save(self):
        """
        Args:

        """
        torch.save(self.model.state_dict(),
                   "../data/Results/discriminator_{}.pth".format(self.grid['model_name']))

    def load(self):
        """
        Args:

        """
        try:
            self.model.load_state_dict(torch.load(
                "../data/Results/discriminator_{}.pth".format(self.grid['model_name'])))
            self.model.eval()
        except:
            pass

    @staticmethod
    def show_parameter():
        print(
            '''
            This is a discriminator model for text classification:

            1. CNN is used

            2. Embedding should be done before input into this model

            3. default loss function is BCEWithLogitsLoss()

            #####
            Instatiate:
            __init__(self,embedding,**param)
                    in which you should define the following dictionary parameters
            e.g.
            embedding = your pre_trained embedding on CPU; type -> np.array

            param = {'max_epochs':64,
                    'learning_rate':1e-3,
                    'batch_size':1,
                    'embed_dim': 100,
                    'drop_out': 0,
                    'kernel_num': 5,                 # number of your feature map
                    'in_channel': 1,                 # for text classification should be one
                    # how many conv net are used in parallel in text classification
                    'parallel_layer':3,
                    'model_name': 'discriminator']
                    'device':device}

            #####
            Iterate epoch: run_epoch(X_train, y_train, X_test, y_test)

            this function automates epochs training and evaluation with X_train, y_train, X_test, y_test

            Arg:
                input:
                X_train [N_samples,seq_len] -> word indices type long()
                y_train [N_samples,]        -> boolean tensor 

                X_test [N_samples,seq_len]  -> word indices type long()
                y_test [N_samples,]         -> boolean tensor 
                
                output:
                best_valid_loss
                m -> model that gives the best_valid_loss

            #####
            Training: training(X_train, y_train)

            this function train the model with X_train, y_train

            Args:
                X_train [N_samples,seq_len]
                y_train [N_samples,] ; boolean, long type, 1d tensor

            #####
            Evaluation: evaluation(X_test, y_test)

            this function evaluation the model with X_test, y_test

            Args:
                X_test [N_samples,seq_len]
                y_test [N_samples,] ; boolean, long type, 1d tensor


            #####
            Prediction: self.model(embedded)

            this function classifies the embedded sequence 
            
            Args:

            embedded type -> torch.Tensor
            embedded dim -> [batch_size,seq_len,embed_dim] 
          '''
        )
