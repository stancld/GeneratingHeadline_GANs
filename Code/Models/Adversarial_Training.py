"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London

File: Adversarial_Training.py

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

# Run model codes
exec(open('Code/Models/Attention_seq2seq.py').read())
exec(open('Code/Models/generator_training_class.py').read())
exec(open('Code/Models/CNN_text_clf.py').read())
exec(open('Code/Models/discriminator_training_class.py').read())



# Class
class AdversarialTraining:
    """
    """
    def __init__(self, generator_class, discriminator_class, optimiser_D, optimiser_G,
                 text_dictionary, embeddings, **kwargs):
        """
        :param generator:
            type:
            description:                
        :param discriminator:
            type:
            description:
        :param loss_function:
            type:
            description:
        :param optimiser:
            type:
            description:
        :param batch_size:
            type:
            description:
        :param text_dictionary:
            type:
            description:
        :param embeddings:
            type:
            description:
        """
        # Grid
        self.grid = {'max_epochs': kwargs['max_epochs'],
                     'batch_size': kwargs['batch_size'],
                     'learning_rate_D': kwargs['learning_rate_D'],
                     'learning_rate_G': kwargs['learning_rate_G'],
                     'l2_reg': kwargs['l2_reg'],
                     'clip': kwargs['clip'],    
                     'model_name': kwargs['model_name'],
                     }
        
        # Store essential parameters and objects
        #self.embeddings = nn.Embedding.from_pretrained(
        #    torch.from_numpy(embeddings), freeze=True)
        
        self.generator = generator_class.model
        self.discriminator = discriminator_class.model
        
        self.device = kwargs['device']
        self.loss_function_D = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_function_G = 1
        self.optimiser_D_ = optimiser_D
        self.optimiser_G_ = optimiser_G
        
    def training(self,
                 X_train, X_train_lengths, y_train, y_train_lengths,
                 X_val, X_val_lengths, y_val, y_val_lengths):
        """
        :param X_train:
            type:
            description:
        :param X_train_lengths:
            type:
            description:
        :param y_train:
            type:
            description:
        :param y_train_lengths:
            type:
            description:
        :param X_val:
            type:
            description:
        :param X_val_lengths:
            type:
            description:
        :param y_val:
            type:
            description:
        :param y_val_lengths:
            type:
            description:
        :param labels_train:
            type:
            description:
        :param labels_val:
            type:
            description:
        """
        # measure the time of training
        start_time = time.time()        
        # measure the time to print some output every 10 minutes
        time_1 = time.time()
        
        ### generate batches
        # training data
        (input_train, input_train_lengths,
         target_train, target_train_lengths) = self._generate_batches(padded_input = X_train,
                                                                      input_lengths = X_train_lengths,
                                                                      padded_target = y_train,
                                                                      target_lengths = y_train_lengths)
    
        # validation data
        (input_val, input_val_lengths,
         target_val, target_val_lengths) = self._generate_batches(padded_input = X_val,
                                                                  input_lengths = X_val_lengths,
                                                                  padded_target = y_val,
                                                                  target_lengths = y_val_lengths)
        
        # Save number of batches of training and validation sets for a proper computation of losses
        self.n_batches = X_train.shape[0]
        self.n_batches_val = X_val.shape[0]
        # indices for reshuffling data before running each epoch
        input_arr = np.arange(input_train.shape[0])
        
        for epoch in range(self.start_epoch, self.grid['max_epochs']):
            # run the training
            self.generator.train()
            self.discriminator.train()
            epoch_Loss_D = 0
            epoch_Loss_G = 0
            batch = 0
            
            # shuffle the data
            reshuffle = np.random.shuffle(input_arr)
            input_train, input_train_lengths = input_train[reshuffle].squeeze(0), input_train_lengths[reshuffle].squeeze(0)
            target_train, target_train_lengths = target_train[reshuffle].squeeze(0), target_train_lengths[reshuffle].squeeze(0)
            
            # Initialize optimise
            self.optimiser_D = self.optimiser_D_(self.discriminator.parameters(), lr= (0.98**epoch) * self.grid['learning_rate_D'],
                                                 weight_decay = self.grid['l2_reg'])
            self.optimiser_G = self.optimiser_G_(self.generator.parameters(), lr= (0.98**epoch) * self.grid['learning_rate_G'],
                                                 weight_decay = self.grid['l2_reg'])
            
            for input, target, seq_length_input, seq_length_target in zip(input_train,
                                                                          target_train,
                                                                          input_train_lengths,
                                                                          target_train_lengths,
                                                                          ):
                batch += 1
                # Paragraphs
                input = torch.from_numpy(
                    input[:seq_length_input.max()]
                ).long()
                # Summaries
                target = torch.from_numpy(
                target[:seq_length_target.max()]
                ).long().to(self.device)           
                                
                #####
                # (1) Update Discriminator: we maximize BCELoss given as log(D(x)) + log(1 - D(G(z)))
                # This is done in two subsequent steps according to https://github.com/soumith/ganhacks
                #####
                # create vectors of 1s and 0s representing labels of real and generated/fake summaries
                real_labels = torch.ones(self.batch_size).to(self.device)
                fake_labels = torch.zeros(self.batch_size).to(self.device)
                
                
                ## Compute log(D(x)) using batch of real examples
                self.optimiser_D.zero_grad()
                # dicriminator output
                output_D, real_labels_flatten = self.discriminator.forward(target, real_labels)
                # calculate loss function on the batch of real examples
                error_D_real = self.loss_function_D(output_D, real_labels_flatten)
                # Calculate gradient
                error_D_real.backward()
                
                ## Compute log(1 - D(G(z)))
                # Generate summaries
                output_G = self.generator.model(seq2seq_input = input, input_lengths = seq_length_input,
                                                target = target, teacher_forcing_ratio = 1,
                                                adversarial = True)
                output_G = F.log_softmax(output_G, dim = 2)
                # discriminator output D(G(z))
                output_D_G, fake_labels_flatten = self.discriminator.forward(output_G, fake_labels)
                # calculate loss function on the batch of fake examples
                error_D_fake = self.loss_function_D(output_D_G, fake_labels_flatten)
                # calculate gradient
                error_D_fake.backward()
                
                # sum gradients from computation both on real and fake examples
                error_D = error_D_real + error_D_fake
                
                # update step
                self.optimiser_D.step()
                
                # cleaning and saving
                epoch_Loss_D += error_D
            
            return epoch_Loss_D
                
                
    
    def _generate_batches(self, padded_input, input_lengths, padded_target, target_lengths):
        """
        :param input:
            type:
            description:
        :param inout_lengths:
            type:
            description:
        :param target:
            type:
            description:
        :param target_lengths:
            type:
            description:
            
        :return input_batches:
            type:
            description:
        :return input_lengths:
            type:
            description:
        :return target_batches:
            type:
            description:
        :return target_lengths:
            type:
            description:
        """
        # determine a number of batches
        n_batches = padded_input.shape[1] // self.grid['batch_size']
        self.n_batches = n_batches
        
        # Generate input and target batches
            #dimension => [total_batchs, seq_length, batch_size, embed_dim], for target embed_dim is irrelevant
                #seq_length is variable throughout the batches
        input_batches = np.array(
            np.split(padded_input[:, :(n_batches * self.grid['batch_size'])], n_batches, axis = 1)
            )
        target_batches = np.array(
            np.split(padded_target[:, :(n_batches * self.grid['batch_size'])], n_batches, axis = 1)
            )
        # Split input and target lenghts into batches as well
        input_lengths = np.array(
            np.split(input_lengths[:(n_batches * self.grid['batch_size'])], n_batches, axis = 0)
            )
        target_lengths = np.array(
            np.split(target_lengths[:(n_batches * self.grid['batch_size'])], n_batches, axis = 0)
            )
        
        """
        # trim sequences in individual batches
        for batch in range(n_batches):
            input_batches[batch] = input_batches[batch, input_lengths[batch].max():, :, :]
            target_batches[batch] = target_batches[batch, target_lengths[batch].max():, :]
        """
        # return prepared data
        return (input_batches, input_lengths,
                target_batches, target_lengths)
    
    def _embedding_layer(self, x):
        '''
        To embedd x 

        Args: 
            input x -> [N_samples,seq_len]
            output -> [N_samples,seq_len,embedding_dimension]
        '''
        return self.embedding_layer(x).float()
    
    def load(self):
        """
        :param name_path:
            type:
            description:
        """
        try:
            self.start_epoch = 0
        except:
            self.start_epoch = 0
        
        
        