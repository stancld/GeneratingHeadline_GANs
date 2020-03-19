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
from rouge import Rouge

# Run model codes
exec(open('Code/Models/Attention_seq2seq.py').read())
exec(open('Code/Models/generator_training_class.py').read())
exec(open('Code/Models/CNN_text_clf.py').read())
exec(open('Code/Models/discriminator_training_class.py').read())



# Class
class AdversarialTraining:
    """
    """
    def __init__(self, generator_class, discriminator_class, optimiser_D, optimiser_G, **kwargs):
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
                     'G_multiple': kwargs['G_multiple'],
                     'l2_reg': kwargs['l2_reg'],
                     'clip': kwargs['clip'],    
                     'model_name': kwargs['model_name'],
                     'text_dictionary': kwargs['text_dictionary'],
                     'headline_dictionary': kwargs['headline_dictionary'],
                     'noise_std': kwargs['noise_std']
                     }
        
        # Store essential parameters and objects
        #self.embeddings = nn.Embedding.from_pretrained(
        #    torch.from_numpy(embeddings), freeze=True)
        
        self.generator = generator_class
        self.discriminator = discriminator_class
        
        self.device = kwargs['device']
        self.loss_function_D = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_function_G = nn.CrossEntropyLoss().to(self.device)
        self.optimiser_D_ = optimiser_D
        self.optimiser_G_ = optimiser_G
        
        self.pad_idx = self.grid['headline_dictionary'].word2index['<pad>']
        self.eos_idx = self.grid['headline_dictionary'].word2index['eos']
        
        self.rouge = Rouge()
        
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
        self.n_batches = input_train.shape[0]
        self.n_batches_val = input_val.shape[0]
        # indices for reshuffling data before running each epoch
        input_arr = np.arange(input_train.shape[0])
        
        for epoch in range(self.start_epoch, self.grid['max_epochs']):
            # run the training
            self.generator.model.train()
            self.discriminator.model.train()
            epoch_Loss_D = 0
            epoch_Loss_G = 0
            batch = 0
            
            # shuffle the data
            reshuffle = np.random.shuffle(input_arr)
            input_train, input_train_lengths = input_train[reshuffle].squeeze(0), input_train_lengths[reshuffle].squeeze(0)
            target_train, target_train_lengths = target_train[reshuffle].squeeze(0), target_train_lengths[reshuffle].squeeze(0)
            
            # Initialize optimise
            self.optimiser_D = self.optimiser_D_(self.discriminator.model.parameters(), lr= (0.98**epoch) * self.grid['learning_rate_D'],
                                                 weight_decay = self.grid['l2_reg'])
            self.optimiser_G = self.optimiser_G_(self.generator.model.parameters(), lr= (0.98**epoch) * self.grid['learning_rate_G'],
                                                 weight_decay = 0.0)
            
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
                real_labels = torch.ones(self.grid['batch_size']).to(self.device)
                fake_labels = torch.zeros(self.grid['batch_size']).to(self.device)
                
                
                ## Compute log(D(x)) using batch of real examples
                self.optimiser_D.zero_grad()
                # dicriminator output
                output_D, real_labels_flatten = self.discriminator.forward(target.permute(1,0), real_labels) #discriminator needs transpose input
                # calculate loss function on the batch of real examples
                error_D_real = self.loss_function_D(output_D, real_labels_flatten)
                # Calculate gradient
                error_D_real.backward(retain_graph = True)
                
                ## Compute log(1 - D(G(z)))
                # Generate summaries
                output_G = self.generator.model(seq2seq_input = input, input_lengths = seq_length_input,
                                                target = target, teacher_forcing_ratio = 1,
                                                adversarial = True, noise_std = self.grid['noise_std'])
                # discriminator output D(G(z))
                output_D_G, fake_labels_flatten = self.discriminator.forward(output_G.argmax(dim = 2).long().permute(1,0), fake_labels) #discriminator needs transpose input
                # calculate loss function on the batch of fake examples
                error_D_fake = self.loss_function_D(output_D_G, fake_labels_flatten)
                # calculate gradient
                error_D_fake.backward(retain_graph = True)
                
                # sum gradients from computation both on real and fake examples
                error_D = (error_D_real + error_D_fake) / 2
                
                # update step
                self.optimiser_D.step()
                
                # cleaning and saving
                epoch_Loss_D += ( (error_D - epoch_Loss_D) / batch )
                print(f'GAN Loss = {epoch_Loss_D:.3f}')
                
                #####
                # (2) Update Generator: we maximize log(D(G(z)))
                # 
                #####
                for _ in range(self.grid['G_multiple']):
                    self.optimiser_G.zero_grad()
                    if _ != 0:
                        # Generate summaries
                        output_G = self.generator.model(seq2seq_input = input, input_lengths = seq_length_input,
                                                        target = target, teacher_forcing_ratio = 1,
                                                        adversarial = True, noise_std = self.grid['noise_std'])
                    # FORWARD pass with updated discriminator
                    output_D, real_labels_flatten = self.discriminator.forward(output_G.argmax(dim = 2).long().permute(1,0), real_labels)
                    # Compute loss function
                    # output_G = F.log_softmax(output_G, dim = 2) # not necessary in for cross-entropy loss
                    
                    # Pack output and target padded sequence
                    ## Determine a length of output sequence based on the first occurrence of <eos>
                    seq_length_output = (output_G.argmax(2) == self.grid['text_dictionary'].word2index['eos']).int().argmax(0).cpu().numpy()
                    seq_length_output += 1
                                        
                    # determine seq_length for computation of loss function based on max(seq_lenth_target, seq_length_output)
                    seq_length_loss = np.array(
                        (seq_length_output, seq_length_target)
                        ).max(0)
                    
                    output_G = nn.utils.rnn.pack_padded_sequence(output_G,
                                                               lengths = seq_length_loss,
                                                               batch_first = False,
                                                               enforce_sorted = False).to(self.device)
                    
                    target_padded = nn.utils.rnn.pack_padded_sequence(target,
                                                               lengths = seq_length_loss,
                                                               batch_first = False,
                                                               enforce_sorted = False).to(self.device)
                    
                    # Compute loss
                    error_G_2 = self.loss_function_G(output_G[0], target_padded[0])
                    error_G_1 = self.loss_function_D(output_D_G, real_labels_flatten)
                    error_G = error_G_1 * error_G_2
                    
                    # Calculate gradient
                    error_G.backward(retain_graph = True)
                    # Update step
                    self.optimiser_G.step()
                    # cleaning
                    del output_G, target_padded
                    torch.cuda.empty_cache()
                print(f'Generator loss: {error_G:.3f}')
                
                #### MEASUREMENT ####
                if batch % 20 == 0:
                    self.generator.model.eval()
                    self.discriminator.model.eval()
                    val_batch = 0
                    val_loss = 0
                    self.rouge1, self.rouge2, self.rougeL = 0, 0, 0
                    outputs_true = 0
                    for input, target, seq_length_input, seq_length_target in zip(input_val,
                                                                              target_val,
                                                                              input_val_lengths,
                                                                              target_val_lengths,
                                                                              ):
                        val_batch += 1
                        # Paragraphs
                        input = torch.from_numpy(
                            input[:seq_length_input.max()]
                        ).long()
                        # Summaries
                        target = torch.from_numpy(
                        target[:seq_length_target.max()]
                        ).long().to(self.device)           
                        
                        # Eventually we are mainly interested in the generator performance measured by ROUGE metrics and fooling discriminator (may be measured by accuracy)
                        ## GENERATOR perfrormance
                        output_G = self.generator.model(seq2seq_input = input, input_lengths = seq_length_input,
                                                        target = target, teacher_forcing_ratio = 0,
                                                        adversarial = False, noise_std = 0)
                        
                        val_loss += self.validation_loss_eval(output_G, target, seq_length_target)
                        
                        hypotheses = output_G.argmax(dim = 2).permute(1,0).cpu().numpy()
                        hypotheses = [' '.join([self.grid['headline_dictionary'].index2word[index] for index in hypothesis if ( index != self.pad_idx) & (index != self.eos_idx)][1:]) for hypothesis in hypotheses]
                        references = [' '.join([self.grid['headline_dictionary'].index2word[index] for index in ref if ( index != self.pad_idx) & (index != self.eos_idx)][1:]) for ref in target.permute(1,0).cpu().numpy()]
                        ROUGE = [self.rouge_get_scores(hyp, ref) for hyp, ref in zip(hypotheses, references)]
                        self.rouge1 += ( (np.array([x[0]['rouge-1']['f'] for x in ROUGE if x != 'drop']).mean() - self.rouge1) / val_batch )
                        self.rouge2 += ( (np.array([x[0]['rouge-2']['f'] for x in ROUGE if x != 'drop']).mean() - self.rouge2) / val_batch )
                        self.rougeL += ( (np.array([x[0]['rouge-l']['f'] for x in ROUGE if x != 'drop']).mean() - self.rougeL) / val_batch )
                        
                        ## DISCRIMINATOR performance
                        output_D, real_labels_flatten = self.discriminator.forward(target.permute(1,0), real_labels) #discriminator needs transpose input
                        outpud_D = output_D.detach().cpu().numpy()
                        output_labels = np.array(
                            [1 if x>=0 else 0 for x in outpud_D]
                            )
                        outputs_true += sum(output_labels == real_labels_flatten.cpu().numpy())
                        
                        output_G = F.log_softmax(output_G, dim = 2).argmax(dim = 2).long()
                        output_D_G, fake_labels_flatten = self.discriminator.forward(output_G.permute(1,0), fake_labels) #discriminator needs transpose input
                        outpud_D_G = output_D_G.detach().cpu().numpy()
                        output_labels = np.array(
                            [1 if x>=0 else 0 for x in outpud_D_G]
                            )
                        outputs_true += sum(output_labels == fake_labels_flatten.cpu().numpy())
                        # cleaning
                        del output_D, output_D_G
                        torch.cuda.empty_cache()
                        
                    acc = 100 * float(outputs_true) / (2*self.n_batches_val*self.grid['batch_size'])
                    val_loss /= val_batch
                    
                    # Eventually we are mainly interested in the generator performance measured by ROUGE metrics and fooling discriminator (may be measured by accuracy)
                    print(f'Epoch: {epoch+1:.0f}')
                    print(f'Generator performance after {100*batch/self.n_batches:.2f} % of examples.')
                    print(f'ROUGE-1 = {100*self.rouge1:.2f} | ROUGE-2 = {100*self.rouge2:.2f} | ROUGE-l = {100*self.rougeL:.2f} | Cross-Entropy = {val_loss:.3f} | Discriminator accuracy = {acc:.2f} %.')
                    self.generator.model.train()
                    self.discriminator.model.train()
            
    def rouge_get_scores(self, hyp, ref):
      """
      """
      try:
        return self.rouge.get_scores(hyp, ref)
      except:
        return "drop"
    
    def validation_loss_eval(self, output_G, target, seq_length_target):
        """
        Function handling the course of computing Cross-Entropy loss.
        """            
        # Pack output and target padded sequence
        ## Determine a length of output sequence based on the first occurrence of <eos>
        seq_length_output = (output_G.argmax(2) == self.grid['text_dictionary'].word2index['eos']).int().argmax(0).cpu().numpy()
        seq_length_output += 1
                            
        # determine seq_length for computation of loss function based on max(seq_lenth_target, seq_length_output)
        seq_length_loss = np.array(
            (seq_length_output, seq_length_target)
            ).max(0)
        
        output_G = nn.utils.rnn.pack_padded_sequence(output_G,
                                                   lengths = seq_length_loss,
                                                   batch_first = False,
                                                   enforce_sorted = False).to(self.device)
        
        target = nn.utils.rnn.pack_padded_sequence(target,
                                                   lengths = seq_length_loss,
                                                   batch_first = False,
                                                   enforce_sorted = False).to(self.device)
        # loss function
        loss = 0
        loss += self.loss_function_G(output_G[0], target[0]).item()
        # cleaning
        del output_G, target, seq_length_output, seq_length_loss
        torch.cuda.empty_cache()
        return loss
    
    def generate_summaries(self, input_val, input_val_lengths, target_val, target_val_lengths):
        """
        :param input_val:
            type:
            description:
        :param input_val_lengths:
            type:
            description:
        :param target_val:
            type:
            description:
        :param target_val_lengths:
            type:
            description:
                
        :return sumaries:
            type:
            description:
        """
        self.generator.model.eval()
        
        (input_val, input_val_lengths,
        target_val, target_val_lengths) = self._generate_batches(padded_input = input_val,
                                                                      input_lengths = input_val_lengths,
                                                                      padded_target = target_val,
                                                                      target_lengths = target_val_lengths)
        OUTPUT = []
        for input, target, seq_length_input, seq_length_target in zip(input_val,
                                                                      target_val,
                                                                      input_val_lengths,
                                                                      target_val_lengths
                                                                      ):
            ## FORWARD PASS
            # Prepare RNN-edible input - i.e. pack padded sequence
            # trim input, target
            input = torch.from_numpy(
                input[:seq_length_input.max()]
                ).long()
            target = torch.from_numpy(
                target[:seq_length_target.max()]
                ).long().to(self.device)
                        
            output = self.generator.model(seq2seq_input = input, input_lengths = seq_length_input,
                                          target = target, teacher_forcing_ratio = 0,
                                          adversarial = False, noise_std = 0)
            del input, target
            torch.cuda.empty_cache()
            
            OUTPUT.append(
                output.argmax(dim = 2).cpu().numpy()
                )
        
        return np.array(OUTPUT)
    
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
        
        
        