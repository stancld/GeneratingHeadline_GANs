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
    Class utuilizing generator and discriminator training utilities 
    and applied them to adversarial training defined in the paper.
    """
    def __init__(self, generator_class, discriminator_class, optimiser_D, optimiser_G, **kwargs):
        """
        :param generator_class:
            type: Class
            description: Initialized class for a training of the generator               
        :param discriminator:
            type: Class
            description: Initialized class for a training of the discriminator
        :param optimiser_D:
            type: torch.optim
            description: Optimizer class used for a training of the discriminator
        :param optimiser_G:
            type: torch.optim
            description: Optimizer class used for a training of the generator
        """
        # Grid
        self.grid = {'max_epochs': kwargs['max_epochs'], # maximum number of epochs used for adversarial training
                     'batch_size': kwargs['batch_size'], # batch size used for adversarial training. This must comply with the batch size used for the generator pretraining
                     'learning_rate_D': kwargs['learning_rate_D'], # learning rate for the discriminator's optimizer
                     'learning_rate_G': kwargs['learning_rate_G'], # learning rate for the generator's optimizer
                     'G_multiple': kwargs['G_multiple'], # number of update steps done for the generator for each batch
                     'l2_reg': kwargs['l2_reg'], # L2 penalty parameter used for regularization of the discriminator
                     'clip': kwargs['clip'], # parameter for the maximum absolute value of gradient of the generator
                     'model_name': kwargs['model_name'], # model name; string
                     'text_dictionary': kwargs['text_dictionary'], # dictionary for the paragraohs/articles
                     'headline_dictionary': kwargs['headline_dictionary'], # dictionary for the headliens/summaries
                     'noise_std': kwargs['noise_std'], # noise introduced to the hidden state of encoder-decoder during adversarial training to mimic an idea of the GAN from CV
                     'optim_d_prob': kwargs['optim_d_prob'] # the probability an update step for the discriminator is taken with
                     }
        # Store essential parameters and objects
        self.generator = generator_class
        self.discriminator = discriminator_class
        # do some checks
        assert self.grid['batch_size'] == self.generator.batch_size, "Batch_size used of adversarial training must be comply with the one used for the generator pretraining!"

        # Store other essential parameters and objects
        self.device = kwargs['device']
        self.loss_function_D = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_function_G = nn.CrossEntropyLoss().to(self.device)
        # Initialize optimiser + set lr_scheduler for generator
        self.optimiser_D = optimiser_D(self.discriminator.model.parameters(), lr= self.grid['learning_rate_D'],
                                                weight_decay = self.grid['l2_reg'])
        self.optimiser_G = optimiser_G(self.generator.model.parameters(), lr= self.grid['learning_rate_G'],
                                                weight_decay = 1e-4)
        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimiser_G, lr_lambda = lambda lr: 0.98)
        
        # Store indices of <pad> and <eos>
        self.pad_idx = self.grid['headline_dictionary'].word2index['<pad>']
        self.eos_idx = self.grid['headline_dictionary'].word2index['eos']
        
        # Initialize module for the ROUGE metrics
        self.rouge = Rouge()

        # required for initialization
        self.epoch = 0
        
    def training(self,
                 X_train, X_train_lengths, y_train, y_train_lengths,
                 X_val, X_val_lengths, y_val, y_val_lengths):
        """
        Function running adversarial training for a given number of epochs. Early-stopping rule is not set here.
        training methodoly is described in the attached paper at the repo.

        :param X_train:
            type: Numpy array: [seq_len, n_samples]
            description: Input articles in the form of padded sequences of indexed words.
        :param X_train_lengths:
            type: Numpy array: [n_samples,]
            description: Length of paragraphs used for padding sequences to have the same length.
        :param y_train: 
            type: Numpy array: [seq_len, n_samples]
            description: Taget summaries in the form of padded sequences of indexed words.
        :param y_train_lengths: 
            type: Numpy array: [n_samples,]
            description: Length of summaries used for masking during the computation of loss function.
        :param X_val:
            type: [seq_len, n_samples]
            description: Input articles in the form of padded sequences of indexed words.
        :param X_val_lengths:
            type: Numpy array: [n_samples,]
            description: Length of paragraphs used for padding sequences to have the same length.
        :param y_val:
            type: Numpy array: [seq_len, n_samples]
            description: Taget summaries in the form of padded sequences of indexed words.
        :param y_val_lengths:
            type: Numpy array: [n_samples,]
            description: Length of summaries used for masking during the computation of loss function.

        :return:
            There is no object returned by this function. Ultimately, adversarial training class
            contains trained generator, discriminator and also contains the state of their optimizers.
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
        
        ##### RUN TRAINING #####
        for epoch in range(self.start_epoch, self.grid['max_epochs']):
            # save epoch used for saving model
            self.epoch = epoch 
            # set generator.model and discriminator.model to training state
            self.generator.model.train()
            self.discriminator.model.train()
            
            # Zero losses for a given epoch
            epoch_Loss_D = 0
            epoch_Loss_G = 0
            batch, batch_D= 0, 0            
            
            # shuffle the data (this is done so that our models will not overfit for a given order of examples)
            reshuffle = np.random.shuffle(input_arr)
            input_train, input_train_lengths = (
                input_train[reshuffle].squeeze(0),
                input_train_lengths[reshuffle].squeeze(0)
            )
            target_train, target_train_lengths = (
                target_train[reshuffle].squeeze(0),
                target_train_lengths[reshuffle].squeeze(0)
            )
            
            ##### RUN through BATCHES #####
            for input, target, seq_length_input, seq_length_target in zip(input_train,
                                                                          target_train,
                                                                          input_train_lengths,
                                                                          target_train_lengths,
                                                                          ):
                # counter
                batch += 1
                
                # Put paragraphs to Torch.Tensor
                input = torch.from_numpy(
                    input[:seq_length_input.max()]
                ).long()
                # Put summaries to Torch.Tensor and move to GPU 
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
                
                # Run the update step for the discriminator (this is conditioned by probability specified in paraeters)
                optim_D = np.random.random() < self.grid['optim_d_prob']
                if optim_D:
                    # counter
                    batch_D += 1
                    
                    #### Compute log(D(x)) using the batch of real examples ####
                    self.optimiser_D.zero_grad()
                    # dicriminator output
                    output_D, real_labels_flatten = self.discriminator.forward(
                        target.permute(1,0),
                        real_labels
                    )  #discriminator needs transpose input
                    
                    # calculate loss function on the batch of real examples
                    error_D_real = self.loss_function_D(
                        output_D,
                        real_labels_flatten
                    )
                                        
                    # Calculate gradient
                    error_D_real.backward(retain_graph = True) # we need to retain backprop graph as this fragment is also used the generator's optimizer
                    
                    #### Compute log(1 - D(G(z))) using the bacth of fake examples ####
                    # Generate summaries
                    output_G = self.generator.model(
                        seq2seq_input = input,
                        input_lengths = seq_length_input,
                        target = target,
                        teacher_forcing_ratio = 1,
                        adversarial = True,
                        noise_std = self.grid['noise_std']
                    )

                    # discriminator output D(G(z))
                    output_D_G, fake_labels_flatten = self.discriminator.forward(output_G.argmax(dim = 2).long().permute(1,0), fake_labels) #discriminator needs transpose input
                    # calculate loss function on the batch of fake examples
                    error_D_fake = self.loss_function_D(
                        output_D_G,
                        fake_labels_flatten
                    )
                    # calculate gradient
                    error_D_fake.backward(retain_graph = True) # we need to retain backprop graph as this fragment is also used the generator's optimizer
                    
                    # sum gradients from computation both on real and fake examples
                    error_D = (error_D_real + error_D_fake) / 2
                    
                    # update step
                    self.optimiser_D.step()
                    
                    # cleaning and saving
                    epoch_Loss_D += ( (error_D - epoch_Loss_D) / batch_D )
                    del output_D_G
                    torch.cuda.empty_cache()
                
                #####
                # (2) Update Generator: we maximize log(D(G(z)))
                # 
                #####
                for _ in range(self.grid['G_multiple']):
                    self.optimiser_G.zero_grad()
                    if (_ != 0) | (optim_D != True): #output_G is not necessary to be computed for the first update step on a batch given the two conditiosn as this has already been done
                        # Generate summaries
                        output_G = self.generator.model(
                            seq2seq_input = input,
                            input_lengths = seq_length_input,
                            target = target,
                            teacher_forcing_ratio = 1,
                            adversarial = True,
                            noise_std = self.grid['noise_std']
                        )

                    ## Compute the first term of the loss function: - log(D(G(x)))
                    # FORWARD pass with updated discriminator
                    output_D, real_labels_flatten = self.discriminator.forward(
                        output_G.argmax(dim = 2).long().permute(1,0),
                        real_labels
                    )

                    ## Compute the second term of the loss function: CrossEntropy calculated over the generated sequence measuring the difference between generated and target summary
                    # Pack output and target padded sequence
                    ## Determine a length of output sequence based on the first occurrence of <eos>
                    seq_length_output = (output_G.argmax(2) == self.grid['text_dictionary'].word2index['eos']).int().argmax(0).cpu().numpy()
                    seq_length_output += 1
                                        
                    # determine seq_length for computation of loss function based on max(seq_lenth_target, seq_length_output)
                    seq_length_loss = np.array(
                        (seq_length_output, seq_length_target)
                        ).max(0)
                    
                    # pack padded output headlines
                    output_G = nn.utils.rnn.pack_padded_sequence(
                        output_G,
                        lengths = seq_length_loss,
                        batch_first = False,
                        enforce_sorted = False
                    ).to(self.device)
                    
                    # pack target headlines
                    target_padded = nn.utils.rnn.pack_padded_sequence(
                        target,
                        lengths = seq_length_loss,
                        batch_first = False,
                        enforce_sorted = False
                    ).to(self.device)
                    
                    # Compute loss: G = G_1 * G_2
                    error_G_2 = self.loss_function_G(
                        output_G[0],
                        target_padded[0]
                    )
                    error_G_1 = self.loss_function_D(
                        output_D,
                        real_labels_flatten
                    )
                    error_G = error_G_1 * error_G_2
                    
                    # Calculate gradient
                    error_G.backward(retain_graph = True)
                    # Update step
                    self.optimiser_G.step()
                    # cleaning
                    del output_G, target_padded
                    torch.cuda.empty_cache()
                
                #### MEASUREMENT #### - this returns results computed on validation set throughout the training to monitor the progress
                ## Return: ROUGE-1, ROUGE-2, ROUGE-L, accuracy of discriminator
                if batch % 20 == 0:
                    # turn the generator and the discriminator to evalutaion mode
                    self.generator.model.eval()
                    self.discriminator.model.eval()
                    
                    # zero losses, ROUGE metrics, and accuracy for the discriminator
                    val_batch = 0
                    val_loss = 0
                    self.rouge1, self.rouge2, self.rougeL = 0, 0, 0
                    outputs_true = 0

                    ### Run through batches###
                    for input, target, seq_length_input, seq_length_target in zip(input_val,
                                                                              target_val,
                                                                              input_val_lengths,
                                                                              target_val_lengths,
                                                                              ):
                        # counter                                                        
                        val_batch += 1
                        
                        # Put paragraphs to Torch.Tensor
                        input = torch.from_numpy(
                            input[:seq_length_input.max()]
                        ).long()
                        
                        # Put headlines to Torch.Tensor and move them to GPU
                        target = torch.from_numpy(
                            target[:seq_length_target.max()]
                        ).long().to(self.device)           
                        
                        # Eventually we are mainly interested in the generator performance measured by ROUGE metrics and fooling discriminator (may be measured by accuracy)
                        ## GENERATOR perfrormance
                        output_G = self.generator.model(
                            seq2seq_input = input,
                            input_lengths = seq_length_input,
                            target = target,
                            teacher_forcing_ratio = 0,
                            adversarial = False,
                            noise_std = 0
                        )
                        
                        ## add batch loss to the validation loss
                        val_loss += self.validation_loss_eval(
                            output_G,
                            target,
                            seq_length_target
                        )
                        
                        # Derive output headlines by restoring sentences from returned indices
                        hypotheses = output_G.argmax(dim = 2).permute(1,0).cpu().numpy() # move hypothesis to CPU to be able run list comprehension below
                        hypotheses = [' '.join([self.grid['headline_dictionary'].index2word[index] for index in hypothesis if ( index != self.pad_idx) & (index != self.eos_idx)][1:]) for hypothesis in hypotheses]
                        
                        # Derive target headlines by restoring sentences from returned indices
                        references = [' '.join([self.grid['headline_dictionary'].index2word[index] for index in ref if ( index != self.pad_idx) & (index != self.eos_idx)][1:]) for ref in target.permute(1,0).cpu().numpy()]
                        
                        # Calculate ROUGE metrics
                        ROUGE = [self.rouge_get_scores(hyp, ref) for hyp, ref in zip(hypotheses, references)]
                        self.rouge1 += ( (np.array([x[0]['rouge-1']['f'] for x in ROUGE if x != 'drop']).mean() - self.rouge1) / val_batch )
                        self.rouge2 += ( (np.array([x[0]['rouge-2']['f'] for x in ROUGE if x != 'drop']).mean() - self.rouge2) / val_batch )
                        self.rougeL += ( (np.array([x[0]['rouge-l']['f'] for x in ROUGE if x != 'drop']).mean() - self.rougeL) / val_batch )
                        
                        ### DISCRIMINATOR performance
                        ## (1) Run on real examples
                        # compute discriminator's output
                        output_D, real_labels_flatten = self.discriminator.forward(
                            target.permute(1,0), #discriminator needs transpose input
                            real_labels
                        ) 
                        outpud_D = output_D.detach().cpu().numpy() # move discriminator's output to CPU to be able run list comprehension below
                        # cleaning to avoid running out of CUDA memory
                        torch.cuda.empty_cache()                        
                        # Derive Real/Fake labels produced by discriminator
                        output_labels = np.array(
                            [1 if x>=0 else 0 for x in outpud_D]
                            )
                        outputs_true += sum(output_labels == real_labels_flatten.cpu().numpy())
                        
                        ## (2) Run on fake/generated examples
                        # derive indexed words from generator's prediction and turn into long which is accepted by torch mudles
                        output_G = F.log_softmax(output_G, dim = 2).argmax(dim = 2).long() 
                        # compute discriminator's output
                        output_D_G, fake_labels_flatten = self.discriminator.forward(
                            output_G.permute(1,0), #discriminator needs transpose input
                            fake_labels
                        )
                        # move output_D_G to CPU so that we can run list comprehension below 
                        outpud_D_G = output_D_G.detach().cpu().numpy()
                        # cleaning to avoid running out of CUDA memory
                        torch.cuda.empty_cache()
                        # Derive Real/Fake labels produced by discriminator
                        output_labels = np.array(
                            [1 if x>=0 else 0 for x in outpud_D_G]
                            )
                        outputs_true += sum(output_labels == fake_labels_flatten.cpu().numpy())
                        # cleaning to avoid running out of CUDA memory
                        del output_D, output_D_G
                        torch.cuda.empty_cache()

                    # Compute accuracy and mean validation loss                        
                    acc = 100 * float(outputs_true) / (2*self.n_batches_val*self.grid['batch_size'])
                    val_loss /= val_batch

                    # Eventually we are mainly interested in the generator performance measured by ROUGE metrics and fooling discriminator (may be measured by accuracy)
                    print(
                        f'Epoch: {epoch+1:.0f}'
                    )
                    print(
                        f'Generator performance after {100*batch/self.n_batches:.2f} % of examples.'
                    )
                    print(
                        f'ROUGE-1 = {100*self.rouge1:.3f} | ROUGE-2 = {100*self.rouge2:.3f} | ROUGE-l = {100*self.rougeL:.3f} | Discriminator accuracy = {acc:.2f} %.'
                    )
                    
                    # turn the generator and discriminator back to the training mode
                    self.generator.model.train()
                    self.discriminator.model.train()
            
            # decrease learning rate for a generator after the epoch
            self.lr_scheduler.step()
            # save the models' state and also the states of their optimizer
            self.save()
            
    def rouge_get_scores(self, hyp, ref):
      """
      HELPER function for computing ROUGE. Some summaries are kind of flawed (empty) thus they are needed to drop.

      :param hyp:
        type: String
        description: Output summary/headline returned by the model
      :param ref:
        type: String
        description: Reference/target summary/headline

      :return:
        Either ROUGE scores for a given pair of hyp and ref, or "drop" if a summary is invalid
      """
      try:
        return self.rouge.get_scores(hyp, ref)
      except:
        return "drop"
    
    def validation_loss_eval(self, output_G, target, seq_length_target):
        """
        Function handling the course of computing Cross-Entropy loss.
        
        :param output_G:
            type: Torch.Tensor
            description: Probabilites produced by generators
        :param target:
            type: Torch.Tensor
            description: Padded sequences of indexed words representing target summaries
        :param seq_length_target:
            type: Numpy.Array
            description: An array containing the lengths of target summaries

        :return loss:
            type: Float
            description: Cross-entropy loss (mean)
        """            
        # Pack output and target padded sequence
        ## Determine a length of output sequence based on the first occurrence of <eos>
        seq_length_output = (output_G.argmax(2) == self.grid['text_dictionary'].word2index['eos']).int().argmax(0).cpu().numpy()
        seq_length_output += 1
                            
        # determine seq_length for computation of loss function based on max(seq_lenth_target, seq_length_output)
        seq_length_loss = np.array(
            (seq_length_output, seq_length_target)
            ).max(0)
        
        # Pack padded generator's output and move to the GPU for the loss computation
        output_G = nn.utils.rnn.pack_padded_sequence(
            output_G,
            lengths = seq_length_loss,
            batch_first = False,
            enforce_sorted = False
        ).to(self.device)
        
        # Pack padded target sequences and move thme to the GPU for the loss computation
        target = nn.utils.rnn.pack_padded_sequence(
            target,
            lengths = seq_length_loss,
            batch_first = False,
            enforce_sorted = False
        ).to(self.device)
        
        # loss function
        loss = 0
        loss += self.loss_function_G(
            output_G[0],
            target[0]
        ).item()
        # cleaning
        del output_G, target, seq_length_output, seq_length_loss
        torch.cuda.empty_cache()
        
        #return loss
        return loss
    
    def generate_summaries(self, input_val, input_val_lengths, target_val, target_val_lengths):
        """
        A function that is responsible for generating summaries.
        Target summary is not actually used by this function, however, it is required to pass it to generator.model # this is desired to change

        :param input_val:
            type: [seq_len, n_samples]
            description: Input articles in the form of padded sequences of indexed words.
        :param input_val_lengths:
            type: Numpy array: [n_samples,]
            description: Length of paragraphs used for padding sequences to have the same length.
        :param target_val:
            type: Numpy array: [seq_len, n_samples]
            description: Taget summaries in the form of padded sequences of indexed words.
        :param target_val_lengths:
            type: Numpy array: [n_samples,]
            description: Length of summaries used for masking during the computation of loss function.
                
        :return summaries:
            type: Numpy.Array
            description: Summaries generated by the trained model.
        """
        # Turn the generator to the eval mode
        self.generator.model.eval()
        
        # Generate batches
        (input_val, input_val_lengths,
        target_val, target_val_lengths) = self._generate_batches(
            padded_input = input_val,
            input_lengths = input_val_lengths,
            padded_target = target_val,
            target_lengths = target_val_lengths
        )
        # create an empty list for storing output headlies
        OUTPUT = []
        ### RUN through BATCHE###
        for input, target, seq_length_input, seq_length_target in zip(input_val,
                                                                      target_val,
                                                                      input_val_lengths,
                                                                      target_val_lengths
                                                                      ):
            ## FORWARD PASS
            # Prepare RNN-edible input - i.e. pack padded sequence
            
            # trim input and put store it to Torch.Tensor
            input = torch.from_numpy(
                input[:seq_length_input.max()]
                ).long()
            # trim target and put store it to Torch.Tensor and move to GPU
            target = torch.from_numpy(
                target[:seq_length_target.max()]
                ).long().to(self.device)

            # FORWARD PASS through generator            
            output = self.generator.model(
                seq2seq_input = input,
                input_lengths = seq_length_input,
                target = target,
                teacher_forcing_ratio = 0,
                adversarial = False,
                noise_std = 0
            )
            # cleanin to avoid running out of CUDA memory
            del input, target
            torch.cuda.empty_cache()
            
            # Store indexed word into output. These are stored again on CPU
            OUTPUT.append(
                output.argmax(dim = 2).cpu().numpy()
                )
        
        return np.array(OUTPUT)
    
    def _generate_batches(self, padded_input, input_lengths, padded_target, target_lengths):
        """
        A module responsible for generating batches.
        At this stage, we prefer storing everythin on CPU as Numpy.Arrays.
        It might be preferable to direcetly output Torch.Tensor to cut some operations during the training,

        :param input:
            type: Numpy.Array: [seq_len, n_examples]
            description: Input articles
        :param inout_lengths:
            type: Numpy.Array: [n_examples,]
            description: Lenghts of input articles
        :param target:
            type: Numpy.Array: [seq_len, n_examples]
            description: Target summaries
        :param target_lengths:
            type: Numpy.Array: [n_examples]
            description: Lenght of target summaries
            
        :return input_batches:
            type: Numpy.Array: [n_batches, seq_len, batch_size]
            description: Input articles splitted into individual batches
        :return input_lengths:
            type: Numpy.Array: [n_batches, batch_size]
            description: Lenghts of input articles splitted into individual batches
        :return target_batches:
            type: Numpy.Array: [n_batches, seq_len, batch_size]
            description: Target summaries splitted into individual batches
        :return target_lengths:
            type: Numpy.Array: [n_batches, batch_size]
            description: Lenghts of target summaries splitted into individual batches
        """
        # determine a number of batches; here we drop a few examples which does not constitute the whole batch
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
    
    def save(self):
        """
        A module which saves the models' states and also the states of their optimizer sinto the predefined path given by the model name 
        """
        # Save generator state and state of its optimizer
        torch.save(
            self.generator.model.state_dict(),
            "../data/Results/{}.pth".format(self.grid['model_name'])
        )
        torch.save(
            self.optimiser_G.state_dict(),
            "../data/Results/opt_g_{}.pth".format(self.grid['model_name'])
        )

        # Save discriminator state and state of its optimizer
        torch.save(
            self.discriminator.model.state_dict(),
            "../data/Results/disc_{}.pth".format(self.grid['model_name'])
        )
        torch.save(
            self.optimiser_D.state_dict(),
            "../data/Results/opt_d_{}.pth".format(self.grid['model_name'])
        )

        # save a piece of information containing a number of epoch done
        f = open(
            f"epochs_{self.grid['model_name']}.txt",
            'a'
        )
        f.write(str(self.epoch+1))
        f.close()
    
    def load(self):
        """
        Loading models' states and the states of their optimizers.
        """
        try:
            
            # Load generator states and its optimizer
            self.generator.model.load_state_dict(
                torch.load("../data/Results/{}.pth".format(self.grid['model_name']))
                )
            self.optimiser_G.load_state_dict(
                torch.load("../data/Results/opt_g_{}.pth".format(self.grid['model_name']))
                )
            
            # Load discriminator states and its optimizer
            self.discriminator.model.load_state_dict(
                torch.load("../data/Results/disc_{}.pth".format(self.grid['model_name']))
                )
            self.optimiser_D.load_state_dict(
                torch.load("../data/Results/opt_d_{}.pth".format(self.grid['model_name']))
                )

            # load startin epoch
            self.start_epoch = int(np.loadtxt(f"epochs_{self.grid['model_name']}.txt"))
            print('Model state has been successfully loaded.')
        except: # this is used when above files are not available, i.e. when the training with a given model_name is run for the first time
            self.start_epoch = 0
            print('No state has been loaded.')