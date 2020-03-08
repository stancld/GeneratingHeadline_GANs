"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London

File: GRU-attention-generator.py
Source: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-the-model

Collaborators:
    - Daniel Stancl
    - Dorota Jagnesakova
    - Guoliang HE
    - Zakhar Borok`
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder
class Encoder(nn.Module):
  """
  """
  def __init__(self, pre_train_weight, batch_size):
    """
    """
    super(Encoder, self).__init__()
    self.hidden_size = pre_train_weight.shape[1] # dimension of word vectors
    self.input_size = pre_train_weight.shape[0]   # number of all words
    self.batch_size = batch_size

    self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pre_train_weight), 
                                                  freeze=True)
    
    self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                       num_layers=1, bidirectional = True, batch_first = True)

  def forward(self, input, hidden):
    """
    :param input:
    :param hidden:
    """
    embedded = self.embedding(input).float()
    output = embedded.transpose(0,1)
    output, hidden = self.lstm(output, hidden)
    return output, hidden

  def init_hidden(self):
    """
    """
    return torch.zeros(1, self.batch_size, self.hidden_size, device = device)






# Attention mechanism
class Attention(nn.Module):
  def __init__(self, hidden_state: 'hidden state in decoder RNN'):
    super(Attention, self).__init__()
    self.attn = nn.Linear(hidden_state * 2, hidden_state) # here assumre hidden = input size
    self.attn_weight_coe = nn.Parameter(torch.rand(hidden_state))
    
  def forward(self, hidden, encoder_outputs):
    seq_len = encoder_outputs.size(0)
    atten_hidden = hidden.unsqueeze(0).repeat(seq_len, 1, 1).transpose(0, 1) # (batch_size,seq_len,hidden)
    encoder_outputs = encoder_outputs.transpose(0, 1)  # (batch_size,seq_len,hidden)

    attn = F.tanh(self.attn(torch.cat([atten_hidden, encoder_outputs], dim=2))).transpose(1, 2)  #[batch_size,hidden,seq_len]
    attn_weight = self.attn_weight_coe.repeat(encoder_outputs.size(0), 1).unsqueeze(1) # [batch_size,1,hidden]
    attention = torch.bmm(attn_weight,attn).squeeze(1) # [batch_size,seq_len]
    scored_attn = F.softmax(attention, dim=1).unsqueeze(1)
    # store atten_hidden and encoder_outputs for computing Generation Probability below
    self.atten_hidden, self.encoder_outputs = atten_hidden, encoder_outputs 
    return scored_attn # **[batch_size,1,seq_len]

# Generation probability
class GenerationProbability(nn.Module):
    def __init__(self, pre_train_weight: np.array, batch_size):
        super(GenerationProbability, self).__init__()
        self.hidden_state = pre_train_weight.shape[1]
        self.attention = Attention(self.hidden_state)
        self.batch_size = batch_size
        self.context_weight = nn.Linear(self.hidden_state)
    
    def forward(self, input: 'index of a single word, i.e. len is 1',\
                           prev_hidden, encoder_outputs): # N = sequence length for output! hidden = feature size):
        embedded = self.embedding(input).float()  # (batch_size,hidden)
        embedded = self.dropout(embedded).view(1, self.batch_size, -1) # (1,batch_size,hidden)
        scorred_attn = self.attention(prev_hidden[-1], encoder_outputs) # **[batch_size,1,seq_len]
        context = torch.einsum(
            'ijk, ikh -> ih1', scorred_attn, self.attention.atten_hidden
            )
        