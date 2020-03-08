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
  def __init__(self,pre_train_weight,batch_size):
    super(Encoder, self).__init__()
    self.hidden_state = pre_train_weight.shape[1] # dimension of word vectors
    self.input_size = pre_train_weight.shape[0]   # number of all words
    self.batch_size = batch_size

    self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pre_train_weight),freeze=True)
    
    self.gru = nn.GRU(input_size=self.hidden_state,hidden_size=self.hidden_state,\
                                  num_layers=1)
    
    self.ini_hidden = torch.zeros(1, self.batch_size, self.hidden_state)

  def forward(self, input):
    embedded = self.embedding(input).float()
    input_to_rnn = embedded.transpose(0,1)
    (output,hidden) = self.gru(input_to_rnn,self.ini_hidden)
    return output, hidden

# Decoder
class Attention(nn.Module):
  def __init__(self, hidden_state: 'hidden state in decoder RNN'):
    super(Attention, self).__init__()
    self.attn = nn.Linear(hidden_state * 2, hidden_state) # here assumre hidden = input size
    self.attn_weight_coe = nn.Parameter(torch.rand(hidden_state))
    
  def forward(self,hidden,encoder_outputs):
    seq_len = encoder_outputs.size(0)
    atten_hidden = hidden.unsqueeze(0).\
                      repeat(seq_len, 1, 1).transpose(0, 1) # (batch_size,seq_len,hidden)
    encoder_outputs = encoder_outputs.transpose(0, 1)  # (batch_size,seq_len,hidden)

    attn = F.relu(self.attn(torch.cat([atten_hidden, encoder_outputs],\
                          dim=2))).transpose(1, 2)  #[batch_size,hidden,seq_len]
    attn_weight = self.attn_weight_coe.repeat(encoder_outputs.size(0),\
                                1).unsqueeze(1) # [batch_size,1,hidden]
    attention = torch.bmm(attn_weight,attn).squeeze(1) # [batch_size,seq_len]
    scored_attn = F.softmax(attention, dim=1).unsqueeze(1)
    return scored_attn # **[batch_size,1,seq_len]

class Decoder(nn.Module):
  def __init__(self,pre_train_weight: np.array, batch_size, dropout_p=0.1):
    super(Decoder, self).__init__()

    self.hidden_state = pre_train_weight.shape[1]
    self.output_size = pre_train_weight.shape[1]
    self.dropout_p = dropout_p
    self.batch_size = batch_size

    self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pre_train_weight),freeze=True)

    self.attention = Attention(self.hidden_state)

    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(input_size = self.hidden_state * 2 , hidden_size = self.hidden_state)
    self.out = nn.Linear(self.hidden_state*2, self.output_size)

  def forward(self, input: 'index of a single word, i.e. len is 1',\
                           prev_hidden, encoder_outputs): # N = sequence length for output! hidden = feature size
    # Get the embedding of the current input word (last output word)
    embedded = self.embedding(input).float()  # (batch_size,hidden)
    embedded = self.dropout(embedded).view(1,self.batch_size,-1) # (1,batch_size,hidden)

    scored_attn = self.attention(prev_hidden[-1],encoder_outputs) # **[batch_size,1,seq_len]
    encoder_outputs = encoder_outputs.transpose(0, 1)   # [batch_size,seq_len,hidden]
    context = torch.bmm(scored_attn,encoder_outputs)    # [batch_size,1,hidden]
    context = context.transpose(0, 1)  # [1,batch_size,hidden]

    rnn_input = torch.cat([embedded, context],dim=2) # [1,batch_size,hidden*2] 

    rnn_output, hidden = self.gru(rnn_input, prev_hidden) 

    # format rnn_output
    rnn_output = rnn_output.squeeze(0)  
    context = context.squeeze(0)
    output = self.out(torch.cat([rnn_output, context], dim = 1))
    output = F.log_softmax(output, dim=1)    # [batch_size,output_size]
    return (output, hidden, scored_attn)

  def get_decoder_hidden(self,encoder_hidden): # from encode ouput hidden to decoder initial hidden
    return(encoder_hidden)

# Seq2seq model
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
  
  def forward(self, train_input: '[seq_len,batch_size,input_size]',\
              target: '[seq_len,batch_size,input_size]',\
              teacher_forcing_ratio=0.5):
    batch_size = train_input.size(1)
    max_len = target.size(0)      # may be slightly longer?
    vocab_size = self.decoder.output_size # = word embedding space? 

    # encoder 
    # encoder_output.size[seq_len,batch_size,input_size]
    # enc_out_hidden.size[1,batch_size,input_size]
    encoder_output, enc_out_hidden = self.encoder(train_input) # enc_ini_hidden assume to be 0

    dec_hidden = self.decoder.get_decoder_hidden(enc_out_hidden[:1]) # hidden[:1].size(batch_size,hidden)

    dec_output = torch.zeros(max_len, batch_size, vocab_size).to(device)
    temp_output = target.data[0, :]  # TODO must the first ouput = <sos> index, size(batch_size,hidden)

    #decoder   haven't checked yet
    for t in range(1, max_len):
      temp_output, dec_hidden, attn_weights = self.decoder(
              temp_output, dec_hidden, encoder_output)
      dec_output[t] = temp_output
      is_teacher = random.random() < teacher_forcing_ratio
      top1 = temp_output.data.max(1)[1]
      temp_output = (target.data[t] if is_teacher else top1).to(device)
    return dec_output