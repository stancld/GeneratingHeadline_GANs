import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_parameter():

    print(
        ''' Hello:

    run function: instan_things to instantiate your model, 
            in which you should define the following dictionary parameters

    param = {'max_epochs':3,
            'learning_rate':1e-3,       
            'clip':1,                  # clip grad norm
            'teacher_forcing_ratio':1, # during training
            'OUTPUT_DIM':1,
            'ENC_EMB_DIM':21,
            'ENC_HID_DIM':32,
            'DEC_HID_DIM':32,
            'ENC_DROPOUT':0,
            'DEC_DROPOUT':0,
            'device':device}
      
    run function: seq2seq_running to train your model,
            in which you should pass:
    grid, model, optimiser, lossfunction, X_train, y_train, X_test, y_test, teacher_forcing_ratio
    
    run function: seq2seq_evaluate to evaluate, 
            in which you should pass:
    model, X_test, y_test, lossfunction
    
    to predict you do:
    model(self, seq2seq_input, target, teacher_forcing_ratio = 0)

    seq2seq_input = [seq_len, batch size,Enc_emb_dim]
    target = [trg_len, batch size,output_dim], trg_len is prediction len
    
    '''
    )


class _Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input):

        # enc_input = [enc_input_len, batch size,emb_dim]

        embedded = self.dropout(enc_input)

        # embedded = [enc_input_len, batch size, emb_dim]

        outputs, hidden = self.rnn(embedded)

        # outputs = [enc_input len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        return outputs, hidden


class _Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class _Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim,  dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        #self.embedding = nn.Embedding(output_dim, output_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + output_dim, dec_hid_dim)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, hidden, encoder_outputs):

        # dec_input = [1,batch size,dec_emb dim]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        embedded = self.dropout(dec_input)

        # embedded = [1, batch size, dec_emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hid dim * 2]

        # print('embedded',embedded.size())
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + dec_emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)


class _Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, seq2seq_input, target, teacher_forcing_ratio=0.5):

        # seq2seq_input = [seq_len, batch size,Enc_emb_dim]
        # target = [trg_len, batch size,output_dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = seq2seq_input.shape[1]
        trg_len = target.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(seq2seq_input)

        # check: make dimension consistent
        dec_input = target[0]
        dec_input = dec_input.unsqueeze(0)
        # print('dec_input dim:',dec_input.size())

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(dec_input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            top1 = top1.unsqueeze(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = target[t] if teacher_force else top1
            dec_input = dec_input.unsqueeze(1).float()
        return outputs

    def save(self, name_path):
        torch.save(self.state_dict(), name_path)  # e.g. 'encoder_model.pt'

    def load(self, name_path):
        self.load_state_dict(torch.load(name_path))
        self.eval()


def seq2seq_training(model, X_train, y_train, optimiser, lossfunction, clip, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0

    local_batch, local_labels = X_train.unsqueeze(
        1).to(device), y_train.unsqueeze(1).to(device)
    local_labels = local_labels.unsqueeze(2)

    # print('input')
    # print(local_batch.size())
    # print(local_labels.size())

    # forward pass
    optimiser.zero_grad()

    local_output = model(seq2seq_input=local_batch, target=local_labels,
                         teacher_forcing_ratio=teacher_forcing_ratio)
    local_output = local_output.view(-1)[1:]
    local_labels = local_labels.view(-1)[1:]

    # print('output:')
    # print(local_output.size())
    # print(local_labels.size())
    loss = lossfunction(local_output, local_labels)

    # backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimiser.step()
    epoch_loss += loss.item()
    return epoch_loss


def seq2seq_evaluate(model, X_test, y_test, lossfunction):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        local_batch, local_labels = X_test.unsqueeze(
            1).to(device), y_test.unsqueeze(1).to(device)
        local_labels = local_labels.unsqueeze(2)

        local_output = model(seq2seq_input=local_batch,
                             target=local_labels, teacher_forcing_ratio=0)

        local_output = local_output.view(-1)[1:]
        local_labels = local_labels.view(-1)[1:]

        loss = lossfunction(local_output, local_labels)
        epoch_loss += loss.item()
    return epoch_loss


def seq2seq_running(grid, model, optimiser, lossfunction, X_train, y_train, X_test, y_test, teacher_forcing_ratio):

    best_valid_loss = float('inf')

    for epoch in range(grid['max_epochs']):

        train_loss = seq2seq_training(
            model, X_train, y_train, optimiser, lossfunction, grid['clip'], teacher_forcing_ratio)
        valid_loss = seq2seq_evaluate(model, X_test, y_test, lossfunction)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            m = copy.deepcopy(model)
            print(f'Epoch: {epoch+1}:')
            print(f'Train Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

    return best_valid_loss, m


def instan_things(**kwargs):
    grid = {'max_epochs': kwargs['max_epochs'],
            'learning_rate': kwargs['learning_rate'],
            'clip': kwargs['clip'],
            # during training
            'teacher_forcing_ratio': kwargs['teacher_forcing_ratio']
            }
    OUTPUT_DIM = kwargs['OUTPUT_DIM']
    ENC_EMB_DIM = kwargs['ENC_EMB_DIM']
    #DEC_EMB_DIM = 1
    ENC_HID_DIM = kwargs['ENC_HID_DIM']
    DEC_HID_DIM = kwargs['DEC_HID_DIM']
    ENC_DROPOUT = kwargs['ENC_DROPOUT']
    DEC_DROPOUT = kwargs['DEC_DROPOUT']
    device = kwargs['device']

    attn = _Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = _Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = _Decoder(output_dim=OUTPUT_DIM,  enc_hid_dim=ENC_HID_DIM,
                   dec_hid_dim=DEC_HID_DIM, dropout=DEC_DROPOUT, attention=attn)
    model = _Seq2Seq(enc, dec, device).to(device)

    # mse loss
    optimiser = optim.Adam(model.parameters(), lr=grid['learning_rate'])
    lossfunction = nn.MSELoss().to(device)

    return (grid, model, optimiser, lossfunction)
