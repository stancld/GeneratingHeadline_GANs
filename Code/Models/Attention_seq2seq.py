"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London
File: Attention_seq2seq.py
Description of our model:
Collaborators:
    - Daniel Stancl
    - Dorota Jagnesakova
    - Guoliang HE
    - Zakhar Borok`
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# Encoder
class _Encoder(nn.Module):

    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, rnn_num_layers,
                 dropout, embeddings, device):
        """
        Args:
            emb_dim -> int

            enc_hid_dim -> int

            dec_hid_dim -> int

            dropout -> float

            embeddings -> Tensor

            device -> str

        """
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=enc_hid_dim,
                          num_layers=rnn_num_layers,
                          bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.embeddings = embeddings
        self.device = device

    def forward(self, enc_input, input_lengths):
        """
        Args:
            enc_input -> Tensor: [enc_input_len, batch size]

            input_lengths +++++++

        Returns:
            outputs -> Tensor: [enc_seq_len, batch size, enc hid dim * 2]

            hidden -> Tensor: [batch size, dec hid dim]
        """

        # embedding and dropout layer
        embedded = self.dropout(
            torch.tensor(
                [[self.embeddings[x] for x in enc_input[:, seq]]
                    for seq in range(enc_input.shape[1])]
            ).permute(1, 0, 2).to(self.device)
        ).float()  # [enc_input_len, batch size, emb_dim]

        # pack padded_layers

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # feed through RNN
        outputs, hidden = self.rnn(embedded)

        # cleaning
        del embedded
        torch.cuda.empty_cache()

        # unpacking
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

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

        return outputs, hidden


class _Attention(nn.Module):
    """
    """

    def __init__(self, enc_hid_dim, dec_hid_dim):
        """
        Args:
            enc_hid_dim -> int: the encoder hidden dimension
            dec_hid_dim -> int: the decoder hidden dimension
        """
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        """
        Args:

            hidhen -> Tensor: [batch size, dec hid dim]

            encoder_outputs -> Tensor: [enc_seq_len, batch size, enc hid dim * 2]

            mask -> 

        Returns:

            softmax(attention) -> Tensor: [batch size, enc_seq_len] 
        """

        enc_seq_len = encoder_outputs.shape[0]

        # repeat decoder hidden state enc_seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, enc_seq_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, enc_seq_len, dec hid dim]
        # encoder_outputs = [batch size, enc_seq_len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, enc_seq_len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # ignoring
        attention = attention.masked_fill(mask == 0, -1e12)

        # cleaning
        del energy, encoder_outputs, hidden
        torch.cuda.empty_cache()

        return F.softmax(attention, dim=1)


class _Decoder(nn.Module):
    """
    """

    def __init__(self, output_dim, enc_hid_dim,  dec_hid_dim, rnn_num_layers,
                 dropout, attention, embeddings, device):
        """
        Args:

        output_dim -> int

        enc_hid_dim -> int

        dec_hid_dim -> int

        dropout -> float 

        attention -> attention object

        embeddings -> Tensor: pre-trained embeddings

        device -> str

        """
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        #self.embedding = nn.Embedding(output_dim, output_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + embeddings.shape[1], dec_hid_dim,
                          num_layers=rnn_num_layers)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + embeddings.shape[1], output_dim)

        self.dropout = nn.Dropout(dropout)
        self.embeddings = embeddings
        self.device = device

    def forward(self, dec_input, hidden, encoder_outputs, mask):
        """
        Args:

            dec_input -> Tensor: [1,batch size,dec_emb dim]

            hidden -> Tensor: [batch size, dec hid dim]

            encoder_outputs -> Tensor: [enc_seq_len, batch size, enc hid dim * 2]

        Returns:

            prediction -> Tensor: [batch size, output dim]

            hidden -> Tensor: [batch size, dec hid dim]
        """

        embedded = self.dropout(
            torch.tensor(
                [self.embeddings[x] for x in dec_input]
            ).to(self.device)
        ).float().unsqueeze(0)

        attention = (
            self.attention(hidden, encoder_outputs, mask)
        ).unsqueeze(1)  # attention = [batch size, 1, enc_seq_len]

        # encoder_outputs = [batch size, enc_seq_len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = torch.bmm(attention, encoder_outputs)

        # weighted = [1, batch size, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)

        # print('embedded',embedded.size())
        # rnn_input = [1, batch size, (enc hid dim * 2) + dec_emb dim]
        rnn_input = torch.cat((embedded, weighted), dim=2).float()

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0).float())

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        # assert (output == hidden).all() # just check

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1)
        )

        # clearing GPU memory
        del embedded, output, weighted
        torch.cuda.empty_cache()

        return prediction, hidden.squeeze(0), attention.squeeze(1)


class _Seq2Seq(nn.Module):
    """
    """

    def __init__(self, encoder, decoder, device, embeddings, text_dictionary):
        """
        Args:

            encoder -> encoder object

            decoder -> decoder object

            device -> str

            embeddings -> Tensor: pre-trained embedding

            text_dictionary -> dict
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.embeddings = embeddings
        self.text_dictionary = text_dictionary

    def __mask__(self, input):
        """
        Args:
            input

        Returns
            mask
        """
        return torch.tensor(
            (input != self.text_dictionary['<pad>'])
        ).to(self.device).permute(1, 0)

    def __mask_from_seq_lengths__(self, input_lengths):
        """
        Args:
            input_lengths

        Returns
            mask:
        """
        return torch.from_numpy(
            np.array(
                [np.c_[np.ones((1, i)), np.zeros(
                    (1, input_lengths.max() - i))].reshape(-1) for i in input_lengths]
            )
        ).to(self.device)

    def forward(self, seq2seq_input, input_lengths, target, teacher_forcing_ratio=0.5, adversarial=False, noise_std=0.0):
        """
        Args:
            seq2seq_input -> Tensor: [seq_len, batch size]

            target -> Tensor: [trg_len, batch size,output_dim]

            teacher_forcing_ratio -> int: probability of teacher forcing

            input_lengths ++++++

            adversarial training +++++

            noise_std +++++++

        Returns:
            outputs -> Tensor: [trg_len, batch_size, trg_vocab_size] 
        """

        batch_size = seq2seq_input.shape[1]
        trg_len = target.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder dec_outputs
        dec_outputs = torch.zeros(trg_len, batch_size,
                                  trg_vocab_size)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden_ = self.encoder(seq2seq_input, input_lengths)

        # introduce noise if adversarial = True
        if (adversarial == True) & (np.random.random() > .5):
            hidden = hidden_.clone() + torch.from_numpy(
                np.random.normal(0, noise_std, size=hidden_.shape)
            ).to(self.device).float()
            hidden = hidden.float()
        else:
            hidden = hidden_

        dec_input = target[0]
        mask = self.__mask_from_seq_lengths__(input_lengths)

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            #output, hidden = self.decoder(dec_input, hidden, encoder_outputs)
            output, hidden, a_ = self.decoder(
                dec_input, hidden, encoder_outputs, mask)

            # cleaning
            del a_

            # place predictions in a tensor holding predictions for each token
            dec_outputs[t] = output.cpu()

            # decide if we are going to use teacher forcing or not
            teacher_force = np.random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input; else, use predicted token
            dec_input = target[t] if teacher_force else top1
            dec_input = dec_input.cpu().numpy()
            torch.cuda.empty_cache()
        return dec_outputs.to(self.device)

    def save(self, name_path):
        """
        save the model to name_path
        """
        torch.save(self.state_dict(), name_path)

    def load(self, name_path):
        """
        laod the model to name_path
        """
        self.load_state_dict(torch.load(name_path))
        self.eval()
