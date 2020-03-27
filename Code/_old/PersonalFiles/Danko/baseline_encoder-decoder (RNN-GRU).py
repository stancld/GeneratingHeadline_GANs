"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London

File: RNN_model
comment: baseline encoder-decoder model - RNN with an attention mechanism

Collaborators:
    - Daniel Stancl
    - Dorota Jagnesakova
    - Guoliang HE
    - Zakhar Borok`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size):
        """
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.gru(hidden_size, hidden_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    
    def initHidden(self):
        if torch.cuda.is_available():
            return torch.normal(0, 0.02, (1, 1, self.hidden_size)).cuda()
        else:
            return torch.normal(0, 0.02, (1, 1, self.hidden_size))
        
        
class Decoder(nn.Module):
    """
    """
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.gru(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, input, hidden):
        output = F.relu(
            self.embedding(input).view(1, 1, -1)
            )
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        if torch.cuda.is_available():
            return torch.normal(0, 0.02, (1, 1, self.hidden_size)).cuda()
        else:
            return torch.normal(0, 0.02, (1, 1, self.hidden_size))
    