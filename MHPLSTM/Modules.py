import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class FirstModule(nn.Module):
    """Input gate + forget gate + hidden state"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(FirstModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.ifh1 = nn.Linear(input_size, hidden_size * 3, bias=bias)
        self.h2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs):
        """
            input = (i_vectors, s_vectors)
            i_vectors.shape: (seq_length, input_size / 2)
            s_vectors.shape: (seq_length, input_size / 2)
        """

        i_vectors, s_vectors = inputs
        v_vectors = []
        for i in range(len(i_vectors)):
            v_vectors.append(torch.cat((i_vectors[i], s_vectors[i]), dim=0))
        v = torch.stack([v_vector.squeeze() for v_vector in v_vectors])

        gates = self.ifh1(v)
        ig, fg, h = gates.chunk(3, 1)

        ig = torch.sigmoid(ig)
        fg = torch.sigmoid(fg)
        h = self.h2(torch.tanh(h))
        hr = h * ig

        return fg, hr


class SecondModule(nn.Module):
    """output gate"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(SecondModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.o = nn.Linear(input_size, hidden_size, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs):
        """"
            inputs = (i_vectors, c_vectors)
            i_vectors.shape: (seq_length, input_size / 2)
            c_vectors.shape: (seq_length, input_size / 2)
            
        """
        
        i_vectors, c_vectors = inputs
        ic_vectors = []
        for i in range(len(i_vectors)):
            ic_vectors.append(torch.cat((i_vectors[i], c_vectors[i]), dim=0))
        ic = torch.stack([ic_vector.squeeze() for ic_vector in ic_vectors])
        c = torch.stack([c_vector.squeeze() for c_vector in c_vectors])

        og = torch.sigmoid(self.o(ic))
        o = c * og

        return o