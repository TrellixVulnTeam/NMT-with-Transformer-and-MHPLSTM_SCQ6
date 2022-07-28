import torch
import torch.nn as nn
import numpy as np


class FirstModule(nn.Module):
    """Input gate + forget gate + hidden state"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(FirstModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.ifh1 = nn.Linear(input_size * 2, hidden_size * 3, bias=bias)
        self.h2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs):
        """
            input = i_vectors
            i_vectors.shape: (seq_length, input_size)
        """
        i_vectors = inputs
        seq_length = i_vectors.size(0)
        i_vectors = i_vectors.chunk(seq_length)
        
        s_vectors = []
        s_ith = torch.zeros(1, self.input_size)
        s_vectors.append(s_ith)

        for i in range(len(i_vectors) - 1):
            s_ith = torch.add(s_ith, i_vectors[i])
            s_vectors.append(s_ith)
        
        v_vectors = []
        for i in range(len(i_vectors)):
            v_vectors.append(torch.cat((i_vectors[i], s_vectors[i]), dim=1))
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
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super(SecondModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias

        self.o = nn.Linear(input_size + hidden_size, output_size, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.output_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs):
        """"
            inputs = (i_vectors, c_vectors)
            i_vectors.shape: (seq_length, input_size)
            c_vectors.shape: (seq_length, hidden_size)
        """
        i_vectors, c_vectors = inputs
        seq_length = i_vectors.size(0)
        i_vectors = i_vectors.chunk(seq_length)
        c_vectors = c_vectors.chunk(seq_length)

        ic_vectors = []
        for i in range(len(i_vectors)):
            ic_vectors.append(torch.cat((i_vectors[i], c_vectors[i]), dim=1))
        ic = torch.stack([ic_vector.squeeze() for ic_vector in ic_vectors])
        c = torch.stack([c_vector.squeeze() for c_vector in c_vectors])

        og = torch.sigmoid(self.o(ic))
        o = c * og

        return o