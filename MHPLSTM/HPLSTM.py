import torch
import torch.nn as nn
from Modules import FirstModule, SecondModule


class HPLSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super(HPLSTMModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias

        self.firstmodule = FirstModule(input_size, hidden_size, bias=bias)
        self.secondmodule = SecondModule(input_size, hidden_size, output_size, bias=bias)

    def forward(self, inputs):
        """
            input = i_vectors
            i_vectors.shape: (seq_length, input_size)
        """
        i_vectors = inputs
        seq_length = len(inputs)
        fg, hr = self.firstmodule(i_vectors)

        fg_vectors = fg.chunk(seq_length, 0)
        hr_vectors = hr.chunk(seq_length, 0)
        c0 = torch.zeros((1, self.hidden_size))
        c_vectors = []
        for i in range(seq_length):
            c_ith = torch.add(c0 * fg_vectors[i], hr_vectors[i])
            c_vectors.append(c_ith)
            c0 = c_ith
        
        o = self.secondmodule((i_vectors, c_vectors))
        o = o.chunk(seq_length, 0)

        return o


class MultiHeadHPLSTMModule(nn.Module):
    """"""