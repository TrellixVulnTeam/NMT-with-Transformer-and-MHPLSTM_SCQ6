import torch
import torch.nn as nn
import numpy as np
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
        seq_length = i_vectors.size(0)
        fg, hr = self.firstmodule(i_vectors)

        fg_vectors = fg.chunk(seq_length, 0)
        hr_vectors = hr.chunk(seq_length, 0)
        c0 = torch.zeros((1, self.hidden_size))
        c_vectors = []
        for i in range(seq_length):
            c_ith = torch.add(c0 * fg_vectors[i], hr_vectors[i])
            c_vectors.append(c_ith)
            c0 = c_ith
        c_vectors = torch.stack([c_vector.squeeze() for c_vector in c_vectors])
        
        o = self.secondmodule((i_vectors, c_vectors))

        return o


class MultiHeadHPLSTMModule1(nn.Module):
    def __init__(self, n_head, d_input, d_hidden, d_output, bias=True):
        super(MultiHeadHPLSTMModule1, self).__init__()
        self.n_head = n_head
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.bias = bias

        self.input_linear = nn.Linear(n_head * d_input, n_head * d_input, bias=bias)
        self.output_linear = nn.Linear(n_head * d_output, n_head * d_output, bias=bias)
        self.reset_parameters()

        self.hplstmmodule = HPLSTMModule(d_input, d_hidden, d_output, bias=bias)

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.d_hidden * self.n_head)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs):
        """
            inputs = i_vectors
            i_vectors.shape: (seq_length, input_size) # input_size = n_head * d_input
        """
        i_vectors = inputs
        seq_length = i_vectors.size(0)

        # input linear projection
        i_vectors = self.input_linear(i_vectors)
        # split
        i_vectors = i_vectors.view(seq_length, self.n_head, self.d_input)
        i_vectors = i_vectors.contiguous().view(-1, self.d_input)
        # hplstm
        o_vectors = self.hplstmmodule(i_vectors)
        o_vectors = o_vectors.view(seq_length, self.n_head, self.d_output)
        # concat
        o_vectors = o_vectors.contiguous().view(seq_length, -1)
        # output linear projection
        o_vectors = self.output_linear(o_vectors)

        return o_vectors


class MultiHeadHPLSTMModule2(nn.Module):
    def __init__(self, n_head, d_input, d_hidden, d_output, bias=True):
        super(MultiHeadHPLSTMModule2, self).__init__()
        self.n_head = n_head
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.bias = bias

        self.input_linear = nn.Linear(n_head * d_input, n_head * d_input, bias=bias)
        self.output_linear = nn.Linear(n_head * d_output, n_head * d_output, bias=bias)
        self.reset_parameters()

        self.hplstm_list = nn.ModuleList([
            HPLSTMModule(d_input, d_hidden, d_output, bias=bias)
            for _ in range(n_head)])

    
    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.d_hidden * self.n_head)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs):
        """
            inputs = i_vectors
            i_vectors.shape: (seq_length, input_size) # input_size = n_head * d_input
        """
        i_vectors = inputs
        seq_length = i_vectors.size(0)

        # input linear projection
        i_vectors = self.input_linear(i_vectors)
        # split
        i_vectors = i_vectors.chunk(self.n_head, 1)
        # hplstm
        o_vectors = []
        for i, hplstmmodule in enumerate(self.hplstm_list):
            o_vectors_ith = hplstmmodule(i_vectors[i])
            o_vectors.append(o_vectors_ith)

        # concat
        o_vectors = torch.cat(o_vectors, dim=1)
        # output linear projection
        o_vectors = self.output_linear(o_vectors)

        return o_vectors