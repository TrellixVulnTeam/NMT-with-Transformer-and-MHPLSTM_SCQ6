import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class LSTMCell(nn.Module):
    """LSTMCell for consideration"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size)
        #       cy: of shape (batch_size, hidden_size)
        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))
            hx = (hx, hx)

        hx, cx = hx

        gates = self.xh(input) + self.hh(hx)

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t
        hy = o_t * torch.tanh(cy)

        return (hy, cy)


class LSTM(nn.Module):
    """Complete LSTM module for consideration"""
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):
        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
            h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l
            outs.append(hidden_l[0])

        out = outs[-1].squeeze()
        out = self.fc(out)

        return out


class BidirRecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(BidirRecurrentModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias))

        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hx=None):
        # Input of shape (batch_size, sequence length, input_size)
        #
        # Output of shape (batch_size, output_size)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            hT = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
            hT = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        outs = []
        outs_rev = []

        hidden_forward = list()
        for layer in range(self.num_layers):
            hidden_forward.append((h0[layer, :, :], h0[layer, :, :]))

        hidden_backward = list()
        for layer in range(self.num_layers):
            hidden_backward.append((hT[layer, :, :], hT[layer, :, :]))

        for t in range(input.shape[1]):
            for layer in range(self.num_layers):
                if layer == 0:
                    # Forward net
                    h_forward_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden_forward[layer][0], hidden_forward[layer][1])
                        )
                    # Backward net
                    h_back_l = self.rnn_cell_list[layer](
                        input[:, -(t + 1), :],
                        (hidden_backward[layer][0], hidden_backward[layer][1])
                        )
                else:
                    # Forward net
                    h_forward_l = self.rnn_cell_list[layer](
                        hidden_forward[layer - 1][0],
                        (hidden_forward[layer][0], hidden_forward[layer][1])
                        )
                    # Backward net
                    h_back_l = self.rnn_cell_list[layer](
                        hidden_backward[layer - 1][0],
                        (hidden_backward[layer][0], hidden_backward[layer][1])
                        )

                hidden_forward[layer] = h_forward_l
                hidden_backward[layer] = h_back_l

            outs.append(h_forward_l[0])
            outs_rev.append(h_back_l[0])

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()
        out_rev = outs_rev[0].squeeze()
        out = torch.cat((out, out_rev), 1)
        out = self.fc(out)

        return out


# Testing
X_one_time_step = torch.rand((16, 128))
lstm_cell = LSTMCell(128, 256)
h_one_time_step, c_one_time_step = lstm_cell(X_one_time_step)
print("X_one_time_step.shape: ", X_one_time_step.shape)
print("h_one_time_step.shape: ", h_one_time_step.shape)
print("c_one_time_step.shape: ", c_one_time_step.shape)

X_1way = torch.rand((16, 256, 128))
lstm = LSTM(128, 256, 12, True, 10)
output_1way = lstm(X_1way)
print("X_1way.shape: ", X_1way.shape)
print("output_1way.shape: ", output_1way.shape)

X = torch.rand((16, 256, 128))
bi_lstm = BidirRecurrentModel(128, 256, 12, True, 10)
output = bi_lstm(X)
print("X.shape: ", X.shape)
print("output.shape: ", output.shape)