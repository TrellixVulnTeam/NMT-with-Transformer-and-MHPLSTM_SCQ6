import BagOfWords
import torch
import torch.nn as nn

class HPLSTM(nn.Module):
    """Highly Parallelized LSTM"""
    def __init__(self, tokenizer, vocab_size, d_input, d_hidden, d_output, device):
        super(HPLSTM, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.device = device
        self.params = self.init_parameters(d_input, d_hidden, d_output, device)

    def init_parameters(self, d_input, d_hidden, d_output, device):
        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        def three():
            return normal((d_input, d_hidden)), normal(d_hidden, d_hidden), torch.zeros(d_hidden, device=device)

        W_vi, W_hi, b_i = three()
        W_vf, W_hf, b_f = three()
        W_vh1, W_hh1, b_h1 = three()
        W_vh2, W_hh2, b_h2 = three()
        W_hq = torch.randn((d_hidden, d_output), device=device)
        b_q = torch.zeros(d_output, device=device)

        params = [W_vi, W_hi, b_i, W_vf, W_hf, b_f, W_vh1, W_hh1, b_h1, W_vh2, W_hh2, b_h2, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)

        return params

    def forward_1half(self, inputs):
        bag_of_words = self.bag_of_words
        inputs = 2